# ============================================================================
# Download & Save QPU Noise Model + Coupling Map + Best Layout
# ============================================================================
# This script connects to IBM Quantum, downloads the full backend data from a
# real QPU (ibm_kingston by default), and saves it locally so all VQE scripts
# can run offline — exactly like the original em_vqe_h2_*.py tagged files.
#
# Output files (in noise_model/ subfolder):
#   - noise_model.pkl         : Serialized NoiseModel (pickle)
#   - coupling_map.pkl        : Backend coupling map (for transpilation)
#   - instruction_durations.pkl: Gate durations (for DD scheduling)
#   - backend_properties.json : Readout errors, gate errors, best layout
#   - model_info.txt          : Human-readable metadata summary
# ============================================================================

import os
import json
import pickle
import datetime
import numpy as np

import mapomatic as mm
from qiskit import transpile
from qiskit.circuit.library import efficient_su2
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.transpiler import InstructionDurations

# ─── Configuration ──────────────────────────────────────────────────────────
BACKEND_NAME = "ibm_kingston"
FALLBACK_BACKENDS = ["ibm_fez", "ibm_marrakesh"]
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "noise_model")

# ─── Connect to IBM Quantum ────────────────────────────────────────────────
print("=" * 70)
print("NOISE MODEL DOWNLOADER")
print("=" * 70)

service = QiskitRuntimeService()
print(f"Connected to IBM Quantum service")

# Try to get the requested backend, with fallbacks
backend = None
tried = []
for name in [BACKEND_NAME] + FALLBACK_BACKENDS:
    try:
        backend = service.backend(name)
        print(f"✓ Connected to backend: {backend.name}")
        break
    except Exception as e:
        tried.append(name)
        print(f"✗ Backend '{name}' not available: {e}")

if backend is None:
    raise RuntimeError(f"Could not connect to any backend. Tried: {tried}")

# ─── Extract Noise Model ───────────────────────────────────────────────────
print(f"\nExtracting noise model from {backend.name}...")
noise_model = NoiseModel.from_backend(backend)
print(f"  Noise model extracted successfully")
print(f"  Basis gates: {noise_model.basis_gates}")
print(f"  Noise qubits: {noise_model.noise_qubits}")

# ─── Build local simulator (same as tagged files) ──────────────────────────
print(f"\nBuilding AerSimulator.from_backend({backend.name})...")
backend_sim = AerSimulator.from_backend(backend)
backend_sim.set_options(method="density_matrix")
print(f"  Local simulator ready: {backend_sim.name}")

# ─── Extract Coupling Map ──────────────────────────────────────────────────
print(f"\nExtracting coupling map...")
coupling_map = backend_sim.configuration().coupling_map if hasattr(backend_sim, 'configuration') else None
if coupling_map is None:
    try:
        coupling_map = backend.coupling_map.get_edges()
    except:
        coupling_map = None
print(f"  Coupling map: {'extracted' if coupling_map else 'not available'}")

# ─── Extract Instruction Durations ──────────────────────────────────────────
print(f"\nExtracting instruction durations...")
try:
    durations = InstructionDurations.from_backend(backend)
    print(f"  Instruction durations extracted successfully")
except Exception as e:
    print(f"  Could not extract durations: {e}")
    # Fallback: try from target
    try:
        durations = backend.target.durations()
        print(f"  Instruction durations from target extracted successfully")
    except:
        durations = None
        print(f"  No duration info available — DD will use defaults")

# ─── Run Mapomatic for Best Layout ─────────────────────────────────────────
print(f"\nRunning mapomatic for best qubit layout...")
n_qubits = 2
reps = 1
ansatz = efficient_su2(n_qubits, entanglement="pairwise", reps=reps)

rng = np.random.default_rng(42)
test_params = rng.uniform(-np.pi, np.pi, size=ansatz.num_parameters)
test_circ = ansatz.assign_parameters(test_params)
dummy_circ = transpile(test_circ, backend, optimization_level=3)
small_circ = mm.deflate_circuit(dummy_circ)
layouts = mm.matching_layouts(small_circ, backend)
scores = mm.evaluate_layouts(small_circ, layouts, backend)
best_layout = list(scores[0][0])

print(f"  Best physical qubits (mapomatic): {best_layout}")
print(f"  Top 5 layouts: {scores[:5]}")

# ─── Extract Backend Properties ─────────────────────────────────────────────
print(f"\nExtracting backend properties...")
properties = backend.properties()

# Collect readout errors
readout_errors = {}
if properties is not None:
    for q in range(backend.num_qubits):
        try:
            re = properties.readout_error(q)
            readout_errors[str(q)] = re
        except Exception:
            pass

# Collect gate errors
gate_errors = {}
if properties is not None:
    for gate in properties.gates:
        key = f"{gate.gate}_{gate.qubits}"
        for param in gate.parameters:
            if param.name == "gate_error":
                gate_errors[key] = param.value

# Collect T1, T2 times
qubit_properties = {}
if properties is not None:
    for q in range(backend.num_qubits):
        try:
            t1 = properties.t1(q)
            t2 = properties.t2(q)
            qubit_properties[str(q)] = {"T1": t1, "T2": t2}
        except Exception:
            pass

# ─── Save Everything ────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 1. Full AerSimulator (pickle) — preserves coupling map, target, config, everything
backend_sim_path = os.path.join(OUTPUT_DIR, "backend_sim.pkl")
with open(backend_sim_path, "wb") as f:
    pickle.dump(backend_sim, f)
print(f"\n✓ AerSimulator saved: {backend_sim_path}")

# 2. Instruction Durations (pickle) — needed for DD scheduling
durations_path = os.path.join(OUTPUT_DIR, "instruction_durations.pkl")
with open(durations_path, "wb") as f:
    pickle.dump(durations, f)
print(f"✓ Instruction durations saved: {durations_path}")

# 4. Backend Properties JSON (includes best_layout from mapomatic)
props_path = os.path.join(OUTPUT_DIR, "backend_properties.json")
props_data = {
    "backend_name": backend.name,
    "num_qubits": backend.num_qubits,
    "best_layout": best_layout,
    "readout_errors": readout_errors,
    "gate_errors": gate_errors,
    "qubit_properties": qubit_properties,
    "download_timestamp": timestamp,
}
with open(props_path, "w") as f:
    json.dump(props_data, f, indent=2, default=str)
print(f"✓ Backend properties saved: {props_path}")

# 5. Human-readable info file
info_path = os.path.join(OUTPUT_DIR, "model_info.txt")
with open(info_path, "w") as f:
    f.write("=" * 70 + "\n")
    f.write("NOISE MODEL INFORMATION\n")
    f.write("=" * 70 + "\n")
    f.write(f"Backend Name:       {backend.name}\n")
    f.write(f"Number of Qubits:   {backend.num_qubits}\n")
    f.write(f"Download Timestamp: {timestamp}\n")
    f.write(f"Basis Gates:        {noise_model.basis_gates}\n")
    f.write(f"Noise Qubits:       {noise_model.noise_qubits}\n")
    f.write(f"Best Layout (mapomatic): {best_layout}\n")
    f.write(f"\n{'=' * 70}\n")
    f.write("READOUT ERRORS\n")
    f.write(f"{'=' * 70}\n")
    for q, err in sorted(readout_errors.items(), key=lambda x: int(x[0])):
        f.write(f"  Qubit {q}: {err:.6f}\n")
    f.write(f"\n{'=' * 70}\n")
    f.write("QUBIT T1/T2 TIMES (seconds)\n")
    f.write(f"{'=' * 70}\n")
    for q, props_q in sorted(qubit_properties.items(), key=lambda x: int(x[0])):
        f.write(f"  Qubit {q}: T1 = {props_q['T1']:.6e} s, T2 = {props_q['T2']:.6e} s\n")
    f.write(f"\n{'=' * 70}\n")
    f.write(f"GATE ERRORS (sample)\n")
    f.write(f"{'=' * 70}\n")
    for gate, err in sorted(gate_errors.items())[:20]:
        f.write(f"  {gate}: {err:.6e}\n")
    if len(gate_errors) > 20:
        f.write(f"  ... and {len(gate_errors) - 20} more gates\n")

print(f"✓ Model info saved: {info_path}")

# ─── Summary ────────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("DOWNLOAD COMPLETE")
print(f"{'=' * 70}")
print(f"Backend:          {backend.name}")
print(f"Qubits:           {backend.num_qubits}")
print(f"Best Layout:      {best_layout} (mapomatic)")
print(f"Readout errors:   {len(readout_errors)} qubits")
print(f"Gate errors:      {len(gate_errors)} gates")
print(f"Coupling map:     {'saved' if coupling_map else 'not available'}")
print(f"Durations:        {'saved' if durations else 'not available'}")
print(f"Files saved to:   {OUTPUT_DIR}")
print(f"\nYou can now run the VQE scripts offline using this noise model!")
