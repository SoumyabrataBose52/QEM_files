# ============================================================================
# VQE for H₂ — Dynamical Decoupling (DD) Error Mitigation
# ============================================================================
# Runs VQE with DD applied during idle qubit times. DD inserts X-X gate pairs
# to suppress decoherence (T1/T2 relaxation) errors. Uses saved noise model,
# coupling map, instruction durations, and mapomatic layout.
#
# Matches the pattern from the original em_vqe_h2_dd.py tagged file.
#
# Prerequisites: Run download_noise_model.py first
# ============================================================================

import os
import sys
import json
import pickle
import datetime
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import efficient_su2, XGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler import PassManager, InstructionDurations
from qiskit.transpiler.passes import PadDynamicalDecoupling, ALAPScheduleAnalysis
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.primitives import BackendEstimatorV2

# ─── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NOISE_MODEL_DIR = os.path.join(SCRIPT_DIR, "noise_model")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Load Noise Model + Coupling Map + Durations ───────────────────────────
print("=" * 70)
print("VQE — DYNAMICAL DECOUPLING (DD)")
print("=" * 70)

noise_model_path = os.path.join(NOISE_MODEL_DIR, "backend_sim.pkl")
durations_path = os.path.join(NOISE_MODEL_DIR, "instruction_durations.pkl")
props_path = os.path.join(NOISE_MODEL_DIR, "backend_properties.json")

if not os.path.exists(noise_model_path):
    print(f"ERROR: Backend simulator not found at {noise_model_path}")
    print("Please run download_noise_model.py first!")
    sys.exit(1)

with open(noise_model_path, "rb") as f:
    backend_sim = pickle.load(f)
with open(durations_path, "rb") as f:
    instruction_durations = pickle.load(f)
with open(props_path, "r") as f:
    backend_props = json.load(f)

backend_name = backend_props["backend_name"]
best_layout = backend_props["best_layout"]

print(f"Loaded AerSimulator from: {backend_name}")
print(f"Local simulator ready: {backend_sim.name}")
print(f"Best layout (mapomatic): {best_layout}")

# ─── H₂ Hamiltonian ────────────────────────────────────────────────────────
H2_op = SparsePauliOp.from_list([
    ("II", -1.052373245772859),
    ("IZ",  0.39793742484318045),
    ("ZI", -0.39793742484318045),
    ("ZZ", -0.01128010425623538),
    ("XX",  0.18093119978423156),
])
observable = H2_op

print(f"\nObservable defined: {observable.num_qubits} qubits")
for pauli, coeff in zip(H2_op.paulis, H2_op.coeffs):
    print(f"  Term: {pauli}  Coeff: {coeff.real:+.15f}")

A = np.array(H2_op.to_matrix())
eigenvalues, eigenvectors = np.linalg.eigh(A)
exact_ground_state_energy = min(eigenvalues).real
print(f"\nAll eigenvalues: {np.round(eigenvalues.real, 6)}")
print(f"Exact ground state energy: {exact_ground_state_energy:.6f} Hartree")

ideal_state = Statevector.from_label('00')
exact_00_val = ideal_state.expectation_value(observable).real
print(f"Exact |00⟩ expectation value: {exact_00_val:.5f}")

# ─── Ansatz ─────────────────────────────────────────────────────────────────
n_qubits = 2
reps = 1
ansatz = efficient_su2(n_qubits, entanglement="pairwise", reps=reps)

print(f"\nAnsatz: EfficientSU2")
print(f"  Qubits: {n_qubits}")
print(f"  Reps: {reps}")
print(f"  Num parameters: {ansatz.num_parameters}")
print(f"  Depth: {ansatz.depth()}")

# ─── ISA Transpilation ─────────────────────────────────────────────────────
target = backend_sim.target
pm_gen = generate_preset_pass_manager(target=target, optimization_level=3)
ansatz_isa = pm_gen.run(ansatz)

print(f"\nBest physical qubits (mapomatic): {best_layout}")
print(f"ISA ansatz depth: {ansatz_isa.depth()}")
print(f"ISA ansatz gates: {dict(ansatz_isa.count_ops())}")

hamiltonian_isa = observable.apply_layout(ansatz_isa.layout)
print(f"Hamiltonian mapped to ISA layout successfully")

# ─── DD Helper ──────────────────────────────────────────────────────────────
def apply_local_dd(circuit, durations, active_qubits=None):
    """Apply Dynamical Decoupling using XpXm (X-X) sequence."""
    pm = PassManager([
        ALAPScheduleAnalysis(durations),
        PadDynamicalDecoupling(durations, [XGate(), XGate()], qubits=active_qubits)
    ])
    return pm.run(circuit)

print("\nDD helper loaded!")
dd_sequence = "X-X (XpXm)"
print(f"DD Sequence: {dd_sequence}")

# ─── VQE Cost Function ─────────────────────────────────────────────────────
dd_circuit_depths = []

def vqe_cost_func(params, ansatz_isa, hamiltonian_isa, estimator, cost_history_dict,
                   best_layout, observable, aer_backend, ansatz_original, durations):
    """VQE cost function with Dynamical Decoupling."""
    bound_circ = ansatz_original.assign_parameters(params)
    t_circ = transpile(bound_circ, aer_backend, initial_layout=best_layout, optimization_level=3)
    dd_circ = apply_local_dd(t_circ, durations, active_qubits=best_layout)
    obs_mapped = observable.apply_layout(t_circ.layout)
    energy = float(estimator.run([(dd_circ, obs_mapped)]).result()[0].data.evs)

    dd_circuit_depths.append(dd_circ.depth())

    cost_history_dict["iters"] += 1
    cost_history_dict["prev_vector"] = params
    cost_history_dict["cost_history"].append(energy)
    cost_history_dict["param_history"].append(params.copy())
    print(f"Iters. done: {cost_history_dict['iters']} [Current cost: {energy:.6f}] [DD depth: {dd_circ.depth()}]")
    print(f"  Parameters: {params}")
    return energy

# ─── Run VQE ────────────────────────────────────────────────────────────────
shots = 10000
precision_val = 1 / np.sqrt(shots)
max_iter = 250

aer_estimator = BackendEstimatorV2(backend=backend_sim, options={"default_precision": precision_val})

rng_init = np.random.default_rng(1234)
x0 = rng_init.uniform(-np.pi, np.pi, size=ansatz_isa.num_parameters)

print(f"\n{'=' * 70}")
print("STARTING VQE OPTIMIZATION — DYNAMICAL DECOUPLING")
print(f"{'=' * 70}")
print(f"Initial parameters: {np.round(x0, 4)}")
print(f"Number of parameters: {len(x0)}")
print(f"Shots per circuit: {shots}")
print(f"Max iterations: {max_iter}")
print(f"Exact ground state energy: {exact_ground_state_energy:.6f}")
print(f"DD Sequence: {dd_sequence}")
print(f"Active Qubits: {best_layout}")
print(f"{'=' * 70}")

cost_history = {"prev_vector": None, "iters": 0, "cost_history": [], "param_history": []}

res = minimize(
    vqe_cost_func,
    x0,
    args=(ansatz_isa, hamiltonian_isa, aer_estimator, cost_history,
          best_layout, observable, backend_sim, ansatz, instruction_durations),
    method="cobyla",
    options={"maxiter": max_iter},
)

converged_energy = cost_history["cost_history"][-1]
converged_params = cost_history["param_history"][-1]
abs_error = abs(converged_energy - exact_ground_state_energy)
pct_error = abs(abs_error / exact_ground_state_energy) * 100

print(f"\n{'=' * 70}")
print("VQE COMPLETE — DYNAMICAL DECOUPLING")
print(f"{'=' * 70}")
print(f"  Converged Energy: {converged_energy:.6f}")
print(f"  Exact Energy:     {exact_ground_state_energy:.6f}")
print(f"  Absolute Error:   {abs_error:.6f}")
print(f"  Percentage Error: {pct_error:.2f}%")
print(f"  Iterations:       {cost_history['iters']}")
print(f"  Optimizer Success:{res.success}")
print(f"  Message:          {res.message}")

# ─── Save Results ───────────────────────────────────────────────────────────
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = os.path.join(RESULTS_DIR, f"results_dd_{timestamp}.txt")

with open(results_file, "w") as f:
    f.write("=" * 70 + "\n")
    f.write("VQE RESULTS: DYNAMICAL DECOUPLING (DD)\n")
    f.write("=" * 70 + "\n")
    f.write(f"Backend: {backend_name} (noise model)\n")
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"Shots: {shots}\n")
    f.write(f"Max Iterations: {max_iter}\n")
    f.write(f"Initial Seed: 1234\n\n")

    f.write("=" * 70 + "\n")
    f.write("HAMILTONIAN\n")
    f.write("=" * 70 + "\n")
    for pauli, coeff in zip(H2_op.paulis, H2_op.coeffs):
        f.write(f"Term: {pauli}  Coeff: {coeff.real:+.15f}\n")
    f.write(f"\nEigenvalues: {np.round(eigenvalues.real, 6).tolist()}\n")
    f.write(f"Exact Ground State Energy: {exact_ground_state_energy:.6f} Hartree\n")
    f.write(f"Exact |00> Expectation Value: {exact_00_val:.5f}\n\n")

    f.write("=" * 70 + "\n")
    f.write("ANSATZ CIRCUIT\n")
    f.write("=" * 70 + "\n")
    f.write(f"Type: EfficientSU2\n")
    f.write(f"Qubits: {n_qubits}\n")
    f.write(f"Reps: {reps}\n")
    f.write(f"Num Parameters: {ansatz.num_parameters}\n")
    f.write(f"Ansatz Depth: {ansatz.depth()}\n\n")

    f.write("=" * 70 + "\n")
    f.write("ISA TRANSPILATION\n")
    f.write("=" * 70 + "\n")
    f.write(f"ISA Depth: {ansatz_isa.depth()}\n")
    f.write(f"ISA Gates: {dict(ansatz_isa.count_ops())}\n")
    f.write(f"Best Layout (mapomatic): {best_layout}\n")
    f.write(f"Hamiltonian mapped to ISA: True\n\n")

    f.write("=" * 70 + "\n")
    f.write("EM TECHNIQUE CONFIG\n")
    f.write("=" * 70 + "\n")
    f.write(f"Technique: Dynamical Decoupling (DD)\n")
    f.write(f"DD Sequence: {dd_sequence}\n")
    f.write(f"Active Qubits: {best_layout}\n")
    avg_dd_depth = np.mean(dd_circuit_depths) if dd_circuit_depths else 0
    f.write(f"Average DD Circuit Depth: {avg_dd_depth:.1f}\n\n")

    f.write("=" * 70 + "\n")
    f.write("INITIAL PARAMETERS\n")
    f.write("=" * 70 + "\n")
    f.write(f"x0 = {x0.tolist()}\n\n")

    f.write("=" * 70 + "\n")
    f.write("ITERATION HISTORY\n")
    f.write("=" * 70 + "\n")
    for i, (e, p, d) in enumerate(zip(
        cost_history["cost_history"], cost_history["param_history"], dd_circuit_depths)):
        f.write(f"Iter {i+1:4d} | Energy: {e:+.10f} | DD_Depth: {d} | Params: {p.tolist()}\n")

    f.write(f"\n{'=' * 70}\n")
    f.write("FINAL SUMMARY\n")
    f.write("=" * 70 + "\n")
    f.write(f"Converged Energy:   {converged_energy:.10f} Hartree\n")
    f.write(f"Exact Energy:       {exact_ground_state_energy:.10f} Hartree\n")
    f.write(f"Absolute Error:     {abs_error:.10f}\n")
    f.write(f"Percentage Error:   {pct_error:.4f}%\n")
    f.write(f"Total Iterations:   {cost_history['iters']}\n")
    f.write(f"Optimizer Success:  {res.success}\n")
    f.write(f"Optimizer Message:  {res.message}\n")
    f.write(f"Final Parameters:   {converged_params.tolist()}\n")

print(f"\n✓ Results saved to: {results_file}")
