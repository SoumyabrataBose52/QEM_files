# Quantum Error Mitigation for Quantum Algorithms — Project Results Timeline

> **Project Title:** Quantum Error Mitigation for Quantum Algorithms  
> **Scope:** H₂ Hamiltonian · EfficientSU2 Ansatz · VQE · Real QPUs & Local Noise Models  
> **Key Backends Used:** ibm_fez (156 qubits) · ibm_torino · ibm_marrakesh · ibm_kingston (133 qubits) · FakeLimaV2 · AerSimulator

---

## Stage 1 — Baseline: Local Simulations with Error Mitigation

**Files:** *(Conceptual baseline — not separately saved as a notebook)*  
**Goal:** Establish a clean baseline of Error Mitigation (EM) techniques on ideal and noisy local simulators before touching real QPUs.

### What Was Done
- Set up the **H₂ Hamiltonian** as a `SparsePauliOp` with 5 Pauli terms:
  - `II`, `IZ`, `ZI`, `ZZ`, `XX`
  - Exact ground-state energy (NumPy reference): **−1.85728 Hartree**
- Used **EfficientSU2** (also `n_local` with `ry` rotation blocks and `cz` entanglement) as the ansatz circuit.
- First executed on **AerSimulator** (ideal statevector / density matrix simulation).
- Then repeated using **FakeLimaV2** — a fake backend that simulates a 5-qubit IBM QPU noise profile.
- Applied Qiskit's built-in EM options (where available) to compare noisy vs. mitigated results.

### Key Results
- Ideal (noiseless) AerSimulator closely matches exact ground state energy.
- FakeLimaV2 introduced realistic gate and readout noise, degrading accuracy.
- Built-in EM via `EstimatorOptions` worked cleanly at this stage since FakeLimaV2 is compatible.

---

## Stage 2 — VQE on FakeLimaV2: Optimizer Comparison

**File:** [`fakelima_opti_diff.ipynb`](file:///d:/minor_report/fakelima_opti_diff.ipynb)  
**Goal:** Run VQE on a fake (noisy) backend and compare different classical optimizers.

### What Was Done
- Backend: `FakeLimaV2` (fake IBM 5-qubit Lima backend)
- Ansatz: `n_local(2, rotation_blocks="ry", entanglement_blocks="cz")`
- Compared classical optimizers: **SPSA**, **SLSQP**, **COBYLA**
- Used `AerEstimator` (Qiskit Aer's EstimatorV2) as the primitive
- Transpiled the circuit for FakeLimaV2

### Key Results
- Different optimizers show different convergence behavior under noise.
- SPSA tends to be more noise-robust due to its stochastic gradient nature.
- SLSQP and COBYLA converge faster in ideal conditions but can be more sensitive to noise perturbations.
- Showed optimizer choice matters when doing VQE under realistic hardware conditions.

---

## Stage 3 — Error Mitigation on Real QPUs (ibm_fez, ibm_torino, ibm_marrakesh, ibm_kingston)

**File:** [`Combine_EM (1).ipynb`](file:///d:/minor_report/Combine_EM%20%281%29.ipynb)  
**Goal:** Apply all 4 major EM techniques directly on real IBM QPUs and compare results against ideal values.

### What Was Done
- Connected to **real IBM Quantum backends** using `QiskitRuntimeService`:
  - `ibm_fez` (156 qubits)
  - `ibm_torino`
  - `ibm_marrakesh`
  - `ibm_kingston` (133 qubits)
- Used **EfficientSU2** ansatz, transpiled with `generate_preset_pass_manager` at optimization level 3.
- Ran in **Batch mode** (`Batch` + `EstimatorV2`) for efficiency.
- Applied 4 EM techniques via `EstimatorOptions`:
  1. **No Mitigation** (baseline)
  2. **Dynamical Decoupling (DD)**
  3. **Pauli Twirling**
  4. **TREX** (Twirled Readout Error eXtinction) — `resilience_level=1`
  5. **ZNE** (Zero-Noise Extrapolation) — `resilience_level=2`
- Documented results from IBM Quantum Platform runs and compared across QPUs and EM methods.

### Key Results
- **TREX** showed consistent readout error correction across all QPUs.
- **ZNE** was most effective for gate-noise-dominated regimes but added circuit overhead.
- **DD** helped primarily when qubits had significant idle time between gates.
- **Pauli Twirling** converted coherent errors to stochastic noise, aiding averaging.
- Results were compared with the exact ground state (−1.85728 Hartree) from NumPy.

---

## Stage 4 — Noise Model Extraction & QPU-Like Local Simulation

**Files:** [`test_01_DD (2).ipynb`](file:///d:/minor_report/test_01_DD%20%282%29.ipynb), [`test_02_TREX (2).ipynb`](file:///d:/minor_report/test_02_TREX%20%282%29.ipynb), [`test_03_Twirling (1).ipynb`](file:///d:/minor_report/test_03_Twirling%20%281%29.ipynb), [`test_04_ZNE (2).ipynb`](file:///d:/minor_report/test_04_ZNE%20%282%29.ipynb)

**Goal:** Instead of using generic fake backends, **fetch the real noise model** from an actual QPU and simulate locally, faithfully imitating the 156/133-qubit backend architecture.

### What Was Done
- Fetched noise model from **ibm_fez** (156 qubits):
  ```python
  real_backend = service.backend("ibm_fez")
  aer_backend = AerSimulator.from_backend(real_backend)
  aer_backend.set_options(method="density_matrix")
  ```
- **Problem Discovered:** Running local simulation on a 156-qubit backend was computationally infeasible for density matrix simulation — memory and compute exploded.
- **Solution — Mapomatic:** Used `mapomatic` library to find the **optimal 2-qubit subgraph** of the 156-qubit topology that best matches the ansatz circuit requirements.
  ```python
  dummy_circ = transpile(mirror_circuit, real_backend, optimization_level=3)
  small_circ = mm.deflate_circuit(dummy_circ)
  best_layout = mm.best_overall_layout(small_circ, aer_backend)
  # Result: best physical qubits [131, 132] on ibm_fez
  ```
- From this point on, **mapomatic was used everywhere** for QPU-like local simulations.
- Applied all 5 EM modes as separate test notebooks:
  1. **No Mitigation** (baseline)
  2. **DD** — `test_01_DD (2).ipynb`
  3. **TREX** — `test_02_TREX (2).ipynb`
  4. **Pauli Twirling** — `test_03_Twirling (1).ipynb`
  5. **ZNE** — `test_04_ZNE (2).ipynb`

### Critical Insight
> Qiskit's built-in EM techniques (via `EstimatorOptions`) are **cloud-only** — they only work with real IBM QPU sessions, NOT with local `AerSimulator`. To apply EM on local noise-model simulations, **custom implementations had to be built from scratch** (see Stage 5).

### Mapomatic Results (ibm_fez)
- Best physical qubit pair selected: **[131, 132]**
- This allowed the 156-qubit backend to be simulated locally with realistic noise, without a memory explosion.

---

## Stage 5 — Custom Local EM Implementations (Core Research Contribution)

**Files:** [`em_vqe_nm_fez_h2/`](file:///d:/minor_report/em_vqe_nm_fez_h2), [`em_vqe_nm_kingston_h2/`](file:///d:/minor_report/em_vqe_nm_kingston_h2)  
**Goal:** Since IBM's cloud EM API is unavailable locally, build **custom EM functions from scratch** for use with local `AerSimulator` + noise model.

### The Core Problem
| Cloud API (Unavailable Locally) | Custom Local Implementation |
|---|---|
| `options.resilience.zne_mitigation = True` | `fold_local(circuit, scale)` |
| `options.dynamical_decoupling.enable = True` | `apply_local_dd(circuit, backend)` |
| `options.resilience.measure_mitigation = True` | `apply_local_trex_to_observable(...)` |
| `options.twirling.enable_gates = True` | `apply_pauli_twirling(circuit, ...)` |

This required **multiple iterations of research, failure analysis, bug fixes, and refinements** to make each local function approximate the original cloud technique as closely as possible.

### Custom Functions Built

#### 1. Dynamical Decoupling (DD) — `vqe_dd.py`
- **Function:** `apply_local_dd(circuit, backend, active_qubits=None)`
- Uses Qiskit's `ALAPScheduleAnalysis` + `PadDynamicalDecoupling` transpiler passes
- Inserts X-X (XpXm) pairs into idle qubit slots
- Targets only the **best layout qubits** (from mapomatic)
- **Best for:** Circuits with significant idle time; T1/T2 decoherence suppression

#### 2. TREX (Twirled Readout Error eXtinction) — `vqe_trex.py`
- **Function:** `apply_local_trex_to_observable(observable_2q, readout_errors_dict, layout)`
- Reads readout error rates `p0`, `p1` from **saved** `backend_properties.json`
- For each non-Identity Pauli term, scales coefficients by `1 / (1 - 2p)` per qubit
- **Mathematical basis:** If `⟨O⟩_noisy = (1 - 2p) × ⟨O⟩_ideal`, then `⟨O⟩_mitigated = ⟨O⟩_noisy / (1 - 2p)`
- Zero overhead (no extra circuits — only modifies the observable)
- **Best for:** Backends with readout errors > 1%

#### 3. Zero-Noise Extrapolation (ZNE) — `vqe_zne.py`
- **Function:** `fold_local(circuit, scale)`
- Folds circuits at scales **[1, 3, 5]**: `C`, `C·C†·C`, `C·C†·C·C†·C`
- Runs each folded circuit through AerEstimator and collects energies
- Fits a **linear extrapolation** (polyfit degree 1) to extrapolate to zero noise
- Stores per-scale energies + fit slope/intercept per VQE iteration
- **Best for:** Gate-noise-dominated regimes with approximately linear noise scaling
- **Overhead:** 3× circuit evaluations per iteration

#### 4. Pauli Twirling — `vqe_twirling.py`
- **Function:** `apply_pauli_twirling(circuit, num_twirls=10, seed=42)`
- Generates **10 randomized copies** by wrapping each 2-qubit gate with random Pauli pairs
- Uses Qiskit's `Clifford` class for exact algebraic conjugation
- Averages all 10 results — coherent errors cancel, stochastic errors average out
- Re-transpiles twirled circuits at optimization_level=1
- **Best for:** Circuits with dominant coherent (systematic) 2-qubit gate errors
- **Overhead:** 10× circuit evaluations per iteration

#### 5. No Mitigation Baseline — `vqe_no_mitigation.py`
- Plain VQE without any EM applied
- Used as baseline for all comparisons

### Supporting Scripts
- [`download_noise_model.py`](file:///d:/minor_report/em_vqe_nm_fez_h2/download_noise_model.py) — One-time script to fetch and cache noise model + backend properties from real QPU
- [`visualize_results.py`](file:///d:/minor_report/em_vqe_nm_fez_h2/visualize_results.py) — Comprehensive comparison plots across all 5 EM modes
- [`EM_TECHNIQUES_DOCUMENTATION.md`](file:///d:/minor_report/em_vqe_nm_fez_h2/EM_TECHNIQUES_DOCUMENTATION.md) — Full technical documentation of all local implementations

---

## Stage 6 — VQE with Local Noise Model (ibm_fez): EM Comparison

**Files:**  
- [`vqe_local_nM_fez (2).ipynb`](file:///d:/minor_report/vqe_local_nM_fez%20%282%29.ipynb) — Single backend VQE with fez noise model  
- [`em_vqe_nm_fez_h2/`](file:///d:/minor_report/em_vqe_nm_fez_h2) — Full 5-technique EM suite for fez noise model

**Goal:** Run full VQE with each EM technique on the **ibm_fez noise model** locally, compare convergence and final energy.

### Configuration
- Hamiltonian: H₂ (`SparsePauliOp`, 5 terms)
- Ansatz: `n_local(2, rotation_blocks="ry", entanglement_blocks="cz")`
- Optimizer: **SPSA** (125 iterations)
- Backend: `AerSimulator.from_backend(ibm_fez)` with density matrix method
- Qubit layout: Best 2-qubit subgraph via **mapomatic** (qubits [131, 132])
- EM modes tested: No Mitigation, DD, TREX, ZNE, Pauli Twirling

### Key Results
- **TREX** and **ZNE** showed the most improvement over baseline.
- **DD** provided modest improvement when idle time was present in the transpiled circuit.
- **Pauli Twirling** helped in reducing variance and coherent errors.
- All EM techniques consistently brought results **closer to the exact ground state** (−1.85728 Hartree).

---

## Stage 7 — VQE with Local Noise Model: Multi-Backend Comparison (SPSA)

**File:** [`vqe_comp_nM (2).ipynb`](file:///d:/minor_report/vqe_comp_nM%20%282%29.ipynb)  
**File:** [`vqe_backend_comparison (1).csv`](file:///d:/minor_report/vqe_backend_comparison%20%281%29.csv)

**Goal:** Compare VQE performance across **multiple backend noise models** with SPSA optimizer.

### Configuration
- Same H₂ Hamiltonian and ansatz as Stage 6
- SPSA, 125 iterations, fixed seed 170 for reproducibility
- Backends compared: ibm_fez, ibm_kingston (and others via noise model)
- Used `AerEstimator` with fetched noise models

### Key Results
- Different backends have different noise profiles, leading to different convergence trajectories.
- Backends with lower gate error rates (better-calibrated qubits) converge faster and closer to ideal.
- Data saved to `vqe_backend_comparison (1).csv` for further analysis.

---

## Stage 8 — VQE with Local Noise Model: Multi-Backend Comparison (COBYLA)

**File:** [`vqe_comp_nM-Cobyla.ipynb`](file:///d:/minor_report/vqe_comp_nM-Cobyla.ipynb)  
**File:** [`vqe_backend_comparison_cobyla (1).csv`](file:///d:/minor_report/vqe_backend_comparison_cobyla%20%281%29.csv)

**Goal:** Repeat Stage 7 with **COBYLA** optimizer to compare optimizer behavior under noise.

### Configuration
- Same backends and Hamiltonian
- COBYLA optimizer with up to 250 iterations
- COBYLA converges around < 100 iterations in most runs

### Key Results
- COBYLA converged faster than SPSA in noise-free conditions.
- Under noise, COBYLA was more sensitive to fluctuations in energy estimates.
- Useful for understanding optimizer-noise interaction in VQE.

---

## Stage 9 — Full VQE + EM Suite on Local ibm_fez Noise Model

**Directory:** [`em_vqe_nm_fez_h2/`](file:///d:/minor_report/em_vqe_nm_fez_h2)

**Goal:** Run the complete 5-technique EM suite on VQE using the **ibm_fez noise model**, with full logging and visualization.

### Files
| File | Purpose |
|---|---|
| `download_noise_model.py` | One-time noise model fetch from ibm_fez |
| `vqe_no_mitigation.py` | Baseline VQE, no EM |
| `vqe_dd.py` | VQE + Dynamical Decoupling |
| `vqe_trex.py` | VQE + TREX |
| `vqe_twirling.py` | VQE + Pauli Twirling |
| `vqe_zne.py` | VQE + Zero-Noise Extrapolation |
| `visualize_results.py` | Aggregated comparison plots |
| `EM_TECHNIQUES_DOCUMENTATION.md` | Technical documentation |

### Execution Order Recommended
1. `download_noise_model.py` → cache noise model
2. `vqe_no_mitigation.py` → fastest baseline
3. `vqe_trex.py` → zero overhead EM
4. `vqe_dd.py` → light overhead
5. `vqe_twirling.py` → 10× circuit evaluations
6. `vqe_zne.py` → 3× deeper circuits per iteration
7. `visualize_results.py` → compare all techniques

---

## Stage 10 — Full VQE + EM Suite on Local ibm_kingston Noise Model

**Directory:** [`em_vqe_nm_kingston_h2/`](file:///d:/minor_report/em_vqe_nm_kingston_h2)

**Goal:** Repeat Stage 9 using the **ibm_kingston (133-qubit) noise model** to see how results differ across backend architectures.

### Files (mirror of fez directory)
| File | Purpose |
|---|---|
| `download_noise_model.py` | One-time noise model fetch from ibm_kingston |
| `vqe_no_mitigation.py` | Baseline VQE, no EM |
| `vqe_dd.py` | VQE + Dynamical Decoupling |
| `vqe_trex.py` | VQE + TREX |
| `vqe_twirling.py` | VQE + Pauli Twirling |
| `vqe_zne.py` | VQE + Zero-Noise Extrapolation |
| `visualize_results.py` / `visualize_results_1.py` | Comparison plots |
| `EM_TECHNIQUES_DOCUMENTATION.md` | Technical documentation |

### Key Insight
- ibm_kingston (133 qubits) has a different coupling map and calibration profile than ibm_fez (156 qubits).
- Mapomatic selected a different optimal qubit pair for kingston.
- EM effectiveness varied between backends due to different dominant error sources.

---

## Stage 11 — VQE on Actual QPU (Attempted): ibm_kingston + TREX

**File:** [`em_VQE_H2_trex_qpu_continue_75_100 (1).ipynb`](file:///d:/minor_report/em_VQE_H2_trex_qpu_continue_75_100%20%281%29.ipynb)

**Goal:** Execute VQE on the **actual ibm_kingston QPU** (not local simulation) with TREX error mitigation, in a multi-session "hot-load" strategy.

### What Was Done
- Connected to `ibm_kingston` real QPU via `QiskitRuntimeService`
- Used `mapomatic` to select best physical qubit layout
- Ran **No Mitigation vs TREX** VQE iterations on real hardware
- Used **COBYLA** optimizer with `max_iter=250` (converges around 75–100 iterations)

### Challenge: IBM Quantum Free Account Time Limit
> IBM Quantum free accounts have only **10 minutes of QPU access** per session.  
> With 250 iterations × COBYLA, a single complete run exceeds 10 minutes.

### Workaround: Multi-Account Hot-Loading
- **Saved checkpoint** of optimizer state (current parameters) after each ~75-iteration session
- **Loaded checkpoint** in a new IBM account with fresh 10-min quota
- Repeated across **4 different IBM accounts** to accumulate 100+ iterations
- Observed: at each fresh start, COBYLA **fluctuated** before re-stabilizing — making convergence difficult

### Conclusion
- Full convergence on real QPU was not achievable within free-tier constraints.
- **Partial proof-of-concept** achieved: TREX-mitigated energy values were consistently **closer to −1.85728 Hartree** than no-mitigation runs across the sessions completed.
- This demonstrates that even in constrained real-QPU conditions, EM provides measurable benefit.

---

## Stage 12 — VQE Local Baseline (Clean AerSimulator)

**File:** [`VQE_local.ipynb`](file:///d:/minor_report/VQE_local.ipynb)

**Goal:** Provide a clean VQE reference implementation on ideal AerSimulator for comparison against noisy runs.

### Configuration
- Hamiltonian: H₂ (5-term SparsePauliOp)
- Ansatz: `n_local(2, rotation_blocks="ry", entanglement_blocks="cz")`
- Optimizer: **SPSA**, 125 iterations
- Backend: Ideal (statevector) AerSimulator — no noise

### Key Results
- Ideal AerSimulator converged smoothly to −1.85728 Hartree within ~100 SPSA iterations.
- Confirms ansatz expressibility and optimizer configuration are correct.
- Provides upper-bound on what EM techniques can recover from noisy backends.

---

## Summary: Full Project Progression

```
Stage 1  →  Local AerSimulator + FakeLimaV2 (baseline EM with built-in Qiskit options)
Stage 2  →  VQE on FakeLimaV2 (optimizer comparison: SPSA vs SLSQP vs COBYLA)
Stage 3  →  Real QPU execution (ibm_fez, torino, marrakesh, kingston) with 5 EM modes
Stage 4  →  Noise model extraction from real QPU + Mapomatic for local QPU-like simulation
Stage 5  →  Custom local EM implementations from scratch (DD, TREX, ZNE, Twirling)
Stage 6  →  VQE + EM on ibm_fez noise model (local, SPSA)
Stage 7  →  Multi-backend comparison SPSA (fez + kingston noise models)
Stage 8  →  Multi-backend comparison COBYLA (fez + kingston noise models)
Stage 9  →  Full 5-technique EM VQE suite on ibm_fez noise model (py scripts)
Stage 10 →  Full 5-technique EM VQE suite on ibm_kingston noise model (py scripts)
Stage 11 →  Attempted real QPU VQE (ibm_kingston) + TREX, hot-load multi-account strategy
Stage 12 →  Ideal AerSimulator VQE baseline (clean reference)
```

---

## EM Techniques: Comparison Table

| Technique | Error Type Targeted | Circuit Overhead | Runs/Iteration | Complexity | Best Scenario |
|---|---|---|---|---|---|
| **No Mitigation** | — | None | 1 | Baseline | Reference only |
| **DD** | T1/T2 decoherence | +2 gates per idle slot | 1 | Low | Long idle times in circuit |
| **TREX** | Readout (measurement) | None — modifies observable | 1 | Very Low | High readout error rate (>1%) |
| **ZNE** | Gate noise (any type) | 3× to 5× circuit depth | 3 | High | Linear noise vs. depth scaling |
| **Pauli Twirling** | Coherent gate errors | +Pauli gates around 2q gates | 10 | Medium | Systematic 2-qubit gate errors |

---

## Key Technical Contributions

1. **Custom local EM framework** — Full reimplementation of IBM cloud-only EM techniques for local simulation, with mathematical transparency and per-iteration logging.

2. **Mapomatic integration** — Solved the 156/133-qubit local simulation intractability by selecting optimal 2-qubit subgraphs, enabling faithful noise model simulation.

3. **Multi-backend VQE comparison** — Systematic comparison of VQE + EM across two real QPU noise models (ibm_fez, ibm_kingston) with both SPSA and COBYLA optimizers.

4. **Real QPU proof-of-concept** — Demonstrated measurable EM benefit on actual QPU hardware despite free-tier time constraints, using a checkpoint hot-loading strategy.

5. **Proven EM benefit in VQE** — The core claim of the project: Error Mitigation measurably improves VQE convergence and final energy accuracy, validated on real-QPU-extracted noise models.

---

> **Reference Value:** H₂ exact ground state energy = **−1.857275 Hartree** (NumPy eigensolver)
