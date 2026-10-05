# Error Mitigation Techniques — Local Implementation Documentation

> Complete documentation of all locally-derived EM functions used in the VQE H₂ simulation suite.
> These functions imitate cloud-based IBM Quantum EM primitives for offline/local execution.

---

## Why Local Implementations?

IBM Qiskit Runtime provides built-in error mitigation via `EstimatorOptions` (e.g., `resilience_level`, `twirling`, `zne`), but these are **only available for cloud QPU execution** through IBM Quantum. For local simulation with `AerSimulator`, we must implement equivalent techniques ourselves.

| Cloud API (Unavailable Locally) | Local Function (Our Implementation) |
|---|---|
| `options.resilience.zne_mitigation = True` | `fold_local(circuit, scale)` |
| `options.dynamical_decoupling.enable = True` | `apply_local_dd(circuit, backend)` |
| `options.resilience.measure_mitigation = True` | `apply_local_trex_to_observable(...)` |
| `options.twirling.enable_gates = True` | `apply_pauli_twirling(circuit, ...)` |

**Key benefit**: Our local implementations give us full transparency into what each technique does at the circuit level — something the cloud API abstracts away.

---

## 1. Dynamical Decoupling (DD)

### Function: `apply_local_dd(circuit, backend, active_qubits=None)`

**File**: `vqe_dd.py`

### What It Does
Inserts X-X (XpXm) gate pairs during idle periods on qubits to suppress decoherence errors.

### How It Works
1. **Schedule analysis** (`ALAPScheduleAnalysis`): Analyzes the circuit timing to find idle gaps between gates, using As-Late-As-Possible scheduling
2. **DD padding** (`PadDynamicalDecoupling`): Fills each idle gap with an X-X gate sequence
3. The X-X pair acts as a "spin echo" — the first X flips the qubit state, and after the idle time, the second X flips it back

### Why It Helps
- T1 (energy relaxation) and T2 (dephasing) errors accumulate during idle time
- The X-X sequence refocuses the qubit by periodically flipping it, causing the accumulated phase errors from the environment to cancel out
- Effectively "freezes" the qubit state during idle periods

### Uniqueness of Our Implementation
- Uses Qiskit's built-in `PadDynamicalDecoupling` transpiler pass, which is the same algorithm used on cloud, but we control when/where it's applied
- We can target specific physical qubits via `active_qubits` parameter
- We can inspect the resulting circuit depth increase (logged per iteration)

### Effectiveness
- **Best for**: Circuits with significant idle time (e.g., when one qubit waits while the other is being operated on)
- **Limited when**: All qubits are densely packed with gates (no idle time to fill)
- **Overhead**: Adds 2 gates per idle slot, minimal circuit depth increase

### Parameters
| Parameter | Value | Purpose |
|---|---|---|
| DD Sequence | X-X (XpXm) | Two X gates refocus decoherence |
| Active Qubits | `best_layout` | Only protect the qubits we use |

---

## 2. TREX (Twirled Readout Error eXtinction)

### Function: `apply_local_trex_to_observable(observable_2q, readout_errors_dict, layout)`

**File**: `vqe_trex.py`

### What It Does
Scales observable coefficients to compensate for measurement (readout) errors, so the noisy measurement result approximates the true expectation value.

### How It Works
1. **Get readout error rates** `p0`, `p1` for each physical qubit from the saved backend properties
2. **For each Pauli term** in the Hamiltonian:
   - If the qubit position has a non-Identity operator (X, Y, or Z), multiply the attenuation factor by `(1 - 2*p)` for that qubit
   - Identity operators are NOT affected (readout errors don't change ⟨I⟩ = 1)
3. **Scale UP** each coefficient by dividing by the attenuation factor: `coeff_scaled = coeff / attenuation`

### Mathematical Basis
For a single qubit with readout error `p`:
- True expectation: `⟨O⟩_ideal`
- Measured expectation: `⟨O⟩_noisy = (1 - 2p) × ⟨O⟩_ideal`
- TREX correction: `⟨O⟩_mitigated = ⟨O⟩_noisy / (1 - 2p) ≈ ⟨O⟩_ideal`

For 2 qubits: attenuation factors multiply independently.

### Uniqueness of Our Implementation
- We read readout errors from the **saved** `backend_properties.json` (no live connection needed)
- The cloud TREX uses randomized calibration circuits; our version uses the backend's reported error rates directly
- We log the exact scaling factor for each Pauli term, making the correction fully transparent

### Effectiveness
- **Best for**: Backends with significant readout errors (> 1%)
- **Limited when**: Readout errors are negligible or when gate errors dominate
- **Overhead**: Zero additional circuit runs — only modifies the observable coefficients
- **Caveat**: Assumes readout errors are independent per qubit and time-invariant

### Parameters
| Parameter | Value | Purpose |
|---|---|---|
| Readout Error q0 | From backend props | Bit-flip probability on measurement |
| Readout Error q1 | From backend props | Bit-flip probability on measurement |

---

## 3. Zero-Noise Extrapolation (ZNE)

### Function: `fold_local(circuit, scale)`

**File**: `vqe_zne.py`

### What It Does
Amplifies noise by "folding" the circuit (appending inverse+forward copies), runs at multiple noise levels, then extrapolates to what the result would be at zero noise.

### How It Works
1. **Folding**: For scale `n` (must be odd):
   - Scale 1: `C` (original)
   - Scale 3: `C · C† · C` (3× depth)
   - Scale 5: `C · C† · C · C† · C` (5× depth)
2. **Barriers** inserted between copies to prevent transpiler optimization
3. **Run** each folded circuit independently with the estimator
4. **Extrapolate**: Fit a linear model (polyfit degree 1) to `(scale, energy)` pairs
5. **y-intercept** = zero-noise estimate

### Mathematical Basis
If noise scales linearly with circuit depth:
- `E(λ) = E_ideal + α × λ` where λ is the noise scale
- By measuring at λ = {1, 3, 5}, we fit the line and extrapolate to λ = 0
- `E(0) = E_ideal` (intercept of the fit)

### Uniqueness of Our Implementation
- We fold at the **logical circuit** level before transpilation, preserving the circuit semantics
- The cloud ZNE uses Qiskit Runtime's built-in noise amplification; ours uses explicit circuit duplication
- We store per-scale energies AND the linear fit slope/intercept for every iteration
- We can see how the noise-energy relationship evolves during optimization

### Effectiveness
- **Best for**: Circuits where noise scales approximately linearly with depth
- **Limited when**: Noise is non-linear, or circuit is already too deep (folded versions become meaningless)
- **Overhead**: 3× circuit evaluations per VQE iteration (one per scale) — significantly slower
- **Trade-off**: Most expensive technique but can achieve best accuracy if noise is well-behaved

### Parameters
| Parameter | Value | Purpose |
|---|---|---|
| Noise Scales | [1, 3, 5] | Amplification factors (odd integers) |
| Extrapolation | Linear (polyfit degree 1) | Assumes linear noise-energy relationship |

---

## 4. Pauli Twirling

### Function: `apply_pauli_twirling(circuit, num_twirls=10, seed=42)`

**File**: `vqe_twirling.py`

### What It Does
Creates multiple randomized copies of the circuit by inserting random Pauli gates around 2-qubit gates, then averages the results to convert coherent errors into stochastic noise.

### How It Works
1. **For each 2-qubit gate** (CX, ECR, etc.) in the circuit:
   a. Generate random 2-qubit Pauli `P_pre` from {I,X,Y,Z}⊗{I,X,Y,Z} (16 possibilities)
   b. Compute `P_post = Gate · P_pre · Gate†` using the Clifford algebra
   c. Insert `P_pre` before the gate and `P_post` after the gate
2. This preserves the **ideal** unitary: `P_post · Gate · P_pre = Gate` (algebraic identity)
3. Under **noise**, each twirled copy sees a randomized version of the coherent error
4. **Average** over all copies: coherent errors average to zero, leaving only stochastic noise

### Mathematical Basis
Coherent error: `U_noisy = (I + εH) · U_ideal` where H is a fixed error Hamiltonian
After twirling: `E[U_twirled_noisy] = (1 - ε²) · U_ideal + ε² · depolarizing`
- Coherent (systematic) → Stochastic (random, averages out)

### Uniqueness of Our Implementation
- We use Qiskit's `Clifford` class to compute the conjugated Pauli (exact group theory)
- Single-qubit gates are left untouched (only 2-qubit gates are twirled)
- Failed Clifford computations (non-Clifford gates) fall back gracefully
- We re-transpile twirled circuits at optimization_level=1 to basis gates
- We store **all 10 individual twirl energies** plus their mean and std per iteration

### Effectiveness
- **Best for**: Circuits with dominant coherent (systematic) 2-qubit gate errors
- **Limited when**: Noise is already predominantly stochastic (twirling won't help further)
- **Overhead**: 10× circuit evaluations per iteration (one per twirl)
- **Insight**: The standard deviation of twirl energies measures how much coherent error was present

### Parameters
| Parameter | Value | Purpose |
|---|---|---|
| Num Twirls | 10 | Number of randomized copies |
| Seed | 42 | Reproducibility |
| Retranspile Level | 1 | Light optimization to convert Paulis to basis gates |

---

## Comparison Table

| Technique | Error Type Targeted | Circuit Overhead | Runs per Iteration | Complexity | Best Scenario |
|---|---|---|---|---|---|
| **DD** | T1/T2 decoherence | +2 gates/idle slot | 1 | Low | Long idle times |
| **TREX** | Readout (measurement) | None (modifies observable) | 1 | Very Low | High readout error |
| **ZNE** | Gate noise (any) | 3×-5× circuit depth | 3 | High | Linear noise scaling |
| **Twirling** | Coherent gate errors | +Pauli gates around 2q gates | 10 | Medium | Systematic 2q errors |

## Execution Order Recommendation

1. `download_noise_model.py` — one-time setup
2. `vqe_no_mitigation.py` — baseline (fastest)
3. `vqe_trex.py` — fastest EM technique (no extra circuits)
4. `vqe_dd.py` — moderate speed (1 circuit per iteration, slightly deeper)
5. `vqe_twirling.py` — slower (10 circuits per iteration)
6. `vqe_zne.py` — slowest (3 circuits per iteration, each deeper)
7. `visualize_results.py` — compare everything
