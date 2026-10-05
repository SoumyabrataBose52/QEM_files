# ============================================================================
# Visualize & Compare VQE Error Mitigation Results
# ============================================================================
# Reads all result text files from the results/ folder and generates:
#   1. Convergence overlay plot (all techniques on one chart)
#   2. Final energy bar chart with exact energy reference
#   3. Error comparison (absolute and percentage)
#   4. Summary comparison table
#   5. Parameter evolution comparison
#
# Saves combined plot to results/comparison_<timestamp>.png
# ============================================================================

import os
import re
import glob
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict

# ─── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# ─── Result File Parser ────────────────────────────────────────────────────
def parse_result_file(filepath):
    """Parse a VQE result text file and extract all data."""
    data = {
        "filepath": filepath,
        "filename": os.path.basename(filepath),
        "technique": "",
        "backend": "",
        "timestamp": "",
        "shots": 0,
        "max_iterations": 0,
        "seed": 0,
        "exact_energy": 0.0,
        "exact_00_val": 0.0,
        "eigenvalues": [],
        "hamiltonian_terms": [],
        "ansatz_type": "",
        "ansatz_qubits": 0,
        "ansatz_reps": 0,
        "ansatz_num_params": 0,
        "ansatz_depth": 0,
        "isa_depth": 0,
        "isa_gates": {},
        "best_layout": [],
        "em_config": {},
        "initial_params": [],
        "iteration_energies": [],
        "iteration_params": [],
        "converged_energy": 0.0,
        "abs_error": 0.0,
        "pct_error": 0.0,
        "total_iterations": 0,
        "optimizer_success": False,
        "optimizer_message": "",
        "final_params": [],
    }

    with open(filepath, "r") as f:
        content = f.read()
        lines = content.split("\n")

    # Extract technique from header
    for line in lines[:5]:
        if "VQE RESULTS:" in line:
            data["technique"] = line.split("VQE RESULTS:")[1].strip()
            break

    # Parse key-value pairs
    for line in lines:
        line = line.strip()
        if line.startswith("Backend:"):
            data["backend"] = line.split(":", 1)[1].strip()
        elif line.startswith("Timestamp:"):
            data["timestamp"] = line.split(":", 1)[1].strip()
        elif line.startswith("Shots:"):
            data["shots"] = int(line.split(":")[1].strip())
        elif line.startswith("Max Iterations:"):
            data["max_iterations"] = int(line.split(":")[1].strip())
        elif line.startswith("Initial Seed:"):
            data["seed"] = int(line.split(":")[1].strip())
        elif line.startswith("Exact Ground State Energy:"):
            data["exact_energy"] = float(line.split(":")[1].strip().replace("Hartree", "").strip())
        elif line.startswith("Exact |00> Expectation Value:"):
            data["exact_00_val"] = float(line.split(":")[1].strip())
        elif line.startswith("Eigenvalues:"):
            try:
                ev_str = line.split(":", 1)[1].strip()
                data["eigenvalues"] = eval(ev_str)
            except:
                pass
        elif line.startswith("Type:") and "EfficientSU2" in line:
            data["ansatz_type"] = line.split(":")[1].strip()
        elif line.startswith("Qubits:") and data["ansatz_qubits"] == 0:
            try: data["ansatz_qubits"] = int(line.split(":")[1].strip())
            except: pass
        elif line.startswith("Reps:"):
            try: data["ansatz_reps"] = int(line.split(":")[1].strip())
            except: pass
        elif line.startswith("Num Parameters:"):
            try: data["ansatz_num_params"] = int(line.split(":")[1].strip())
            except: pass
        elif line.startswith("Ansatz Depth:"):
            try: data["ansatz_depth"] = int(line.split(":")[1].strip())
            except: pass
        elif line.startswith("ISA Depth:"):
            try: data["isa_depth"] = int(line.split(":")[1].strip())
            except: pass
        elif line.startswith("ISA Gates:"):
            try: data["isa_gates"] = eval(line.split(":", 1)[1].strip())
            except: pass
        elif line.startswith("Best Layout:"):
            try: data["best_layout"] = eval(line.split(":", 1)[1].strip())
            except: pass
        elif line.startswith("Converged Energy:"):
            try: data["converged_energy"] = float(line.split(":")[1].strip().replace("Hartree", "").strip())
            except: pass
        elif line.startswith("Absolute Error:"):
            try: data["abs_error"] = float(line.split(":")[1].strip())
            except: pass
        elif line.startswith("Percentage Error:"):
            try: data["pct_error"] = float(line.split(":")[1].strip().replace("%", ""))
            except: pass
        elif line.startswith("Total Iterations:"):
            try: data["total_iterations"] = int(line.split(":")[1].strip())
            except: pass
        elif line.startswith("Optimizer Success:"):
            data["optimizer_success"] = "True" in line
        elif line.startswith("Optimizer Message:"):
            data["optimizer_message"] = line.split(":", 1)[1].strip()
        elif line.startswith("x0 ="):
            try: data["initial_params"] = eval(line.split("=", 1)[1].strip())
            except: pass
        elif line.startswith("Final Parameters:"):
            try: data["final_params"] = eval(line.split(":", 1)[1].strip())
            except: pass

    # Parse Hamiltonian terms
    for line in lines:
        if line.strip().startswith("Term:") and "Coeff:" in line:
            parts = line.strip().split()
            term_idx = parts.index("Term:") + 1
            coeff_idx = parts.index("Coeff:") + 1
            data["hamiltonian_terms"].append({
                "term": parts[term_idx],
                "coeff": float(parts[coeff_idx])
            })

    # Parse iteration history
    for line in lines:
        match = re.match(r'\s*Iter\s+(\d+)\s*\|\s*Energy:\s*([\+\-\d\.eE]+)', line)
        if match:
            energy = float(match.group(2))
            data["iteration_energies"].append(energy)

            # Extract params if present
            param_match = re.search(r'Params:\s*\[(.+)\]', line)
            if param_match:
                try:
                    params = [float(x.strip()) for x in param_match.group(1).split(",")]
                    data["iteration_params"].append(params)
                except:
                    pass

    return data


# ─── Find Latest Result Files ──────────────────────────────────────────────
def find_latest_results():
    """Find the most recent result file for each technique."""
    techniques = {
        "no_mitigation": "No Mitigation",
        "dd": "Dynamical Decoupling",
        "trex": "TREX",
        "zne": "ZNE",
        "twirling": "Pauli Twirling",
    }

    results = OrderedDict()
    for key, label in techniques.items():
        pattern = os.path.join(RESULTS_DIR, f"results_{key}_*.txt")
        files = sorted(glob.glob(pattern))
        if files:
            latest = files[-1]  # Alphabetically last = chronologically latest
            results[label] = parse_result_file(latest)
            print(f"✓ Found {label}: {os.path.basename(latest)}")
        else:
            print(f"✗ No results for {label}")

    return results


# ─── Main Visualization ────────────────────────────────────────────────────
print("=" * 70)
print("VQE ERROR MITIGATION COMPARISON VISUALIZER")
print("=" * 70)

if not os.path.exists(RESULTS_DIR):
    print(f"ERROR: Results directory not found: {RESULTS_DIR}")
    print("Run the VQE scripts first!")
    exit(1)

results = find_latest_results()

if len(results) == 0:
    print("\nNo result files found! Please run the VQE scripts first.")
    exit(1)

print(f"\nFound {len(results)} technique(s) to compare")

# Extract exact energy (should be same across all files)
exact_energy = None
for label, data in results.items():
    if data["exact_energy"] != 0:
        exact_energy = data["exact_energy"]
        break
if exact_energy is None:
    exact_energy = -1.8572750302023786  # Fallback

# ─── Color scheme ───────────────────────────────────────────────────────────
COLORS = {
    "No Mitigation": "#e74c3c",
    "Dynamical Decoupling": "#3498db",
    "TREX": "#2ecc71",
    "ZNE": "#f39c12",
    "Pauli Twirling": "#9b59b6",
}

# ─── Plot 1: Convergence Overlay ───────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

ax1 = axes[0, 0]
for label, data in results.items():
    energies = data["iteration_energies"]
    if energies:
        n = len(energies)
        final_e = energies[-1]
        color = COLORS.get(label, "gray")
        ax1.plot(range(1, n + 1), energies,
                label=f"{label} (E={final_e:.4f}, n={n})",
                color=color, linewidth=2, alpha=0.85)

ax1.axhline(y=exact_energy, color='black', linestyle='--', linewidth=1.5, alpha=0.7,
           label=f'Exact GS ({exact_energy:.4f})')
ax1.set_xlabel("Iteration", fontsize=12)
ax1.set_ylabel("Energy (Hartree)", fontsize=12)
ax1.set_title("VQE Convergence — All EM Techniques", fontsize=14)
ax1.legend(fontsize=9, loc='best')
ax1.grid(True, alpha=0.3)

# ─── Plot 2: Final Energy Bar Chart ────────────────────────────────────────
ax2 = axes[0, 1]
labels_plot = []
energies_plot = []
colors_plot = []

for label, data in results.items():
    if data["converged_energy"] != 0:
        labels_plot.append(label)
        energies_plot.append(data["converged_energy"])
        colors_plot.append(COLORS.get(label, "gray"))

if labels_plot:
    bars = ax2.bar(range(len(labels_plot)), energies_plot,
                   color=colors_plot, alpha=0.85, edgecolor='white', linewidth=1.5)
    ax2.axhline(y=exact_energy, color='black', linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'Exact GS ({exact_energy:.4f})')
    ax2.set_xticks(range(len(labels_plot)))
    ax2.set_xticklabels(labels_plot, rotation=25, ha='right', fontsize=9)
    ax2.set_ylabel("Energy (Hartree)", fontsize=12)
    ax2.set_title("Final Converged Energies", fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, axis='y', alpha=0.3)
    for bar, e in zip(bars, energies_plot):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{e:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# ─── Plot 3: Percentage Error Comparison ───────────────────────────────────
ax3 = axes[1, 0]
error_labels = []
pct_errors = []
error_colors = []

for label, data in results.items():
    if data["abs_error"] != 0 or data["converged_energy"] != 0:
        error_labels.append(label)
        ae = data["abs_error"] if data["abs_error"] != 0 else abs(data["converged_energy"] - exact_energy)
        pe = data["pct_error"] if data["pct_error"] != 0 else abs(ae / exact_energy) * 100
        pct_errors.append(pe)
        error_colors.append(COLORS.get(label, "gray"))

if error_labels:
    x = np.arange(len(error_labels))
    bars = ax3.bar(x, pct_errors, 0.5, color=error_colors, alpha=0.4,
                   edgecolor='black', linewidth=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(error_labels, rotation=25, ha='right', fontsize=9)
    ax3.set_ylabel("Percentage Error (%)", fontsize=12)
    ax3.set_title("Percentage Error by Technique", fontsize=14)
    ax3.grid(True, axis='y', alpha=0.3)

    for bar, pe in zip(bars, pct_errors):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{pe:.2f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

# ─── Plot 4: Parameter Evolution Summary ───────────────────────────────────
# Show mean parameter magnitude over iterations as a summary on the main plot
ax4 = axes[1, 1]
has_param_data = False

for label, data in results.items():
    iter_params = data["iteration_params"]
    if iter_params and len(iter_params) > 1:
        has_param_data = True
        # Plot mean absolute parameter value per iteration
        mean_abs = [np.mean(np.abs(p)) for p in iter_params]
        color = COLORS.get(label, "gray")
        ax4.plot(range(1, len(mean_abs) + 1), mean_abs,
                label=f"{label}",
                color=color, linewidth=2, alpha=0.85)

if has_param_data:
    ax4.set_xlabel("Iteration", fontsize=12)
    ax4.set_ylabel("Mean |θ| (rad)", fontsize=12)
    ax4.set_title("Parameter Evolution Summary — All EM Techniques", fontsize=14)
    ax4.legend(fontsize=8, loc='best')
    ax4.grid(True, alpha=0.3)

plt.suptitle(f"VQE Error Mitigation Comparison — H₂ Hamiltonian\n"
             f"Noise Model: {list(results.values())[0].get('backend', 'Unknown')}",
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()

# Save main plot
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
plot_path = os.path.join(RESULTS_DIR, f"comparison_{timestamp}.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Comparison plot saved: {plot_path}")
plt.show()

# ─── Figure 2: Individual Parameter Evolution (θ0–θ7) ─────────────────────
# 8 subplots (2×4 grid), one per parameter, all techniques overlaid
n_params = 8  # EfficientSU2 with 2 qubits, 1 rep
any_param_data = any(len(d["iteration_params"]) > 1 for d in results.values())

if any_param_data:
    fig2, axes2 = plt.subplots(2, 4, figsize=(22, 10))
    axes2_flat = axes2.flatten()

    for p_idx in range(n_params):
        ax = axes2_flat[p_idx]
        for label, data in results.items():
            iter_params = data["iteration_params"]
            if iter_params and len(iter_params) > 1:
                param_vals = [p[p_idx] if p_idx < len(p) else None for p in iter_params]
                param_vals = [v for v in param_vals if v is not None]
                color = COLORS.get(label, "gray")
                ax.plot(range(1, len(param_vals) + 1), param_vals,
                       label=label, color=color, linewidth=1.5, alpha=0.85)

        ax.set_title(f"θ{p_idx}", fontsize=13, fontweight='bold')
        ax.set_xlabel("Iteration", fontsize=9)
        ax.set_ylabel("Value (rad)", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        if p_idx == 0:
            ax.legend(fontsize=7, loc='best')

    fig2.suptitle(f"Individual Parameter Evolution (θ0–θ7) — All EM Techniques\n"
                  f"Noise Model: {list(results.values())[0].get('backend', 'Unknown')}",
                  fontsize=16, fontweight='bold', y=1.01)
    fig2.tight_layout()

    params_plot_path = os.path.join(RESULTS_DIR, f"parameter_evolution_{timestamp}.png")
    fig2.savefig(params_plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Parameter evolution plot saved: {params_plot_path}")
    plt.show()

# ─── Summary Table ──────────────────────────────────────────────────────────
print(f"\n{'=' * 100}")
print("VQE ERROR MITIGATION COMPARISON SUMMARY")
print(f"{'=' * 100}")

rows = []
for label, data in results.items():
    conv_e = data["converged_energy"]
    ae = data["abs_error"] if data["abs_error"] != 0 else abs(conv_e - exact_energy)
    pe = data["pct_error"] if data["pct_error"] != 0 else abs(ae / exact_energy) * 100
    rows.append({
        "EM Technique": label,
        "Converged Energy": f"{conv_e:.6f}",
        "Exact Energy": f"{exact_energy:.6f}",
        "Abs Error": f"{ae:.6f}",
        "% Error": f"{pe:.2f}%",
        "Iterations": data["total_iterations"],
        "Success": data["optimizer_success"],
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False))

# ─── Detailed per-technique summary ────────────────────────────────────────
print(f"\n{'=' * 100}")
print("DETAILED PER-TECHNIQUE INFORMATION")
print(f"{'=' * 100}")
for label, data in results.items():
    print(f"\n--- {label} ---")
    print(f"  File: {data['filename']}")
    print(f"  Backend: {data['backend']}")
    print(f"  Shots: {data['shots']}")
    print(f"  Max Iters: {data['max_iterations']}")
    print(f"  Actual Iters: {data['total_iterations']}")
    print(f"  Converged Energy: {data['converged_energy']:.10f}")
    print(f"  Optimizer Success: {data['optimizer_success']}")
    print(f"  Optimizer Message: {data['optimizer_message']}")
    if data['final_params']:
        print(f"  Final Params: {[f'{p:.4f}' for p in data['final_params']]}")

# ─── Rankings ───────────────────────────────────────────────────────────────
print(f"\n{'=' * 100}")
print("RANKINGS (best → worst, by % error)")
print(f"{'=' * 100}")

ranked = sorted(rows, key=lambda x: float(x["% Error"].replace("%", "")))
for i, r in enumerate(ranked):
    medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][min(i, 4)]
    print(f"  {medal} {r['EM Technique']}: {r['% Error']} error ({r['Converged Energy']} Hartree)")

print(f"\nExact ground state energy: {exact_energy:.10f} Hartree")
print(f"{'=' * 100}")

# ─── Export Comparison Table to Excel & CSV ─────────────────────────────────
# Build a richer DataFrame for export
export_rows = []
for label, data in results.items():
    conv_e = data["converged_energy"]
    ae = data["abs_error"] if data["abs_error"] != 0 else abs(conv_e - exact_energy)
    pe = data["pct_error"] if data["pct_error"] != 0 else abs(ae / exact_energy) * 100
    energies = data["iteration_energies"]
    best_e = min(energies) if energies else conv_e
    export_rows.append({
        "EM Technique": label,
        "Backend": data["backend"],
        "Best Layout (Mapomatic)": str(data.get("best_layout", "")),
        "Shots": data["shots"],
        "Max Iterations": data["max_iterations"],
        "Actual Iterations": data["total_iterations"],
        "Converged Energy (Hartree)": conv_e,
        "Best Energy (Hartree)": best_e,
        "Exact Energy (Hartree)": exact_energy,
        "Absolute Error": ae,
        "Percentage Error (%)": pe,
        "Optimizer Success": data["optimizer_success"],
        "Optimizer Message": data["optimizer_message"],
        "Ansatz Type": data["ansatz_type"],
        "Ansatz Qubits": data["ansatz_qubits"],
        "Ansatz Reps": data["ansatz_reps"],
        "Num Parameters": data["ansatz_num_params"],
        "Ansatz Depth": data["ansatz_depth"],
        "ISA Depth": data["isa_depth"],
        "ISA Gates": str(data["isa_gates"]),
        "Initial Seed": data["seed"],
        "Final Parameters": str(data["final_params"]),
    })

df_export = pd.DataFrame(export_rows)

# Sort by percentage error (best technique first)
df_export = df_export.sort_values("Percentage Error (%)")

# Save as CSV
csv_path = os.path.join(RESULTS_DIR, f"comparison_table_{timestamp}.csv")
df_export.to_csv(csv_path, index=False)
print(f"\n✓ Comparison table (CSV) saved: {csv_path}")

# Save as Excel (with formatting if openpyxl available)
try:
    xlsx_path = os.path.join(RESULTS_DIR, f"comparison_table_{timestamp}.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        # Sheet 1: Summary comparison
        df_export.to_excel(writer, sheet_name="Summary", index=False)

        # Sheet 2: Iteration-by-iteration energy for all techniques
        max_iters = max(len(d["iteration_energies"]) for d in results.values())
        iter_data = {"Iteration": list(range(1, max_iters + 1))}
        for label, data in results.items():
            energies = data["iteration_energies"]
            padded = energies + [None] * (max_iters - len(energies))
            iter_data[label] = padded
        iter_data["Exact Energy"] = [exact_energy] * max_iters
        df_iters = pd.DataFrame(iter_data)
        df_iters.to_excel(writer, sheet_name="Convergence Data", index=False)

        # Sheet 3: Final parameters side by side
        param_rows = []
        for label, data in results.items():
            fp = data["final_params"]
            if fp:
                row = {"EM Technique": label}
                for j, p in enumerate(fp):
                    row[f"θ{j}"] = p
                param_rows.append(row)
        if param_rows:
            df_params = pd.DataFrame(param_rows)
            df_params.to_excel(writer, sheet_name="Final Parameters", index=False)

        # Sheet 4: Rankings
        ranking_rows = []
        for i, r in enumerate(ranked):
            ranking_rows.append({
                "Rank": i + 1,
                "EM Technique": r["EM Technique"],
                "Converged Energy (Hartree)": r["Converged Energy"],
                "Absolute Error": r["Abs Error"],
                "Percentage Error": r["% Error"],
                "Iterations": r["Iterations"],
                "Converged": r["Success"],
            })
        df_rank = pd.DataFrame(ranking_rows)
        df_rank.to_excel(writer, sheet_name="Rankings", index=False)

    print(f"✓ Comparison table (Excel) saved: {xlsx_path}")
    print(f"  Sheets: Summary, Convergence Data, Final Parameters, Rankings")
except ImportError:
    print("  (openpyxl not installed — Excel export skipped, CSV is available)")
except Exception as e:
    print(f"  (Excel export failed: {e} — CSV is available)")

print(f"\n{'=' * 70}")
print("VISUALIZATION COMPLETE")
print(f"{'=' * 70}")
