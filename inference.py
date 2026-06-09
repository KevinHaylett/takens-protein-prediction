"""
inference.py - Protein Structure Prediction
============================================
Takens-Based Transformer (MARINA Architecture)
Kevin R. Haylett, PhD - Manchester, UK

Usage:
    python inference.py

Set MODEL_PATH and CSV_FILE in config.py, then run.
Results saved to config.OUTPUT_DIR.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix OpenMP conflict on Windows/Anaconda

import sys
import json
import warnings
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore', category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from protein.protein_encoder import ProteinEncoder
from protein_tbt import ProteinTBT
from protein.metrics import evaluate_prediction, StructureMetrics


# ============================================================
# CONFIGURATION - set these in config.py or override here
# ============================================================

MODEL_PATH = str(Path(config.CHECKPOINT_DIR) / 'best_model.pt')
CSV_FILE   = "data/csv_files/1A7S.csv"   # Set this to your target protein CSV, e.g. config.CSV_DIR + "/1A7S.csv"
OUTPUT_DIR = config.OUTPUT_DIR


# ============================================================
# FUNCTIONS
# ============================================================

def load_protein_from_csv(csv_file):
    """Load protein Calpha coordinates from a CSV file."""
    print(f"Loading protein from: {csv_file}")
    df = pd.read_csv(csv_file)
    ca_df = df[df['atom_name'] == 'CA'].copy()
    if len(ca_df) == 0:
        raise ValueError(f"No CA atoms found in {csv_file}")
    ca_df = ca_df.sort_values('residue_seq').reset_index(drop=True)
    sequence = ca_df['residue_name'].tolist()
    coords   = ca_df[['x', 'y', 'z']].values.astype(np.float32)
    coords   = coords - coords.mean(axis=0)   # centre (matches training)
    pdb_id   = Path(csv_file).stem
    print(f"  Loaded {len(sequence)} residues")
    return sequence, coords, pdb_id


def predict_structure(model, sequence, encoder, device='cpu'):
    """Predict Calpha coordinates from amino acid sequence."""
    print("Running prediction...")
    input_ids = encoder.encode_sequence(sequence)
    tensor    = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        coords = model.predict(tensor)
    coords = coords.squeeze(0).cpu().numpy()[:len(sequence)]
    print(f"  Predicted {len(coords)} coordinates")
    return coords


def plot_structures(true_coords, pred_coords, pdb_id, rmsd, output_dir):
    """Save 3-panel structure comparison plot."""
    fig = plt.figure(figsize=(15, 5))
    for ax_idx, (coords, label, colour) in enumerate([
        (true_coords, 'True', 'b'),
        (pred_coords, 'Predicted', 'r')
    ], 1):
        ax = fig.add_subplot(1, 3, ax_idx, projection='3d')
        ax.plot(coords[:,0], coords[:,1], coords[:,2],
                f'{colour}-o', markersize=3, linewidth=1, label=label)
        ax.set_xlabel('X (A)'); ax.set_ylabel('Y (A)'); ax.set_zlabel('Z (A)')
        ax.set_title(f'{pdb_id} - {"True" if ax_idx==1 else "Predicted"} Structure')
        ax.legend(); ax.grid(True, alpha=0.3)

    pred_aligned, true_aligned, _ = StructureMetrics.align_structures(pred_coords, true_coords)
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot(true_aligned[:,0], true_aligned[:,1], true_aligned[:,2],
             'b-o', markersize=3, linewidth=1, label='True', alpha=0.7)
    ax3.plot(pred_aligned[:,0], pred_aligned[:,1], pred_aligned[:,2],
             'r-o', markersize=3, linewidth=1, label='Predicted', alpha=0.7)
    ax3.set_xlabel('X (A)'); ax3.set_ylabel('Y (A)'); ax3.set_zlabel('Z (A)')
    ax3.set_title(f'{pdb_id} - Aligned Overlay\nRMSD: {rmsd:.2f}A')
    ax3.legend(); ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(output_dir) / f'{pdb_id}_comparison.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  Structure plot saved: {out}")
    if config.SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def plot_per_residue_rmsd(per_res_rmsd, pdb_id, output_dir):
    """Save per-residue RMSD plot."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(per_res_rmsd)), per_res_rmsd,
            'o-', linewidth=1, markersize=4, color='steelblue')
    ax.axhline(per_res_rmsd.mean(), color='r', linestyle='--', linewidth=2,
               label=f'Mean: {per_res_rmsd.mean():.2f}A')
    ax.axhline(8.0, color='g', linestyle='--', alpha=0.5, linewidth=2,
               label='Target: 8.0A')
    ax.set_xlabel('Residue Index', fontsize=11)
    ax.set_ylabel('RMSD (A)', fontsize=11)
    ax.set_title(f'{pdb_id} - Per-Residue RMSD', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = Path(output_dir) / f'{pdb_id}_per_residue_rmsd.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  Per-residue RMSD plot saved: {out}")
    if config.SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def save_predicted_pdb(pred_coords, sequence, pdb_id, output_dir):
    """Save predicted structure as a PDB file."""
    out = Path(output_dir) / f'{pdb_id}_predicted.pdb'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        f.write(f"HEADER    PREDICTED STRUCTURE - {pdb_id}\n")
        f.write(f"TITLE     TAKENS-BASED TRANSFORMER PREDICTION (MARINA)\n")
        f.write(f"REMARK    Kevin R. Haylett - https://finitemechanics.com\n")
        for i, (aa, coord) in enumerate(zip(sequence, pred_coords)):
            f.write(f"ATOM  {i+1:5d}  CA  {aa:3s} A{i+1:4d}    "
                    f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                    f"  1.00  0.00           C\n")
        f.write("END\n")
    print(f"  Predicted PDB saved: {out}")


def save_results_json(metrics_dict, pdb_id, sequence_length, output_dir):
    """Save structured results for reproducibility and logging."""
    results = {
        "pdb_id": pdb_id,
        "date": datetime.now().isoformat(),
        "architecture": "Takens-Based Transformer (MARINA)",
        "sequence_length": sequence_length,
        "metrics": {
            "rmsd_A": round(metrics_dict['rmsd'], 4),
            "mean_per_residue_rmsd_A": round(float(metrics_dict['per_residue_rmsd'].mean()), 4),
            "gdt_ts": round(metrics_dict['gdt_ts'], 4),
            "tm_score": round(metrics_dict['tm_score'], 4),
        },
        "notes": "Proof of concept - small training set"
    }

    # ── FIX: Convert NumPy types (float32, etc.) so json.dump works on Windows/Anaconda ──
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(i) for i in obj]
        return obj

    results = convert_numpy(results)
    # ───────────────────────────────────────────────────────────────────────────────────

    out = Path(output_dir) / f'{pdb_id}_results.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results JSON saved: {out}")
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("PROTEIN STRUCTURE PREDICTION - INFERENCE")
    print("Takens-Based Transformer (MARINA)")
    print("=" * 70)

    if not CSV_FILE:
        print("\nERROR: Set CSV_FILE at the top of inference.py before running.")
        return

    if not Path(MODEL_PATH).exists():
        print(f"\nERROR: Model not found at {MODEL_PATH}")
        print("Run train.py first to create a trained model.")
        return

    if not Path(CSV_FILE).exists():
        print(f"\nERROR: CSV file not found at {CSV_FILE}")
        return

    # Load protein
    sequence, true_coords, pdb_id = load_protein_from_csv(CSV_FILE)
    print(f"  Protein: {pdb_id}  ({len(sequence)} residues)")

    # Load model
    print("\nLoading model...")
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    model = ProteinTBT(**checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Parameters: {model.get_num_params():,}")

    # Encoder
    encoder = ProteinEncoder()
    one_letter = encoder.decode_sequence(encoder.encode_sequence(sequence), one_letter=True)
    print(f"  Sequence:  {one_letter[:50]}{'...' if len(one_letter)>50 else ''}")

    # Predict
    pred_coords = predict_structure(model, sequence, encoder)

    # Evaluate
    print("\nEvaluating...")
    metrics_dict = evaluate_prediction(pred_coords, true_coords,
                                       sequence=one_letter, verbose=config.VERBOSE)

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Protein:              {pdb_id}")
    print(f"  Length:               {len(sequence)} residues")
    print(f"  RMSD (aligned):       {metrics_dict['rmsd']:.2f} A")
    print(f"  Mean per-residue:     {metrics_dict['per_residue_rmsd'].mean():.2f} A")
    print(f"  GDT_TS:               {metrics_dict['gdt_ts']:.2f}")
    print(f"  TM-score:             {metrics_dict['tm_score']:.4f}")

    # Interpretation
    rmsd = metrics_dict['rmsd']
    if   rmsd < 3.0:  print("\n  Excellent - near-atomic accuracy")
    elif rmsd < 5.0:  print("\n  Very good - high accuracy")
    elif rmsd < 8.0:  print("\n  Good - fold is recognisable (target achieved)")
    elif rmsd < 12.0: print("\n  Fair - some features recognisable")
    else:             print("\n  Needs improvement - try more epochs or more proteins")

    # Save outputs
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    plot_structures(true_coords, pred_coords, pdb_id, metrics_dict['rmsd'], OUTPUT_DIR)
    plot_per_residue_rmsd(metrics_dict['per_residue_rmsd'], pdb_id, OUTPUT_DIR)
    if config.SAVE_PREDICTIONS:
        save_predicted_pdb(pred_coords, sequence, pdb_id, OUTPUT_DIR)
    save_results_json(metrics_dict, pdb_id, len(sequence), OUTPUT_DIR)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("Open the predicted PDB in PyMOL or ChimeraX to view the 3D structure.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
