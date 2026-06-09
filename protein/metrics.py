"""
Protein Structure Metrics and Analysis: metrics.py
Comprehensive evaluation metrics for protein structure prediction.

Includes:
- RMSD (Root Mean Square Deviation)
- GDT (Global Distance Test)
- TM-score (Template Modeling score)
- Per-residue analysis
- Secondary structure comparison
- Contact map accuracy
"""

import numpy as np
import torch
from typing import Tuple, Dict, Optional
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import pdist, squareform


class StructureMetrics:
    """Comprehensive metrics for protein structure prediction."""
    
    @staticmethod
    def align_structures(
        pred_coords: np.ndarray,
        true_coords: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Align predicted structure to true structure using Kabsch algorithm.
        
        Finds optimal rotation and translation to minimize RMSD.
        
        Args:
            pred_coords: [n, 3] predicted coordinates
            true_coords: [n, 3] true coordinates
            
        Returns:
            pred_aligned: [n, 3] aligned predicted coordinates
            true_centered: [n, 3] centered true coordinates
            transform_info: Dictionary with rotation matrix and translation
        """
        assert pred_coords.shape == true_coords.shape
        
        # Center both structures
        pred_centroid = pred_coords.mean(axis=0)
        true_centroid = true_coords.mean(axis=0)
        
        pred_centered = pred_coords - pred_centroid
        true_centered = true_coords - true_centroid
        
        # Compute covariance matrix
        H = pred_centered.T @ true_centered
        
        # Singular Value Decomposition
        U, S, Vt = np.linalg.svd(H)
        
        # Compute rotation matrix
        R = Vt.T @ U.T
        
        # Ensure proper rotation (det(R) = 1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Apply rotation
        pred_aligned = pred_centered @ R.T
        
        transform_info = {
            'rotation': R,
            'pred_centroid': pred_centroid,
            'true_centroid': true_centroid
        }
        
        return pred_aligned, true_centered, transform_info
    
    @staticmethod
    def compute_rmsd(
        pred_coords: np.ndarray,
        true_coords: np.ndarray,
        align: bool = True
    ) -> float:
        """
        Compute Root Mean Square Deviation (RMSD) in Ångströms.
        
        Args:
            pred_coords: [n, 3] predicted coordinates
            true_coords: [n, 3] true coordinates
            align: If True, align structures first (recommended)
            
        Returns:
            RMSD in Ångströms
        """
        if align:
            pred_aligned, true_aligned, _ = StructureMetrics.align_structures(
                pred_coords, true_coords
            )
        else:
            pred_aligned = pred_coords
            true_aligned = true_coords
        
        # Compute squared differences
        sq_diff = ((pred_aligned - true_aligned) ** 2).sum(axis=1)
        
        # RMSD
        rmsd = np.sqrt(sq_diff.mean())
        
        return rmsd
    
    @staticmethod
    def compute_per_residue_rmsd(
        pred_coords: np.ndarray,
        true_coords: np.ndarray,
        align: bool = True
    ) -> np.ndarray:
        """
        Compute per-residue RMSD.
        
        Useful for identifying problematic regions (loops, flexible regions).
        
        Args:
            pred_coords: [n, 3] predicted coordinates
            true_coords: [n, 3] true coordinates
            align: If True, align structures first
            
        Returns:
            per_residue_rmsd: [n] array of RMSD per residue
        """
        if align:
            pred_aligned, true_aligned, _ = StructureMetrics.align_structures(
                pred_coords, true_coords
            )
        else:
            pred_aligned = pred_coords
            true_aligned = true_coords
        
        # Compute per-residue squared deviation
        sq_diff = ((pred_aligned - true_aligned) ** 2).sum(axis=1)
        
        # Per-residue RMSD
        per_residue_rmsd = np.sqrt(sq_diff)
        
        return per_residue_rmsd
    
    @staticmethod
    def compute_gdt_ts(
        pred_coords: np.ndarray,
        true_coords: np.ndarray,
        cutoffs: list = [1.0, 2.0, 4.0, 8.0]
    ) -> float:
        """
        Compute Global Distance Test (GDT_TS) score.
        
        GDT_TS measures the percentage of residues under distance cutoffs.
        Higher is better (0-100 scale).
        
        Standard cutoffs: 1, 2, 4, 8 Ångströms
        
        Args:
            pred_coords: [n, 3] predicted coordinates
            true_coords: [n, 3] true coordinates
            cutoffs: List of distance cutoffs in Ångströms
            
        Returns:
            GDT_TS score (0-100)
        """
        # Align structures
        pred_aligned, true_aligned, _ = StructureMetrics.align_structures(
            pred_coords, true_coords
        )
        
        # Compute distances
        distances = np.sqrt(((pred_aligned - true_aligned) ** 2).sum(axis=1))
        
        # Count residues under each cutoff
        n = len(distances)
        percentages = []
        
        for cutoff in cutoffs:
            count = (distances < cutoff).sum()
            percentage = 100.0 * count / n
            percentages.append(percentage)
        
        # GDT_TS is the average of percentages
        gdt_ts = np.mean(percentages)
        
        return gdt_ts
    
    @staticmethod
    def compute_tm_score(
        pred_coords: np.ndarray,
        true_coords: np.ndarray
    ) -> float:
        """
        Compute Template Modeling (TM) score.
        
        TM-score is length-independent metric for protein similarity.
        Range: (0, 1], where 1 = identical, >0.5 = same fold
        
        Args:
            pred_coords: [n, 3] predicted coordinates
            true_coords: [n, 3] true coordinates
            
        Returns:
            TM-score (0-1)
        """
        # Align structures
        pred_aligned, true_aligned, _ = StructureMetrics.align_structures(
            pred_coords, true_coords
        )
        
        # Length normalization
        L = len(pred_coords)
        d0 = 1.24 * (L - 15) ** (1.0/3.0) - 1.8  # Length-dependent scale
        
        # Compute distances
        distances = np.sqrt(((pred_aligned - true_aligned) ** 2).sum(axis=1))
        
        # TM-score formula
        tm_score = (1.0 / (1.0 + (distances / d0) ** 2)).sum() / L
        
        return tm_score
    
    @staticmethod
    def compute_contact_map(
        coords: np.ndarray,
        threshold: float = 8.0
    ) -> np.ndarray:
        """
        Compute contact map from coordinates.
        
        Contact = distance < threshold (typically 8 Å for Cα atoms)
        
        Args:
            coords: [n, 3] coordinates
            threshold: Distance threshold in Ångströms
            
        Returns:
            contact_map: [n, n] binary contact map
        """
        # Compute pairwise distances
        dist_matrix = squareform(pdist(coords))
        
        # Binary contact map
        contact_map = (dist_matrix < threshold).astype(int)
        
        return contact_map
    
    @staticmethod
    def compute_contact_accuracy(
        pred_coords: np.ndarray,
        true_coords: np.ndarray,
        threshold: float = 8.0,
        min_separation: int = 6
    ) -> Dict[str, float]:
        """
        Compute contact prediction accuracy.
        
        Evaluates:
        - Precision: fraction of predicted contacts that are true
        - Recall: fraction of true contacts that are predicted
        - F1 score: harmonic mean of precision and recall
        
        Args:
            pred_coords: [n, 3] predicted coordinates
            true_coords: [n, 3] true coordinates
            threshold: Distance threshold for contacts
            min_separation: Minimum sequence separation (ignore local contacts)
            
        Returns:
            Dictionary with precision, recall, f1
        """
        # Compute contact maps
        pred_contacts = StructureMetrics.compute_contact_map(pred_coords, threshold)
        true_contacts = StructureMetrics.compute_contact_map(true_coords, threshold)
        
        # Mask for sequence separation
        n = len(pred_coords)
        mask = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + min_separation, n):
                mask[i, j] = True
                mask[j, i] = True
        
        # Apply mask
        pred_contacts_masked = pred_contacts[mask]
        true_contacts_masked = true_contacts[mask]
        
        # Compute metrics
        tp = ((pred_contacts_masked == 1) & (true_contacts_masked == 1)).sum()
        fp = ((pred_contacts_masked == 1) & (true_contacts_masked == 0)).sum()
        fn = ((pred_contacts_masked == 0) & (true_contacts_masked == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    @staticmethod
    def compute_distance_matrix_rmse(
        pred_coords: np.ndarray,
        true_coords: np.ndarray
    ) -> float:
        """
        Compute RMSE of distance matrices.
        
        Rotation-invariant metric.
        
        Args:
            pred_coords: [n, 3] predicted coordinates
            true_coords: [n, 3] true coordinates
            
        Returns:
            RMSE of distance matrices
        """
        # Compute distance matrices
        pred_dist = squareform(pdist(pred_coords))
        true_dist = squareform(pdist(true_coords))
        
        # RMSE
        rmse = np.sqrt(((pred_dist - true_dist) ** 2).mean())
        
        return rmse


class SecondaryStructureMetrics:
    """Metrics for secondary structure prediction."""
    
    # Secondary structure definitions (simplified DSSP)
    SS_HELIX = 'H'  # α-helix
    SS_SHEET = 'E'  # β-sheet
    SS_COIL = 'C'   # coil/loop
    
    @staticmethod
    def assign_simple_ss(coords: np.ndarray) -> np.ndarray:
        """
        Simple secondary structure assignment based on Cα distances.
        
        This is a simplified version - true DSSP requires backbone atoms.
        
        Args:
            coords: [n, 3] Cα coordinates
            
        Returns:
            ss: [n] array of secondary structure codes ('H', 'E', 'C')
        """
        n = len(coords)
        ss = np.array(['C'] * n)
        
        # Compute Cα distances
        dist_matrix = squareform(pdist(coords))
        
        # Helix: residues i and i+4 are close (< 6.5 Å)
        for i in range(n - 4):
            if dist_matrix[i, i+4] < 6.5:
                ss[i:i+5] = 'H'
        
        # Sheet: pairs of residues far in sequence but close in space
        for i in range(n):
            for j in range(i + 5, n):
                if dist_matrix[i, j] < 6.0:
                    ss[i] = 'E'
                    ss[j] = 'E'
        
        return ss
    
    @staticmethod
    def compute_ss_accuracy(
        pred_ss: np.ndarray,
        true_ss: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute secondary structure prediction accuracy.
        
        Args:
            pred_ss: [n] predicted SS codes
            true_ss: [n] true SS codes
            
        Returns:
            Dictionary with Q3 (3-state accuracy) and per-class metrics
        """
        # Overall Q3 accuracy
        q3 = (pred_ss == true_ss).mean()
        
        # Per-class metrics
        metrics = {'q3': q3}
        
        for ss_type in ['H', 'E', 'C']:
            mask = (true_ss == ss_type)
            if mask.sum() > 0:
                acc = (pred_ss[mask] == ss_type).mean()
                metrics[f'{ss_type}_accuracy'] = acc
        
        return metrics


def evaluate_prediction(
    pred_coords: np.ndarray,
    true_coords: np.ndarray,
    sequence: Optional[str] = None,
    true_ss: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Dict:
    """
    Comprehensive evaluation of a single prediction.
    
    Args:
        pred_coords: [n, 3] predicted Cα coordinates
        true_coords: [n, 3] true Cα coordinates
        sequence: Amino acid sequence (optional)
        true_ss: True secondary structure (optional)
        verbose: Print results
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # RMSD
    rmsd = StructureMetrics.compute_rmsd(pred_coords, true_coords, align=True)
    rmsd_unaligned = StructureMetrics.compute_rmsd(pred_coords, true_coords, align=False)
    metrics['rmsd'] = rmsd
    metrics['rmsd_unaligned'] = rmsd_unaligned
    
    # Per-residue RMSD
    per_res_rmsd = StructureMetrics.compute_per_residue_rmsd(pred_coords, true_coords)
    metrics['per_residue_rmsd'] = per_res_rmsd
    metrics['rmsd_mean'] = per_res_rmsd.mean()
    metrics['rmsd_std'] = per_res_rmsd.std()
    metrics['rmsd_max'] = per_res_rmsd.max()
    
    # GDT_TS
    gdt_ts = StructureMetrics.compute_gdt_ts(pred_coords, true_coords)
    metrics['gdt_ts'] = gdt_ts
    
    # TM-score
    tm_score = StructureMetrics.compute_tm_score(pred_coords, true_coords)
    metrics['tm_score'] = tm_score
    
    # Contact accuracy
    contact_metrics = StructureMetrics.compute_contact_accuracy(pred_coords, true_coords)
    metrics.update({f'contact_{k}': v for k, v in contact_metrics.items()})
    
    # Distance matrix RMSE
    dm_rmse = StructureMetrics.compute_distance_matrix_rmse(pred_coords, true_coords)
    metrics['distance_matrix_rmse'] = dm_rmse
    
    # Secondary structure (if provided or can be computed)
    if true_ss is not None:
        pred_ss = SecondaryStructureMetrics.assign_simple_ss(pred_coords)
        ss_metrics = SecondaryStructureMetrics.compute_ss_accuracy(pred_ss, true_ss)
        metrics.update({f'ss_{k}': v for k, v in ss_metrics.items()})
    
    # Print summary
    if verbose:
        print("\n" + "=" * 60)
        print("Structure Prediction Evaluation")
        print("=" * 60)
        print(f"Sequence length: {len(pred_coords)}")
        if sequence:
            print(f"Sequence: {sequence[:50]}...")
        print(f"\nGlobal Metrics:")
        print(f"  RMSD (aligned):    {metrics['rmsd']:.2f} Å")
        print(f"  RMSD (unaligned):  {metrics['rmsd_unaligned']:.2f} Å")
        print(f"  GDT_TS:            {metrics['gdt_ts']:.2f}")
        print(f"  TM-score:          {metrics['tm_score']:.4f}")
        print(f"\nPer-Residue RMSD:")
        print(f"  Mean:  {metrics['rmsd_mean']:.2f} Å")
        print(f"  Std:   {metrics['rmsd_std']:.2f} Å")
        print(f"  Max:   {metrics['rmsd_max']:.2f} Å")
        print(f"\nContact Prediction:")
        print(f"  Precision: {metrics['contact_precision']:.3f}")
        print(f"  Recall:    {metrics['contact_recall']:.3f}")
        print(f"  F1:        {metrics['contact_f1']:.3f}")
        print(f"\nDistance Matrix RMSE: {metrics['distance_matrix_rmse']:.2f} Å")
        
        if 'ss_q3' in metrics:
            print(f"\nSecondary Structure:")
            print(f"  Q3 accuracy: {metrics['ss_q3']:.3f}")
    
    return metrics


if __name__ == "__main__":
    # Test the metrics
    print("=" * 60)
    print("Testing Structure Metrics")
    print("=" * 60)
    
    # Create dummy coordinates
    np.random.seed(42)
    n = 50
    
    # True structure (helical-ish)
    t = np.linspace(0, 4*np.pi, n)
    true_coords = np.stack([
        np.cos(t),
        np.sin(t),
        t * 0.5
    ], axis=-1)
    
    # Predicted structure (similar but with noise)
    pred_coords = true_coords + np.random.randn(n, 3) * 0.5
    
    # Evaluate
    metrics = evaluate_prediction(pred_coords, true_coords, verbose=True)
    
    print("\n" + "=" * 60)
    print("All metrics computed successfully! ✓")
