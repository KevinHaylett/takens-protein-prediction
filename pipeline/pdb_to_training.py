"""
PDB to Training Format: pdb_to_training.py
Converts PDB CSV files (from pdb_to_csv.py) to training-ready format.

Input:  CSV with columns [residue_name, chain_id, residue_seq, x, y, z, ...]
Output: Training samples with [AA_sequence, Cα_coords_x, Cα_coords_y, Cα_coords_z]
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional, List
import argparse
from tqdm import tqdm


class PDBToTrainingConverter:
    """Convert PDB CSV files to training format."""
    
    def __init__(
        self,
        max_seq_len: int = 256,
        center_coords: bool = True,
        normalize_coords: bool = False
    ):
        """
        Args:
            max_seq_len: Maximum sequence length (pad/truncate to this)
            center_coords: If True, center coordinates at origin
            normalize_coords: If True, normalize coordinates (not recommended)
        """
        self.max_seq_len = max_seq_len
        self.center_coords = center_coords
        self.normalize_coords = normalize_coords
    
    def load_pdb_csv(self, csv_file: str) -> pd.DataFrame:
        """
        Load PDB CSV and filter for CA atoms only.
        
        Args:
            csv_file: Path to CSV file from pdb_to_csv.py
            
        Returns:
            DataFrame with CA atoms sorted by residue sequence
        """
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            return None
        
        # Filter for CA atoms only
        if 'atom_name' not in df.columns:
            print(f"Error: 'atom_name' column not found in {csv_file}")
            return None
        
        ca_df = df[df['atom_name'] == 'CA'].copy()
        
        if len(ca_df) == 0:
            print(f"Warning: No CA atoms found in {csv_file}")
            return None
        
        # Sort by residue sequence number
        ca_df = ca_df.sort_values('residue_seq').reset_index(drop=True)
        
        return ca_df
    
    def process_coordinates(
        self,
        coords: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """
        Process coordinates: center and optionally normalize.
        
        Args:
            coords: [seq_len, 3] array of (x, y, z) coordinates
            
        Returns:
            processed_coords: Processed coordinates
            transform_info: Dictionary with transformation parameters
        """
        transform_info = {}
        
        # Center at origin (recommended!)
        if self.center_coords:
            centroid = coords.mean(axis=0)
            coords = coords - centroid
            transform_info['centroid'] = centroid
        
        # Normalize (optional, not usually recommended for proteins)
        if self.normalize_coords:
            std = coords.std()
            if std > 0:
                coords = coords / std
                transform_info['std'] = std
            else:
                transform_info['std'] = 1.0
        
        return coords, transform_info
    
    def pad_or_truncate(
        self,
        sequence: List[str],
        coords: np.ndarray
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Pad or truncate sequence and coordinates to max_seq_len.
        
        Args:
            sequence: List of amino acid 3-letter codes
            coords: [seq_len, 3] coordinates
            
        Returns:
            padded_sequence: Padded/truncated sequence
            padded_coords: Padded/truncated coordinates
            mask: Boolean mask (True = real residue, False = padding)
        """
        seq_len = len(sequence)
        
        # Create mask (True for real residues)
        mask = np.zeros(self.max_seq_len, dtype=bool)
        
        if seq_len >= self.max_seq_len:
            # Truncate
            padded_sequence = sequence[:self.max_seq_len]
            padded_coords = coords[:self.max_seq_len]
            mask[:] = True
        else:
            # Pad
            padded_sequence = sequence + ['GAP'] * (self.max_seq_len - seq_len)
            padded_coords = np.zeros((self.max_seq_len, 3), dtype=np.float32)
            padded_coords[:seq_len] = coords
            mask[:seq_len] = True
        
        return padded_sequence, padded_coords, mask
    
    def convert_single_protein(
        self,
        csv_file: str,
        return_dict: bool = True
    ) -> Optional[dict]:
        """
        Convert a single PDB CSV file to training format.
        
        Args:
            csv_file: Path to CSV file
            return_dict: If True, return dictionary; if False, return tuple
            
        Returns:
            Dictionary with:
                - sequence: List of amino acid 3-letter codes
                - coords_x: [max_seq_len] x coordinates
                - coords_y: [max_seq_len] y coordinates
                - coords_z: [max_seq_len] z coordinates
                - mask: [max_seq_len] boolean mask
                - pdb_id: PDB identifier
                - seq_len: Original sequence length
                - transform_info: Transformation parameters
        """
        # Load and filter
        ca_df = self.load_pdb_csv(csv_file)
        if ca_df is None:
            return None
        
        # Check for missing residues
        residue_nums = ca_df['residue_seq'].values
        if len(residue_nums) > 1:
            gaps = np.diff(residue_nums)
            if np.any(gaps > 1):
                print(f"Warning: Missing residues detected in {csv_file}")
        
        # Extract sequence and coordinates
        sequence = ca_df['residue_name'].tolist()
        coords = ca_df[['x', 'y', 'z']].values.astype(np.float32)
        
        original_seq_len = len(sequence)
        
        # Process coordinates (center, normalize)
        coords, transform_info = self.process_coordinates(coords)
        
        # Pad or truncate
        sequence, coords, mask = self.pad_or_truncate(sequence, coords)
        
        # Get PDB ID from filename
        pdb_id = Path(csv_file).stem
        
        if return_dict:
            return {
                'sequence': sequence,
                'coords_x': coords[:, 0],
                'coords_y': coords[:, 1],
                'coords_z': coords[:, 2],
                'mask': mask,
                'pdb_id': pdb_id,
                'seq_len': original_seq_len,
                'transform_info': transform_info
            }
        else:
            return (sequence, coords[:, 0], coords[:, 1], coords[:, 2], 
                   mask, pdb_id, original_seq_len, transform_info)
    
    def convert_directory(
        self,
        input_dir: str,
        output_dir: str,
        save_format: str = 'pt'
    ):
        """
        Convert all CSV files in a directory.
        
        Args:
            input_dir: Directory containing PDB CSV files
            output_dir: Directory to save processed files
            save_format: 'pt' (PyTorch) or 'npz' (NumPy)
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all CSV files
        csv_files = list(input_path.glob('*.csv'))
        
        if len(csv_files) == 0:
            print(f"No CSV files found in {input_dir}")
            return
        
        print(f"Found {len(csv_files)} CSV files")
        print(f"Processing with max_seq_len={self.max_seq_len}...")
        
        successful = 0
        failed = 0
        skipped_long = 0
        
        for csv_file in tqdm(csv_files, desc="Converting"):
            # Check sequence length before processing
            ca_df = self.load_pdb_csv(csv_file)
            if ca_df is None:
                failed += 1
                continue
            
            if len(ca_df) > self.max_seq_len:
                # Skip proteins longer than max_seq_len
                skipped_long += 1
                continue
            
            # Convert
            result = self.convert_single_protein(csv_file)
            if result is None:
                failed += 1
                continue
            
            # Save
            pdb_id = result['pdb_id']
            output_file = output_path / f"{pdb_id}.{save_format}"
            
            if save_format == 'pt':
                # Save as PyTorch tensors
                torch_data = {
                    'sequence': result['sequence'],  # Keep as list
                    'coords_x': torch.tensor(result['coords_x']),
                    'coords_y': torch.tensor(result['coords_y']),
                    'coords_z': torch.tensor(result['coords_z']),
                    'mask': torch.tensor(result['mask']),
                    'pdb_id': result['pdb_id'],
                    'seq_len': result['seq_len'],
                    'transform_info': result['transform_info']
                }
                torch.save(torch_data, output_file)
                
            elif save_format == 'npz':
                # Save as NumPy compressed
                np.savez_compressed(
                    output_file,
                    sequence=result['sequence'],
                    coords_x=result['coords_x'],
                    coords_y=result['coords_y'],
                    coords_z=result['coords_z'],
                    mask=result['mask'],
                    pdb_id=result['pdb_id'],
                    seq_len=result['seq_len'],
                    **result['transform_info']
                )
            
            successful += 1
        
        print("\n" + "=" * 60)
        print(f"Conversion complete!")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Skipped (too long): {skipped_long}")
        print(f"  Output directory: {output_dir}")
        print("=" * 60)
    
    def get_statistics(self, input_dir: str) -> dict:
        """
        Get statistics about PDB files in directory.
        
        Args:
            input_dir: Directory containing PDB CSV files
            
        Returns:
            Dictionary with statistics
        """
        input_path = Path(input_dir)
        csv_files = list(input_path.glob('*.csv'))
        
        lengths = []
        valid_files = 0
        
        print("Analyzing dataset...")
        for csv_file in tqdm(csv_files):
            ca_df = self.load_pdb_csv(csv_file)
            if ca_df is not None:
                lengths.append(len(ca_df))
                valid_files += 1
        
        if len(lengths) == 0:
            return {}
        
        lengths = np.array(lengths)
        
        stats = {
            'total_files': len(csv_files),
            'valid_files': valid_files,
            'min_length': int(lengths.min()),
            'max_length': int(lengths.max()),
            'mean_length': float(lengths.mean()),
            'median_length': float(np.median(lengths)),
            'std_length': float(lengths.std()),
            'lengths': lengths.tolist()
        }
        
        # Count by bins
        bins = [0, 100, 200, 300, 400, 512, 1024, np.inf]
        labels = ['<100', '100-200', '200-300', '300-400', '400-512', '512-1024', '>1024']
        hist, _ = np.histogram(lengths, bins=bins)
        
        stats['length_distribution'] = dict(zip(labels, hist.tolist()))
        
        return stats


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Convert PDB CSV files to training format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file
  python pdb_to_training.py --input 1A3I.csv --output processed/
  
  # Convert directory
  python pdb_to_training.py --input csv_data/ --output processed/ --max_len 256
  
  # Get statistics
  python pdb_to_training.py --input csv_data/ --stats_only
  
  # Don't center coordinates
  python pdb_to_training.py --input csv_data/ --output processed/ --no_center
        """
    )
    
    parser.add_argument('--input', required=True, help='Input CSV file or directory')
    parser.add_argument('--output', help='Output directory for processed files')
    parser.add_argument('--max_len', type=int, default=256, 
                       help='Maximum sequence length (default: 256)')
    parser.add_argument('--format', choices=['pt', 'npz'], default='pt',
                       help='Output format: pt (PyTorch) or npz (NumPy)')
    parser.add_argument('--no_center', action='store_true',
                       help='Do not center coordinates at origin')
    parser.add_argument('--normalize', action='store_true',
                       help='Normalize coordinates (not recommended)')
    parser.add_argument('--stats_only', action='store_true',
                       help='Only show statistics, do not convert')
    
    args = parser.parse_args()
    
    # Create converter
    converter = PDBToTrainingConverter(
        max_seq_len=args.max_len,
        center_coords=not args.no_center,
        normalize_coords=args.normalize
    )
    
    input_path = Path(args.input)
    
    # Statistics only
    if args.stats_only:
        if input_path.is_file():
            print("Stats only mode requires a directory, not a single file")
            return
        
        stats = converter.get_statistics(args.input)
        
        print("\n" + "=" * 60)
        print("Dataset Statistics")
        print("=" * 60)
        print(f"Total files: {stats['total_files']}")
        print(f"Valid files: {stats['valid_files']}")
        print(f"\nSequence length statistics:")
        print(f"  Min: {stats['min_length']}")
        print(f"  Max: {stats['max_length']}")
        print(f"  Mean: {stats['mean_length']:.1f}")
        print(f"  Median: {stats['median_length']:.1f}")
        print(f"  Std: {stats['std_length']:.1f}")
        print(f"\nLength distribution:")
        for label, count in stats['length_distribution'].items():
            pct = 100 * count / stats['valid_files']
            print(f"  {label:12s}: {count:5d} ({pct:5.1f}%)")
        print(f"\nWith max_len={args.max_len}:")
        total_within = sum(1 for l in stats['lengths'] if l <= args.max_len)
        pct_within = 100 * total_within / stats['valid_files']
        print(f"  {total_within} proteins ({pct_within:.1f}%) fit within max_len")
        print("=" * 60)
        return
    
    # Check output directory
    if args.output is None:
        print("Error: --output required for conversion")
        return
    
    # Convert single file or directory
    if input_path.is_file():
        # Single file
        print(f"Converting single file: {args.input}")
        result = converter.convert_single_protein(args.input)
        
        if result is None:
            print("Conversion failed")
            return
        
        # Save
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"{result['pdb_id']}.{args.format}"
        
        if args.format == 'pt':
            torch_data = {
                'sequence': result['sequence'],
                'coords_x': torch.tensor(result['coords_x']),
                'coords_y': torch.tensor(result['coords_y']),
                'coords_z': torch.tensor(result['coords_z']),
                'mask': torch.tensor(result['mask']),
                'pdb_id': result['pdb_id'],
                'seq_len': result['seq_len'],
                'transform_info': result['transform_info']
            }
            torch.save(torch_data, output_file)
        else:
            np.savez_compressed(
                output_file,
                sequence=result['sequence'],
                coords_x=result['coords_x'],
                coords_y=result['coords_y'],
                coords_z=result['coords_z'],
                mask=result['mask'],
                pdb_id=result['pdb_id'],
                seq_len=result['seq_len'],
                **result['transform_info']
            )
        
        print(f"Saved to {output_file}")
        print(f"Sequence length: {result['seq_len']}")
        
    else:
        # Directory
        converter.convert_directory(args.input, args.output, args.format)


if __name__ == "__main__":
    main()
