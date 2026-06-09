"""
Protein Dataset: protein_dataset.py
PyTorch Dataset class for loading protein structures.

Loads preprocessed protein data from pdb_to_training.py output.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from protein.protein_encoder import ProteinEncoder


class ProteinDataset(Dataset):
    """
    Dataset for protein sequences with Cα coordinates.
    
    Each sample contains:
        - input_ids: [seq_len] amino acid tokens
        - coords_x: [seq_len] x coordinates
        - coords_y: [seq_len] y coordinates
        - coords_z: [seq_len] z coordinates
        - mask: [seq_len] boolean mask (True = real residue, False = padding)
        - pdb_id: protein identifier string
    """
    
    def __init__(
        self,
        data_dir: str,
        encoder: ProteinEncoder,
        max_seq_len: int = 256,
        file_format: str = 'pt',
        filter_by_length: bool = True,
        verbose: bool = True
    ):
        """
        Args:
            data_dir: Directory containing processed protein files
            encoder: ProteinEncoder instance
            max_seq_len: Maximum sequence length (for validation)
            file_format: 'pt' (PyTorch) or 'npz' (NumPy)
            filter_by_length: If True, skip proteins longer than max_seq_len
            verbose: Print loading information
        """
        self.data_dir = Path(data_dir)
        self.encoder = encoder
        self.max_seq_len = max_seq_len
        self.file_format = file_format
        self.filter_by_length = filter_by_length
        self.verbose = verbose
        
        # Find all data files
        if file_format == 'pt':
            self.data_files = sorted(list(self.data_dir.glob('*.pt')))
        else:
            self.data_files = sorted(list(self.data_dir.glob('*.npz')))
        
        if len(self.data_files) == 0:
            raise ValueError(f"No {file_format} files found in {data_dir}")
        
        # Filter by length if requested
        if filter_by_length:
            self.data_files = self._filter_by_length()
        
        if verbose:
            print(f"Loaded {len(self.data_files)} proteins from {data_dir}")
            if len(self.data_files) > 0:
                # Sample a few to check lengths
                sample_lengths = []
                for i in [0, len(self.data_files)//2, -1]:
                    try:
                        sample = self._load_file(self.data_files[i])
                        sample_lengths.append(sample['seq_len'])
                    except:
                        pass
                if sample_lengths:
                    print(f"Sample sequence lengths: {sample_lengths}")
    
    def _filter_by_length(self) -> List[Path]:
        """Filter out proteins longer than max_seq_len."""
        if not self.filter_by_length:
            return self.data_files
        
        valid_files = []
        skipped = 0
        
        for file_path in self.data_files:
            try:
                sample = self._load_file(file_path)
                if sample['seq_len'] <= self.max_seq_len:
                    valid_files.append(file_path)
                else:
                    skipped += 1
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not load {file_path}: {e}")
                skipped += 1
        
        if self.verbose and skipped > 0:
            print(f"Filtered out {skipped} proteins (>{self.max_seq_len} residues)")
        
        return valid_files
    
    def _load_file(self, file_path: Path) -> Dict:
        """Load a single data file."""
        if self.file_format == 'pt':
            data = torch.load(file_path)
            return data
        else:
            data = np.load(file_path, allow_pickle=True)
            return {
                'sequence': data['sequence'].tolist(),
                'coords_x': data['coords_x'],
                'coords_y': data['coords_y'],
                'coords_z': data['coords_z'],
                'mask': data['mask'],
                'pdb_id': str(data['pdb_id']),
                'seq_len': int(data['seq_len'])
            }
    
    def __len__(self) -> int:
        """Return number of proteins in dataset."""
        return len(self.data_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single protein sample.
        
        Returns:
            Dictionary with:
                - input_ids: [seq_len] LongTensor of amino acid tokens
                - coords_x: [seq_len] FloatTensor
                - coords_y: [seq_len] FloatTensor
                - coords_z: [seq_len] FloatTensor
                - mask: [seq_len] BoolTensor
                - pdb_id: string
        """
        # Load data
        data = self._load_file(self.data_files[idx])
        
        # Encode sequence to tokens
        input_ids = self.encoder.encode_sequence(data['sequence'])
        
        # Convert to tensors
        sample = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'coords_x': torch.tensor(data['coords_x'], dtype=torch.float32),
            'coords_y': torch.tensor(data['coords_y'], dtype=torch.float32),
            'coords_z': torch.tensor(data['coords_z'], dtype=torch.float32),
            'mask': torch.tensor(data['mask'], dtype=torch.bool),
            'pdb_id': data['pdb_id']
        }
        
        return sample
    
    def get_sequence(self, idx: int, one_letter: bool = True) -> str:
        """Get amino acid sequence for a protein."""
        sample = self[idx]
        input_ids = sample['input_ids'].tolist()
        mask = sample['mask'].tolist()
        
        # Only decode valid (non-padded) positions
        valid_ids = [input_ids[i] for i in range(len(input_ids)) if mask[i]]
        sequence = self.encoder.decode_sequence(valid_ids, one_letter=one_letter)
        
        return sequence
    
    def get_coordinates(self, idx: int) -> np.ndarray:
        """
        Get coordinates for a protein.
        
        Returns:
            [seq_len, 3] array (only valid residues, no padding)
        """
        sample = self[idx]
        mask = sample['mask'].numpy()
        
        coords = np.stack([
            sample['coords_x'].numpy(),
            sample['coords_y'].numpy(),
            sample['coords_z'].numpy()
        ], axis=-1)
        
        # Return only valid residues
        return coords[mask]
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        seq_lengths = []
        coord_ranges = {'x': [], 'y': [], 'z': []}
        
        print("Computing dataset statistics...")
        for i in range(len(self)):
            if i % max(1, len(self) // 10) == 0:
                print(f"  {i}/{len(self)}", end='\r')
            
            sample = self[i]
            mask = sample['mask']
            seq_len = mask.sum().item()
            seq_lengths.append(seq_len)
            
            # Get coordinate ranges
            valid_x = sample['coords_x'][mask]
            valid_y = sample['coords_y'][mask]
            valid_z = sample['coords_z'][mask]
            
            coord_ranges['x'].extend([valid_x.min().item(), valid_x.max().item()])
            coord_ranges['y'].extend([valid_y.min().item(), valid_y.max().item()])
            coord_ranges['z'].extend([valid_z.min().item(), valid_z.max().item()])
        
        print(f"  {len(self)}/{len(self)}")
        
        seq_lengths = np.array(seq_lengths)
        
        stats = {
            'num_proteins': len(self),
            'seq_len_min': int(seq_lengths.min()),
            'seq_len_max': int(seq_lengths.max()),
            'seq_len_mean': float(seq_lengths.mean()),
            'seq_len_median': float(np.median(seq_lengths)),
            'seq_len_std': float(seq_lengths.std()),
            'coord_ranges': {
                'x': (min(coord_ranges['x']), max(coord_ranges['x'])),
                'y': (min(coord_ranges['y']), max(coord_ranges['y'])),
                'z': (min(coord_ranges['z']), max(coord_ranges['z']))
            }
        }
        
        return stats


def collate_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    All sequences are already padded to max_seq_len,
    so we just need to stack them.
    """
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'coords_x': torch.stack([item['coords_x'] for item in batch]),
        'coords_y': torch.stack([item['coords_y'] for item in batch]),
        'coords_z': torch.stack([item['coords_z'] for item in batch]),
        'mask': torch.stack([item['mask'] for item in batch]),
        'pdb_id': [item['pdb_id'] for item in batch]
    }


if __name__ == "__main__":
    # Test the dataset
    import argparse
    
    parser = argparse.ArgumentParser(description='Test protein dataset')
    parser.add_argument('--data_dir', required=True, help='Directory with processed proteins')
    parser.add_argument('--max_len', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--stats', action='store_true', help='Show dataset statistics')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Testing ProteinDataset")
    print("=" * 60)
    
    # Create encoder
    encoder = ProteinEncoder()
    
    # Create dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    dataset = ProteinDataset(
        data_dir=args.data_dir,
        encoder=encoder,
        max_seq_len=args.max_len,
        file_format='pt',
        verbose=True
    )
    
    if len(dataset) == 0:
        print("Error: Dataset is empty!")
        exit(1)
    
    # Test single sample
    print(f"\nTesting single sample (index 0)...")
    sample = dataset[0]
    
    print(f"  PDB ID: {sample['pdb_id']}")
    print(f"  Input IDs shape: {sample['input_ids'].shape}")
    print(f"  Coords X shape: {sample['coords_x'].shape}")
    print(f"  Coords Y shape: {sample['coords_y'].shape}")
    print(f"  Coords Z shape: {sample['coords_z'].shape}")
    print(f"  Mask shape: {sample['mask'].shape}")
    print(f"  Sequence length (unpadded): {sample['mask'].sum().item()}")
    
    # Show sequence
    sequence = dataset.get_sequence(0, one_letter=True)
    print(f"  Sequence (1-letter): {sequence[:50]}...")
    
    # Test coordinates
    coords = dataset.get_coordinates(0)
    print(f"  Coordinates shape (unpadded): {coords.shape}")
    print(f"  Coordinate ranges:")
    print(f"    X: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}]")
    print(f"    Y: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")
    print(f"    Z: [{coords[:, 2].min():.2f}, {coords[:, 2].max():.2f}]")
    
    # Test DataLoader
    print(f"\nTesting DataLoader...")
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0  # Use 0 for debugging
    )
    
    batch = next(iter(loader))
    print(f"  Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"  Batch coords_x shape: {batch['coords_x'].shape}")
    print(f"  Batch mask shape: {batch['mask'].shape}")
    print(f"  Batch PDB IDs: {batch['pdb_id']}")
    
    # Dataset statistics (if requested)
    if args.stats:
        print(f"\n{'='*60}")
        print("Dataset Statistics")
        print("=" * 60)
        stats = dataset.get_statistics()
        
        print(f"\nNumber of proteins: {stats['num_proteins']}")
        print(f"\nSequence lengths:")
        print(f"  Min: {stats['seq_len_min']}")
        print(f"  Max: {stats['seq_len_max']}")
        print(f"  Mean: {stats['seq_len_mean']:.1f}")
        print(f"  Median: {stats['seq_len_median']:.1f}")
        print(f"  Std: {stats['seq_len_std']:.1f}")
        print(f"\nCoordinate ranges (Ångströms):")
        for axis in ['x', 'y', 'z']:
            range_min, range_max = stats['coord_ranges'][axis]
            print(f"  {axis.upper()}: [{range_min:.2f}, {range_max:.2f}]")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("\nDataset ready for training.")
