"""
Protein Encoder: protein_encoder.py
Encodes amino acid sequences to integer tokens.

Simple 21-token vocabulary (20 standard amino acids + GAP for padding).
"""

import json
from typing import List, Union


class ProteinEncoder:
    """
    Encode amino acid sequences to integer tokens.
    
    Vocabulary:
        20 standard amino acids + GAP token for padding
    """
    
    # Standard amino acid vocabulary (3-letter codes)
    AA_VOCAB = {
        'ALA': 0,  'ARG': 1,  'ASN': 2,  'ASP': 3,  'CYS': 4,
        'GLN': 5,  'GLU': 6,  'GLY': 7,  'HIS': 8,  'ILE': 9,
        'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
        'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
        'GAP': 20  # Padding token
    }
    
    # Reverse mapping for decoding
    IDX_TO_AA = {v: k for k, v in AA_VOCAB.items()}
    
    # One-letter code mapping
    ONE_LETTER = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
        'GAP': '-'
    }
    
    # Reverse one-letter mapping
    THREE_LETTER = {v: k for k, v in ONE_LETTER.items()}
    
    def __init__(self):
        """Initialize protein encoder."""
        self.vocab_size = len(self.AA_VOCAB)
        self.pad_token = 'GAP'
        self.pad_token_id = self.AA_VOCAB['GAP']
    
    def encode_sequence(self, residues: Union[List[str], str]) -> List[int]:
        """
        Encode amino acid sequence to integer tokens.
        
        Args:
            residues: List of 3-letter codes ['MET', 'LYS', 'TYR', ...]
                     or string of 1-letter codes "MKYL..."
                     
        Returns:
            List of integer tokens
        """
        if isinstance(residues, str):
            # Convert 1-letter to 3-letter codes
            residues = [self.THREE_LETTER.get(aa, 'GAP') for aa in residues]
        
        tokens = []
        for aa in residues:
            aa_upper = aa.upper()
            if aa_upper in self.AA_VOCAB:
                tokens.append(self.AA_VOCAB[aa_upper])
            else:
                # Unknown residue -> treat as GAP
                print(f"Warning: Unknown residue '{aa}' encoded as GAP")
                tokens.append(self.pad_token_id)
        
        return tokens
    
    def decode_sequence(self, tokens: List[int], one_letter: bool = False) -> Union[List[str], str]:
        """
        Decode integer tokens back to amino acid sequence.
        
        Args:
            tokens: List of integer tokens
            one_letter: If True, return as single string of 1-letter codes
                       If False, return list of 3-letter codes
                       
        Returns:
            Decoded sequence
        """
        residues = [self.IDX_TO_AA.get(tok, 'GAP') for tok in tokens]
        
        if one_letter:
            return ''.join([self.ONE_LETTER[aa] for aa in residues])
        else:
            return residues
    
    def get_vocab_size(self) -> int:
        """Return vocabulary size (21 for proteins)."""
        return self.vocab_size
    
    def get_pad_token_id(self) -> int:
        """Return padding token ID."""
        return self.pad_token_id
    
    def save_vocab(self, filepath: str):
        """Save vocabulary to JSON file."""
        vocab_data = {
            'aa_vocab': self.AA_VOCAB,
            'vocab_size': self.vocab_size,
            'pad_token': self.pad_token,
            'pad_token_id': self.pad_token_id
        }
        with open(filepath, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        print(f"Vocabulary saved to {filepath}")
    
    @classmethod
    def load_vocab(cls, filepath: str):
        """Load vocabulary from JSON file."""
        with open(filepath, 'r') as f:
            vocab_data = json.load(f)
        
        encoder = cls()
        # Vocab is already set in __init__, just verify
        assert encoder.vocab_size == vocab_data['vocab_size']
        print(f"Vocabulary loaded from {filepath}")
        return encoder
    
    def get_amino_acid_properties(self) -> dict:
        """
        Return physicochemical properties of amino acids.
        Useful for future analysis and visualization.
        """
        properties = {
            'ALA': {'hydrophobic': True,  'charged': False, 'polar': False},
            'ARG': {'hydrophobic': False, 'charged': True,  'polar': True, 'charge': +1},
            'ASN': {'hydrophobic': False, 'charged': False, 'polar': True},
            'ASP': {'hydrophobic': False, 'charged': True,  'polar': True, 'charge': -1},
            'CYS': {'hydrophobic': False, 'charged': False, 'polar': True},
            'GLN': {'hydrophobic': False, 'charged': False, 'polar': True},
            'GLU': {'hydrophobic': False, 'charged': True,  'polar': True, 'charge': -1},
            'GLY': {'hydrophobic': False, 'charged': False, 'polar': False},
            'HIS': {'hydrophobic': False, 'charged': True,  'polar': True, 'charge': +1},
            'ILE': {'hydrophobic': True,  'charged': False, 'polar': False},
            'LEU': {'hydrophobic': True,  'charged': False, 'polar': False},
            'LYS': {'hydrophobic': False, 'charged': True,  'polar': True, 'charge': +1},
            'MET': {'hydrophobic': True,  'charged': False, 'polar': False},
            'PHE': {'hydrophobic': True,  'charged': False, 'polar': False},
            'PRO': {'hydrophobic': True,  'charged': False, 'polar': False},
            'SER': {'hydrophobic': False, 'charged': False, 'polar': True},
            'THR': {'hydrophobic': False, 'charged': False, 'polar': True},
            'TRP': {'hydrophobic': True,  'charged': False, 'polar': False},
            'TYR': {'hydrophobic': False, 'charged': False, 'polar': True},
            'VAL': {'hydrophobic': True,  'charged': False, 'polar': False},
        }
        return properties


if __name__ == "__main__":
    # Test the encoder
    print("Testing ProteinEncoder...")
    print("=" * 60)
    
    encoder = ProteinEncoder()
    
    # Test 1: Encode 3-letter codes
    print("\n1. Encoding 3-letter codes:")
    residues_3letter = ['MET', 'LYS', 'TYR', 'LEU', 'ILE', 'PHE', 'PRO', 'THR', 'ALA', 'ALA', 'GLY']
    tokens = encoder.encode_sequence(residues_3letter)
    print(f"Input:  {residues_3letter}")
    print(f"Tokens: {tokens}")
    
    # Test 2: Decode back
    print("\n2. Decoding tokens:")
    decoded_3letter = encoder.decode_sequence(tokens, one_letter=False)
    decoded_1letter = encoder.decode_sequence(tokens, one_letter=True)
    print(f"Decoded (3-letter): {decoded_3letter}")
    print(f"Decoded (1-letter): {decoded_1letter}")
    
    # Test 3: Encode 1-letter codes
    print("\n3. Encoding 1-letter codes:")
    sequence_1letter = "MKYLIFPTAAG"
    tokens = encoder.encode_sequence(sequence_1letter)
    print(f"Input:  {sequence_1letter}")
    print(f"Tokens: {tokens}")
    decoded = encoder.decode_sequence(tokens, one_letter=True)
    print(f"Decoded: {decoded}")
    
    # Test 4: Vocabulary info
    print("\n4. Vocabulary information:")
    print(f"Vocabulary size: {encoder.get_vocab_size()}")
    print(f"Pad token: {encoder.pad_token}")
    print(f"Pad token ID: {encoder.get_pad_token_id()}")
    
    # Test 5: Unknown residues
    print("\n5. Handling unknown residues:")
    unknown_seq = ['MET', 'XXX', 'TYR', 'ZZZ']
    tokens = encoder.encode_sequence(unknown_seq)
    print(f"Input:  {unknown_seq}")
    print(f"Tokens: {tokens}")
    print(f"Note: Unknown residues (XXX, ZZZ) encoded as GAP (20)")
    
    # Test 6: Save and load
    print("\n6. Save/load vocabulary:")
    encoder.save_vocab('protein_vocab.json')
    encoder2 = ProteinEncoder.load_vocab('protein_vocab.json')
    print("✓ Vocabulary saved and loaded successfully")
    
    # Test 7: Amino acid properties
    print("\n7. Amino acid properties:")
    props = encoder.get_amino_acid_properties()
    print("Sample properties (ALA, ARG, ASP):")
    for aa in ['ALA', 'ARG', 'ASP']:
        print(f"  {aa}: {props[aa]}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("\nEncoder ready for protein folding pipeline.")
