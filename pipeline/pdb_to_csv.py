#!/usr/bin/env python3
"""
PDB to CSV Converter for Protein Folding Training Data

Converts PDB format protein structures to CSV format with:
- Amino acid residue information
- All atom positions (x, y, z coordinates)
- Residue numbers and chain information

Author: Kaevin the Listener
"""

import csv
import argparse
from pathlib import Path
from typing import List, Dict, Tuple


class PDBParser:
    """Parse PDB format files and extract amino acid and atom coordinate data."""
    
    # Standard amino acid three-letter codes
    AMINO_ACIDS = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
    }
    
    def __init__(self, pdb_file: str):
        """Initialize parser with PDB file path."""
        self.pdb_file = Path(pdb_file)
        self.atoms = []
        
    def parse(self) -> List[Dict]:
        """
        Parse PDB file and extract atom records.
        
        Returns:
            List of dictionaries containing atom information
        """
        with open(self.pdb_file, 'r') as f:
            for line in f:
                # Parse ATOM and HETATM records
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    atom_data = self._parse_atom_line(line)
                    if atom_data:
                        self.atoms.append(atom_data)
        
        return self.atoms
    
    def _parse_atom_line(self, line: str) -> Dict:
        """
        Parse a single ATOM/HETATM line from PDB format.
        
        PDB format specification (columns are 1-indexed):
        - Columns 1-6: Record type (ATOM/HETATM)
        - Columns 7-11: Atom serial number
        - Columns 13-16: Atom name
        - Column 17: Alternate location indicator
        - Columns 18-20: Residue name (amino acid)
        - Column 22: Chain identifier
        - Columns 23-26: Residue sequence number
        - Column 27: Insertion code
        - Columns 31-38: X coordinate
        - Columns 39-46: Y coordinate
        - Columns 47-54: Z coordinate
        - Columns 55-60: Occupancy
        - Columns 61-66: Temperature factor
        - Columns 77-78: Element symbol
        """
        try:
            record_type = line[0:6].strip()
            atom_serial = int(line[6:11].strip())
            atom_name = line[12:16].strip()
            alt_loc = line[16:17].strip()
            residue_name = line[17:20].strip()
            chain_id = line[21:22].strip()
            residue_seq = int(line[22:26].strip())
            insertion_code = line[26:27].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            
            # Optional fields (may not be present in all PDB files)
            try:
                occupancy = float(line[54:60].strip())
            except (ValueError, IndexError):
                occupancy = 1.0
                
            try:
                temp_factor = float(line[60:66].strip())
            except (ValueError, IndexError):
                temp_factor = 0.0
                
            try:
                element = line[76:78].strip()
            except IndexError:
                element = atom_name[0]  # Fallback to first character of atom name
            
            return {
                'record_type': record_type,
                'atom_serial': atom_serial,
                'atom_name': atom_name,
                'alt_loc': alt_loc,
                'residue_name': residue_name,
                'chain_id': chain_id,
                'residue_seq': residue_seq,
                'insertion_code': insertion_code,
                'x': x,
                'y': y,
                'z': z,
                'occupancy': occupancy,
                'temp_factor': temp_factor,
                'element': element
            }
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse line: {line.strip()}")
            print(f"Error: {e}")
            return None
    
    def filter_standard_residues(self) -> List[Dict]:
        """Filter atoms to only include standard amino acids."""
        return [atom for atom in self.atoms 
                if atom['residue_name'] in self.AMINO_ACIDS]
    
    def to_csv(self, output_file: str, include_hetatm: bool = False):
        """
        Write parsed atoms to CSV file.
        
        Args:
            output_file: Path to output CSV file
            include_hetatm: Whether to include HETATM records (non-protein atoms)
        """
        atoms_to_write = self.atoms
        
        if not include_hetatm:
            atoms_to_write = [atom for atom in atoms_to_write 
                            if atom['record_type'] == 'ATOM']
        
        fieldnames = [
            'atom_serial', 'atom_name', 'residue_name', 'chain_id', 
            'residue_seq', 'x', 'y', 'z', 'occupancy', 'temp_factor', 
            'element', 'alt_loc', 'insertion_code'
        ]
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for atom in atoms_to_write:
                # Only write fields in fieldnames
                row = {k: atom[k] for k in fieldnames if k in atom}
                writer.writerow(row)
        
        print(f"Converted {len(atoms_to_write)} atoms to {output_file}")
    
    def get_sequence(self) -> str:
        """
        Extract the amino acid sequence from the structure.
        
        Returns:
            String of one-letter amino acid codes
        """
        # Map three-letter to one-letter codes
        aa_map = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }
        
        # Get unique residues in sequence order
        seen = set()
        sequence_residues = []
        
        for atom in self.atoms:
            res_key = (atom['chain_id'], atom['residue_seq'], 
                      atom['insertion_code'], atom['residue_name'])
            if res_key not in seen and atom['residue_name'] in self.AMINO_ACIDS:
                seen.add(res_key)
                sequence_residues.append(atom['residue_name'])
        
        return ''.join(aa_map.get(res, 'X') for res in sequence_residues)


def main():
    """Main entry point for the script - hardcoded paths for Spyder."""
    
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
    input_pdb = config.PDB_DIR
    output_csv = config.CSV_DIR
    include_hetatm = False  # Set to True if you want non-protein atoms
    show_sequence = True    # Set to False to hide sequence
    
    print(f"Converting PDB file: {input_pdb}")
    print(f"Output CSV file: {output_csv}")
    print("-" * 60)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Parse PDB file
    pdb_parser = PDBParser(input_pdb)
    pdb_parser.parse()
    
    # Show sequence if requested
    if show_sequence:
        sequence = pdb_parser.get_sequence()
        print(f"\nAmino acid sequence ({len(sequence)} residues):")
        print(sequence)
        print()
    
    # Convert to CSV
    pdb_parser.to_csv(output_csv, include_hetatm=include_hetatm)
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"  Total atoms parsed: {len(pdb_parser.atoms)}")
    print(f"  Standard amino acid atoms: {len(pdb_parser.filter_standard_residues())}")
    
    # Count chains
    chains = set(atom['chain_id'] for atom in pdb_parser.atoms)
    print(f"  Number of chains: {len(chains)}")
    print(f"  Chain IDs: {', '.join(sorted(chains))}")


if __name__ == '__main__':
    main()
