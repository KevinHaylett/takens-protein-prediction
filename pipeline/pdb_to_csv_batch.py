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


def convert_single_pdb(input_pdb: Path, output_csv: Path, 
                       include_hetatm: bool = False, 
                       show_sequence: bool = False) -> bool:
    """
    Convert a single PDB file to CSV.
    
    Args:
        input_pdb: Path to input PDB file
        output_csv: Path to output CSV file
        include_hetatm: Include non-protein atoms
        show_sequence: Print amino acid sequence
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Parse PDB file
        pdb_parser = PDBParser(input_pdb)
        pdb_parser.parse()
        
        if len(pdb_parser.atoms) == 0:
            print(f"  ⚠ Warning: No atoms found in {input_pdb.name}")
            return False
        
        # Show sequence if requested
        if show_sequence:
            sequence = pdb_parser.get_sequence()
            print(f"  Sequence ({len(sequence)} residues): {sequence[:50]}{'...' if len(sequence) > 50 else ''}")
        
        # Convert to CSV
        pdb_parser.to_csv(output_csv, include_hetatm=include_hetatm)
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error converting {input_pdb.name}: {e}")
        return False


def batch_convert_pdb_to_csv(input_dir: str, output_dir: str,
                              include_hetatm: bool = False,
                              show_sequence: bool = False):
    """
    Batch convert all PDB files in a directory to CSV format.
    
    Args:
        input_dir: Directory containing PDB files
        output_dir: Directory to save CSV files
        include_hetatm: Include non-protein atoms
        show_sequence: Print amino acid sequence for each file
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all PDB files
    pdb_files = sorted(input_path.glob('*.pdb'))
    
    if not pdb_files:
        print(f"No PDB files found in {input_path}")
        return
    
    print("=" * 70)
    print(f"Batch PDB to CSV Converter")
    print(f"Input directory:  {input_path}")
    print(f"Output directory: {output_path}")
    print(f"Files found:      {len(pdb_files)}")
    print("=" * 70)
    
    success_count = 0
    failed_count = 0
    
    for i, pdb_file in enumerate(pdb_files, 1):
        # Create output filename (same name, .csv extension)
        csv_file = output_path / f"{pdb_file.stem}.csv"
        
        print(f"\n[{i}/{len(pdb_files)}] {pdb_file.name} → {csv_file.name}")
        
        if convert_single_pdb(pdb_file, csv_file, include_hetatm, show_sequence):
            success_count += 1
        else:
            failed_count += 1
    
    # Summary
    print("\n" + "=" * 70)
    print(f"Conversion Complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {failed_count}")
    print(f"  Total:   {len(pdb_files)}")
    print(f"  Output:  {output_path.absolute()}")
    print("=" * 70)


def main():
    """Main entry point for the script - batch convert all PDB files."""
    
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
    input_dir = config.PDB_DIR
    output_dir = config.CSV_DIR
    include_hetatm = False  # Set to True if you want non-protein atoms
    show_sequence = False   # Set to True to see sequences (can be verbose)
    
    # Run batch conversion
    batch_convert_pdb_to_csv(
        input_dir=input_dir,
        output_dir=output_dir,
        include_hetatm=include_hetatm,
        show_sequence=show_sequence
    )


if __name__ == '__main__':
    main()
