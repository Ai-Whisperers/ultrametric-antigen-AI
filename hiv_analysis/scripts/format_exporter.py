#!/usr/bin/env python3
"""
Sequence Alignment Format Exporter
Exports HIV sequence alignments to various formats (FASTA, Clustal, Phylip, etc.)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
import textwrap
from datetime import datetime

@dataclass
class ExportConfig:
    """Configuration for alignment export"""
    output_formats: List[str] = None  # ['fasta', 'clustal', 'phylip', 'nexus', 'msf']
    line_length: int = 60
    include_conservation: bool = True
    include_consensus: bool = True
    add_headers: bool = True
    compress_output: bool = False
    
    def __post_init__(self):
        if self.output_formats is None:
            self.output_formats = ['fasta', 'clustal', 'phylip']

class SequenceAlignment:
    """Represents a multiple sequence alignment for export"""
    
    def __init__(self, sequences: Dict[str, str], conservation_scores: List[float] = None, 
                 consensus: str = None):
        self.sequences = sequences
        self.conservation_scores = conservation_scores
        self.consensus = consensus
        self.seq_names = list(sequences.keys())
        self.seq_length = len(next(iter(sequences.values()))) if sequences else 0
        
    @classmethod
    def from_fasta(cls, fasta_file: Union[str, Path]) -> 'SequenceAlignment':
        """Load alignment from FASTA file"""
        sequences = {}
        current_seq = None
        current_name = None
        
        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_name and current_seq:
                        sequences[current_name] = current_seq
                    current_name = line[1:].split()[0]
                    current_seq = ""
                elif line and not line.startswith('#'):
                    current_seq += line.upper()
                    
        if current_name and current_seq:
            sequences[current_name] = current_seq
            
        return cls(sequences)

class AlignmentExporter:
    """Export alignments to various formats"""
    
    def __init__(self, config: ExportConfig = None):
        self.config = config or ExportConfig()
        self.setup_logging()
        
    def setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def export_alignment(self, alignment: SequenceAlignment, 
                        output_prefix: Union[str, Path]) -> Dict[str, str]:
        """
        Export alignment to multiple formats
        
        Args:
            alignment: SequenceAlignment object
            output_prefix: Base path for output files
            
        Returns:
            Dict mapping format name to output file path
        """
        if not alignment.sequences:
            raise ValueError("Empty alignment provided")
            
        output_files = {}
        base_path = Path(output_prefix)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        for fmt in self.config.output_formats:
            self.logger.info(f"Exporting to {fmt.upper()} format...")
            
            if fmt == 'fasta':
                output_file = base_path.with_suffix('.fasta')
                self._export_fasta(alignment, output_file)
            elif fmt == 'clustal':
                output_file = base_path.with_suffix('.aln')
                self._export_clustal(alignment, output_file)
            elif fmt == 'phylip':
                output_file = base_path.with_suffix('.phy')
                self._export_phylip(alignment, output_file)
            elif fmt == 'nexus':
                output_file = base_path.with_suffix('.nex')
                self._export_nexus(alignment, output_file)
            elif fmt == 'msf':
                output_file = base_path.with_suffix('.msf')
                self._export_msf(alignment, output_file)
            else:
                self.logger.warning(f"Unknown format: {fmt}")
                continue
                
            output_files[fmt] = str(output_file)
            self.logger.info(f"Exported {fmt}: {output_file}")
            
        return output_files
        
    def _export_fasta(self, alignment: SequenceAlignment, output_file: Path) -> None:
        """Export to FASTA format"""
        with open(output_file, 'w') as f:
            if self.config.add_headers:
                f.write(f"# HIV Sequence Alignment (FASTA format)\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
                f.write(f"# Sequences: {len(alignment.sequences)}\n")
                f.write(f"# Length: {alignment.seq_length}\n#\n")
                
            for name in alignment.seq_names:
                sequence = alignment.sequences[name]
                f.write(f">{name}\n")
                
                # Write sequence in chunks
                for i in range(0, len(sequence), self.config.line_length):
                    chunk = sequence[i:i + self.config.line_length]
                    f.write(f"{chunk}\n")
                    
            # Add consensus if available
            if self.config.include_consensus and alignment.consensus:
                f.write(f">CONSENSUS\n")
                consensus = alignment.consensus
                for i in range(0, len(consensus), self.config.line_length):
                    chunk = consensus[i:i + self.config.line_length]
                    f.write(f"{chunk}\n")
                    
    def _export_clustal(self, alignment: SequenceAlignment, output_file: Path) -> None:
        """Export to ClustalW format"""
        with open(output_file, 'w') as f:
            # ClustalW header
            f.write("CLUSTAL W (HIV Sequence Alignment)\n\n")
            
            # Calculate maximum name length for formatting
            max_name_len = max(len(name) for name in alignment.seq_names)
            max_name_len = max(max_name_len, 15)  # Minimum width
            
            # Write alignment in blocks
            block_size = self.config.line_length
            
            for block_start in range(0, alignment.seq_length, block_size):
                block_end = min(block_start + block_size, alignment.seq_length)
                
                # Write sequences
                for name in alignment.seq_names:
                    sequence = alignment.sequences[name]
                    seq_chunk = sequence[block_start:block_end]
                    formatted_name = f"{name:<{max_name_len}}"
                    f.write(f"{formatted_name} {seq_chunk}\n")
                    
                # Add conservation line if available
                if (self.config.include_conservation and 
                    alignment.conservation_scores and 
                    block_start < len(alignment.conservation_scores)):
                    
                    cons_chunk = alignment.conservation_scores[block_start:block_end]
                    conservation_line = ''.join([
                        '*' if score > 0.9 else
                        ':' if score > 0.7 else
                        '.' if score > 0.5 else ' '
                        for score in cons_chunk
                    ])
                    spaces = ' ' * max_name_len
                    f.write(f"{spaces} {conservation_line}\n")
                    
                f.write("\n")
                
    def _export_phylip(self, alignment: SequenceAlignment, output_file: Path) -> None:
        """Export to PHYLIP format"""
        with open(output_file, 'w') as f:
            # PHYLIP header: number of sequences and sequence length
            f.write(f"  {len(alignment.sequences)}  {alignment.seq_length}\n")
            
            # Write sequences (PHYLIP has 10-character name limit)
            for i, name in enumerate(alignment.seq_names):
                sequence = alignment.sequences[name]
                
                # Truncate/pad name to 10 characters
                phylip_name = f"{name[:10]:<10}"
                
                # First line includes name
                first_chunk_size = min(self.config.line_length, alignment.seq_length)
                first_chunk = sequence[:first_chunk_size]
                f.write(f"{phylip_name} {first_chunk}\n")
                
                # Subsequent lines for this sequence (if needed)
                for pos in range(first_chunk_size, alignment.seq_length, self.config.line_length):
                    chunk = sequence[pos:pos + self.config.line_length]
                    f.write(f"           {chunk}\n")
                    
    def _export_nexus(self, alignment: SequenceAlignment, output_file: Path) -> None:
        """Export to NEXUS format"""
        with open(output_file, 'w') as f:
            f.write("#NEXUS\n\n")
            f.write("BEGIN DATA;\n")
            f.write(f"  DIMENSIONS NTAX={len(alignment.sequences)} NCHAR={alignment.seq_length};\n")
            f.write("  FORMAT DATATYPE=PROTEIN GAP=-;\n")
            f.write("  MATRIX\n")
            
            # Calculate max name length
            max_name_len = max(len(name) for name in alignment.seq_names)
            
            for name in alignment.seq_names:
                sequence = alignment.sequences[name]
                formatted_name = f"{name:<{max_name_len}}"
                f.write(f"    {formatted_name} {sequence}\n")
                
            f.write("  ;\n")
            f.write("END;\n")
            
    def _export_msf(self, alignment: SequenceAlignment, output_file: Path) -> None:
        """Export to MSF (Multiple Sequence Format)"""
        with open(output_file, 'w') as f:
            # MSF header
            f.write("HIV_sequence_alignment.msf  MSF: {} Type: P Check: 0000 ..\n\n".format(
                alignment.seq_length))
                
            # Sequence information
            for name in alignment.seq_names:
                sequence = alignment.sequences[name]
                # Simple checksum (not GCG compliant but functional)
                checksum = sum(ord(c) for c in sequence) % 10000
                f.write(f" Name: {name:<20} Len: {len(sequence):>6} Check: {checksum:>4} Weight: 1.00\n")
                
            f.write("\n//\n\n")
            
            # Alignment data in blocks
            block_size = 50
            for block_start in range(0, alignment.seq_length, block_size):
                block_end = min(block_start + block_size, alignment.seq_length)
                
                for name in alignment.seq_names:
                    sequence = alignment.sequences[name]
                    seq_chunk = sequence[block_start:block_end]
                    
                    # Format in groups of 10
                    formatted_chunk = ' '.join([
                        seq_chunk[i:i+10] for i in range(0, len(seq_chunk), 10)
                    ])
                    
                    f.write(f"{name:<20} {formatted_chunk}\n")
                    
                f.write("\n")
                
    def create_format_summary(self, output_files: Dict[str, str], 
                            output_dir: Path) -> str:
        """Create summary of exported formats"""
        summary_lines = [
            "# HIV Sequence Alignment Export Summary",
            f"Generated: {datetime.now().isoformat()}",
            f"Total formats: {len(output_files)}",
            "",
            "## Exported Files:",
            ""
        ]
        
        for fmt, filepath in output_files.items():
            file_path = Path(filepath)
            file_size = file_path.stat().st_size if file_path.exists() else 0
            
            summary_lines.extend([
                f"### {fmt.upper()} Format",
                f"- File: `{file_path.name}`",
                f"- Size: {file_size:,} bytes",
                f"- Description: {self._get_format_description(fmt)}",
                ""
            ])
            
        summary_lines.extend([
            "## Format Compatibility:",
            "",
            "- **FASTA**: Universal format, most bioinformatics tools",
            "- **Clustal**: ClustalW/X, MEGA, Jalview",
            "- **PHYLIP**: PHYLIP package, RAxML, PhyML",
            "- **NEXUS**: PAUP*, MrBayes, BEAST",
            "- **MSF**: GCG package, some legacy tools",
            ""
        ])
        
        summary_content = "\n".join(summary_lines)
        summary_file = output_dir / "export_summary.md"
        
        with open(summary_file, 'w') as f:
            f.write(summary_content)
            
        self.logger.info(f"Export summary created: {summary_file}")
        return summary_content
        
    def _get_format_description(self, fmt: str) -> str:
        """Get description for each format"""
        descriptions = {
            'fasta': "Simple text format with sequence headers and data",
            'clustal': "ClustalW format with conservation annotation",
            'phylip': "Phylogenetic analysis format (sequential)",
            'nexus': "Flexible format for phylogenetic data",
            'msf': "Multiple Sequence Format with checksums"
        }
        return descriptions.get(fmt, "Unknown format")

def main():
    """Main entry point for format export"""
    base_dir = Path(__file__).parent.parent
    
    print("🔄 HIV Sequence Alignment Format Exporter")
    print("=" * 50)
    
    # Check for aligned sequences
    aligned_file = base_dir / "data/processed/aligned/hiv1_env_sample_aligned.fasta"
    if not aligned_file.exists():
        print("❌ No aligned sequences found!")
        print("💡 Run mafft_wrapper.py first to create alignments")
        return 1
        
    # Load conservation scores if available
    conservation_file = base_dir / "results/conservation/hiv_env_conservation.json"
    conservation_scores = None
    
    if conservation_file.exists():
        import json
        with open(conservation_file, 'r') as f:
            conservation_data = json.load(f)
            if 'scores' in conservation_data and 'shannon' in conservation_data['scores']:
                conservation_scores = conservation_data['scores']['shannon']
        print("✅ Conservation scores loaded")
    else:
        print("⚠️  No conservation scores found (run conservation_scorer.py)")
        
    # Configure exporter
    config = ExportConfig(
        output_formats=['fasta', 'clustal', 'phylip', 'nexus', 'msf'],
        line_length=60,
        include_conservation=True,
        include_consensus=True,
        add_headers=True
    )
    
    exporter = AlignmentExporter(config)
    
    try:
        # Load alignment
        print(f"📖 Loading alignment: {aligned_file}")
        alignment = SequenceAlignment.from_fasta(aligned_file)
        alignment.conservation_scores = conservation_scores
        print(f"✅ Loaded {len(alignment.sequences)} sequences")
        
        # Generate consensus if not provided
        if alignment.consensus is None and config.include_consensus:
            # Simple consensus generation
            consensus = []
            for pos in range(alignment.seq_length):
                residues = []
                for seq in alignment.sequences.values():
                    if pos < len(seq) and seq[pos] != '-':
                        residues.append(seq[pos])
                        
                if residues:
                    from collections import Counter
                    most_common = Counter(residues).most_common(1)[0][0]
                    consensus.append(most_common)
                else:
                    consensus.append('-')
            alignment.consensus = ''.join(consensus)
            
        # Create output directory
        output_dir = base_dir / "results/exports"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export to all formats
        output_prefix = output_dir / "hiv_env_alignment"
        output_files = exporter.export_alignment(alignment, output_prefix)
        
        # Create summary
        summary = exporter.create_format_summary(output_files, output_dir)
        
        print("✅ Format export completed!")
        print(f"📁 Output directory: {output_dir}")
        print(f"📄 Formats exported: {', '.join(output_files.keys())}")
        
        # Show file sizes
        print("\n📊 Export Summary:")
        for fmt, filepath in output_files.items():
            file_path = Path(filepath)
            size = file_path.stat().st_size if file_path.exists() else 0
            print(f"  {fmt.upper()}: {file_path.name} ({size:,} bytes)")
            
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())