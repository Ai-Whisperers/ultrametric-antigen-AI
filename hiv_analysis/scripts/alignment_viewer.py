#!/usr/bin/env python3
"""
Multiple Sequence Alignment Viewer
Visualizes HIV sequence alignments with conservation coloring and analysis
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from collections import Counter
import re

@dataclass
class ViewerConfig:
    """Configuration for alignment viewer"""
    line_width: int = 80
    show_ruler: bool = True
    show_consensus: bool = True
    show_conservation: bool = True
    color_scheme: str = "conservation"  # conservation, amino_acid, hydrophobicity
    output_format: str = "html"  # html, text, both
    
class SequenceAlignment:
    """Represents a multiple sequence alignment"""
    
    def __init__(self, sequences: Dict[str, str]):
        self.sequences = sequences
        self.seq_names = list(sequences.keys())
        self.seq_length = len(next(iter(sequences.values()))) if sequences else 0
        self.conservation_scores = None
        self.consensus_sequence = None
        
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
                    # Clean header - remove description after first space/tab
                    current_name = line[1:].split()[0] if line[1:].split() else line[1:]
                    current_seq = ""
                elif line and not line.startswith('#'):
                    current_seq += line.upper()
                    
        # Add last sequence
        if current_name and current_seq:
            sequences[current_name] = current_seq
            
        return cls(sequences)
        
    def calculate_conservation(self) -> List[float]:
        """Calculate conservation scores for each position"""
        if not self.sequences:
            return []
            
        conservation_scores = []
        
        for pos in range(self.seq_length):
            # Get all amino acids at this position
            residues = []
            for seq in self.sequences.values():
                if pos < len(seq) and seq[pos] != '-':
                    residues.append(seq[pos])
                    
            if not residues:
                conservation_scores.append(0.0)
                continue
                
            # Calculate conservation as 1 - (diversity / max_diversity)
            residue_counts = Counter(residues)
            total_residues = len(residues)
            
            # Shannon entropy approach
            import math
            entropy = 0.0
            for count in residue_counts.values():
                if count > 0:
                    prob = count / total_residues
                    entropy -= prob * math.log2(prob) if prob > 0 else 0
                    
            # Normalize by maximum possible entropy (log2 of 20 amino acids)
            max_entropy = 4.32  # log2(20)
            conservation = 1.0 - (entropy / max_entropy)
            conservation_scores.append(max(0.0, conservation))
            
        self.conservation_scores = conservation_scores
        return conservation_scores
        
    def generate_consensus(self, threshold: float = 0.5) -> str:
        """Generate consensus sequence"""
        consensus = []
        
        for pos in range(self.seq_length):
            residues = []
            for seq in self.sequences.values():
                if pos < len(seq):
                    residues.append(seq[pos])
                    
            if not residues:
                consensus.append('-')
                continue
                
            # Count residues (excluding gaps)
            non_gap_residues = [r for r in residues if r != '-']
            if not non_gap_residues:
                consensus.append('-')
                continue
                
            residue_counts = Counter(non_gap_residues)
            most_common = residue_counts.most_common(1)[0]
            
            # Use most common if it meets threshold
            if most_common[1] / len(non_gap_residues) >= threshold:
                consensus.append(most_common[0])
            else:
                consensus.append('X')  # Ambiguous position
                
        self.consensus_sequence = ''.join(consensus)
        return self.consensus_sequence

class AlignmentViewer:
    """Visualizes multiple sequence alignments"""
    
    def __init__(self, config: ViewerConfig = None):
        self.config = config or ViewerConfig()
        self.setup_logging()
        
    def setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def view_alignment(self, 
                      alignment: SequenceAlignment,
                      output_file: Union[str, Path] = None,
                      start_pos: int = 0,
                      end_pos: int = None) -> str:
        """
        Generate alignment visualization
        
        Args:
            alignment: SequenceAlignment object
            output_file: Output file path
            start_pos: Start position for viewing
            end_pos: End position for viewing
            
        Returns:
            String representation of the alignment
        """
        if not alignment.sequences:
            raise ValueError("Empty alignment provided")
            
        # Calculate conservation if not already done
        if alignment.conservation_scores is None:
            alignment.calculate_conservation()
            
        # Generate consensus if not already done  
        if alignment.consensus_sequence is None:
            alignment.generate_consensus()
            
        # Set viewing range
        end_pos = end_pos or alignment.seq_length
        start_pos = max(0, start_pos)
        end_pos = min(alignment.seq_length, end_pos)
        
        if self.config.output_format in ["html", "both"]:
            html_output = self._generate_html_view(alignment, start_pos, end_pos)
            if output_file and self.config.output_format == "html":
                html_file = Path(output_file).with_suffix('.html')
                with open(html_file, 'w') as f:
                    f.write(html_output)
                self.logger.info(f"HTML alignment saved to: {html_file}")
                    
        if self.config.output_format in ["text", "both"]:
            text_output = self._generate_text_view(alignment, start_pos, end_pos)
            if output_file and self.config.output_format == "text":
                text_file = Path(output_file).with_suffix('.txt')
                with open(text_file, 'w') as f:
                    f.write(text_output)
                self.logger.info(f"Text alignment saved to: {text_file}")
                    
        # Return appropriate format
        if self.config.output_format == "html":
            return html_output
        elif self.config.output_format == "text":
            return text_output
        else:  # both
            if output_file:
                base_path = Path(output_file)
                html_file = base_path.with_suffix('.html')
                text_file = base_path.with_suffix('.txt')
                
                with open(html_file, 'w') as f:
                    f.write(html_output)
                with open(text_file, 'w') as f:
                    f.write(text_output)
                    
                self.logger.info(f"Alignment saved to: {html_file} and {text_file}")
                
            return text_output
            
    def _generate_text_view(self, alignment: SequenceAlignment, 
                           start_pos: int, end_pos: int) -> str:
        """Generate text-based alignment view"""
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append("HIV SEQUENCE ALIGNMENT VIEWER")
        lines.append("=" * 80)
        lines.append(f"Sequences: {len(alignment.sequences)}")
        lines.append(f"Length: {alignment.seq_length}")
        lines.append(f"Viewing: {start_pos}-{end_pos}")
        lines.append("")
        
        # Process in chunks
        chunk_size = self.config.line_width
        
        for chunk_start in range(start_pos, end_pos, chunk_size):
            chunk_end = min(chunk_start + chunk_size, end_pos)
            
            lines.append(f"Position {chunk_start:>6} - {chunk_end-1}")
            
            # Ruler
            if self.config.show_ruler:
                ruler = self._create_ruler(chunk_start, chunk_end)
                lines.append(" " * 15 + ruler)
                lines.append("")
                
            # Sequences
            max_name_len = max(len(name) for name in alignment.seq_names)
            
            for name in alignment.seq_names:
                seq_chunk = alignment.sequences[name][chunk_start:chunk_end]
                formatted_name = f"{name:<{max_name_len}}"
                
                # Add conservation markers below sequence
                if self.config.show_conservation and alignment.conservation_scores:
                    cons_chunk = alignment.conservation_scores[chunk_start:chunk_end]
                    cons_markers = ''.join(['*' if score > 0.8 else 
                                          ':' if score > 0.6 else
                                          '.' if score > 0.4 else ' '
                                          for score in cons_chunk])
                    lines.append(f"{formatted_name} {seq_chunk}")
                else:
                    lines.append(f"{formatted_name} {seq_chunk}")
                    
            # Consensus sequence
            if self.config.show_consensus and alignment.consensus_sequence:
                consensus_chunk = alignment.consensus_sequence[chunk_start:chunk_end]
                lines.append(" " * max_name_len + f" {consensus_chunk}")
                
            # Conservation line
            if self.config.show_conservation and alignment.conservation_scores:
                cons_chunk = alignment.conservation_scores[chunk_start:chunk_end]
                cons_line = ''.join(['*' if score > 0.8 else 
                                   ':' if score > 0.6 else
                                   '.' if score > 0.4 else ' '
                                   for score in cons_chunk])
                lines.append(" " * max_name_len + f" {cons_line}")
                
            lines.append("")
            
        # Legend
        lines.extend([
            "CONSERVATION LEGEND:",
            "* = Highly conserved (>80%)",
            ": = Moderately conserved (>60%)",
            ". = Weakly conserved (>40%)",
            "  = Variable (<40%)",
            ""
        ])
        
        return "\n".join(lines)
        
    def _generate_html_view(self, alignment: SequenceAlignment,
                           start_pos: int, end_pos: int) -> str:
        """Generate HTML-based alignment view"""
        
        # CSS for styling
        css = """
        <style>
        body { font-family: 'Courier New', monospace; margin: 20px; }
        .alignment-container { background-color: #f9f9f9; padding: 20px; border-radius: 5px; }
        .alignment-header { background-color: #4CAF50; color: white; padding: 10px; margin-bottom: 20px; }
        .sequence-row { margin: 2px 0; }
        .sequence-name { display: inline-block; width: 150px; font-weight: bold; }
        .sequence-data { font-family: monospace; }
        .ruler { color: #666; margin-bottom: 10px; }
        .consensus { background-color: #e8f5e8; }
        .conservation { color: #666; font-size: 0.8em; }
        .highly-conserved { background-color: #4CAF50; color: white; }
        .moderately-conserved { background-color: #8BC34A; }
        .weakly-conserved { background-color: #CDDC39; }
        .variable { background-color: #FFF59D; }
        .gap { color: #ccc; }
        </style>
        """
        
        html_lines = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>HIV Sequence Alignment</title>",
            css,
            "</head>",
            "<body>",
            "<div class='alignment-container'>",
            "<div class='alignment-header'>",
            "<h2>HIV Sequence Alignment Viewer</h2>",
            f"<p>Sequences: {len(alignment.sequences)} | Length: {alignment.seq_length} | Viewing: {start_pos}-{end_pos}</p>",
            "</div>"
        ]
        
        # Process in chunks
        chunk_size = self.config.line_width
        
        for chunk_start in range(start_pos, end_pos, chunk_size):
            chunk_end = min(chunk_start + chunk_size, end_pos)
            
            html_lines.append(f"<div class='chunk'>")
            html_lines.append(f"<h4>Position {chunk_start} - {chunk_end-1}</h4>")
            
            # Ruler
            if self.config.show_ruler:
                ruler = self._create_ruler(chunk_start, chunk_end)
                html_lines.append(f"<div class='ruler'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{ruler}</div>")
                
            # Sequences
            for name in alignment.seq_names:
                seq_chunk = alignment.sequences[name][chunk_start:chunk_end]
                colored_seq = self._color_sequence_html(seq_chunk, alignment.conservation_scores[chunk_start:chunk_end] if alignment.conservation_scores else None)
                html_lines.append(f"<div class='sequence-row'><span class='sequence-name'>{name}</span><span class='sequence-data'>{colored_seq}</span></div>")
                
            # Consensus
            if self.config.show_consensus and alignment.consensus_sequence:
                consensus_chunk = alignment.consensus_sequence[chunk_start:chunk_end]
                html_lines.append(f"<div class='sequence-row consensus'><span class='sequence-name'>CONSENSUS</span><span class='sequence-data'>{consensus_chunk}</span></div>")
                
            html_lines.append("</div><br>")
            
        # Legend
        html_lines.extend([
            "<div class='legend'>",
            "<h3>Conservation Legend:</h3>",
            "<p><span class='highly-conserved'>&nbsp;&nbsp;</span> Highly conserved (&gt;80%)</p>",
            "<p><span class='moderately-conserved'>&nbsp;&nbsp;</span> Moderately conserved (&gt;60%)</p>", 
            "<p><span class='weakly-conserved'>&nbsp;&nbsp;</span> Weakly conserved (&gt;40%)</p>",
            "<p><span class='variable'>&nbsp;&nbsp;</span> Variable (&lt;40%)</p>",
            "</div>",
            "</div>",
            "</body>",
            "</html>"
        ])
        
        return "\n".join(html_lines)
        
    def _create_ruler(self, start: int, end: int) -> str:
        """Create position ruler"""
        ruler = []
        for i in range(start, end):
            if i % 10 == 0:
                ruler.append('|')
            elif i % 5 == 0:
                ruler.append('+')
            else:
                ruler.append('-')
        return ''.join(ruler)
        
    def _color_sequence_html(self, sequence: str, conservation_scores: List[float] = None) -> str:
        """Color sequence based on conservation"""
        if not conservation_scores:
            return sequence.replace('-', '<span class="gap">-</span>')
            
        colored = []
        for i, (residue, score) in enumerate(zip(sequence, conservation_scores)):
            if residue == '-':
                colored.append('<span class="gap">-</span>')
            elif score > 0.8:
                colored.append(f'<span class="highly-conserved">{residue}</span>')
            elif score > 0.6:
                colored.append(f'<span class="moderately-conserved">{residue}</span>')
            elif score > 0.4:
                colored.append(f'<span class="weakly-conserved">{residue}</span>')
            else:
                colored.append(f'<span class="variable">{residue}</span>')
                
        return ''.join(colored)
        
    def create_demo_viewer(self, base_dir: Path) -> str:
        """Create demonstration alignment view"""
        self.logger.info("Creating demo alignment visualization...")
        
        # Look for aligned sequences
        aligned_file = base_dir / "data/processed/aligned/hiv1_env_sample_aligned.fasta"
        if not aligned_file.exists():
            return "Error: No aligned sequences found. Run mafft_wrapper.py first."
            
        # Load alignment
        try:
            alignment = SequenceAlignment.from_fasta(aligned_file)
            self.logger.info(f"Loaded alignment with {len(alignment.sequences)} sequences")
            
            # Generate views
            output_base = base_dir / "results/alignment_view_demo"
            output_base.parent.mkdir(parents=True, exist_ok=True)
            
            # Configure for both outputs
            self.config.output_format = "both"
            
            result = self.view_alignment(alignment, output_base)
            
            return f"Demo alignment view created: {output_base}.html and {output_base}.txt"
            
        except Exception as e:
            return f"Error creating alignment view: {e}"

def main():
    """Main entry point"""
    base_dir = Path(__file__).parent.parent
    
    print("🔬 HIV Sequence Alignment Viewer")
    print("=" * 40)
    
    # Initialize viewer
    config = ViewerConfig(
        line_width=60,
        show_conservation=True,
        show_consensus=True,
        output_format="both"
    )
    
    viewer = AlignmentViewer(config)
    
    # Create demo visualization
    result = viewer.create_demo_viewer(base_dir)
    print(result)
    
    print("\n📊 Features:")
    print("- Conservation scoring and coloring")
    print("- Consensus sequence generation") 
    print("- HTML and text output formats")
    print("- Configurable viewing parameters")
    print("- Ruler and position markers")
    
    print(f"\n📁 Output directory: {base_dir}/results/")
    print("🚀 Alignment viewer ready for production use!")
    
    return 0

if __name__ == "__main__":
    exit(main())