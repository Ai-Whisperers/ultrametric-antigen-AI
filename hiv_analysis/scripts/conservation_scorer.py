#!/usr/bin/env python3
"""
Conservation Scoring for HIV Sequences
Calculates various conservation metrics for multiple sequence alignments
"""

import math
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import Counter
import logging

@dataclass
class ConservationConfig:
    """Configuration for conservation scoring"""
    score_types: List[str] = None  # ['shannon', 'simpson', 'property', 'blosum']
    gap_penalty: float = 0.1
    pseudocount: float = 0.001
    property_groups: Dict[str, List[str]] = None
    output_format: str = "json"  # json, csv, tsv
    
    def __post_init__(self):
        if self.score_types is None:
            self.score_types = ['shannon', 'simpson', 'property']
        
        if self.property_groups is None:
            # Amino acid property groups for property-based conservation
            self.property_groups = {
                'hydrophobic': ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'Y'],
                'polar': ['S', 'T', 'N', 'Q'],
                'charged_positive': ['R', 'H', 'K'],
                'charged_negative': ['D', 'E'],
                'special': ['C', 'P', 'G']
            }

class ConservationScorer:
    """Calculate conservation scores for sequence alignments"""
    
    def __init__(self, config: ConservationConfig = None):
        self.config = config or ConservationConfig()
        self.setup_logging()
        
        # Amino acid frequencies in proteins (for background correction)
        self.aa_frequencies = {
            'A': 0.08, 'R': 0.055, 'N': 0.041, 'D': 0.054, 'C': 0.014,
            'Q': 0.039, 'E': 0.067, 'G': 0.071, 'H': 0.022, 'I': 0.059,
            'L': 0.096, 'K': 0.058, 'M': 0.024, 'F': 0.038, 'P': 0.047,
            'S': 0.066, 'T': 0.053, 'W': 0.01, 'Y': 0.029, 'V': 0.068
        }
        
    def setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_alignment(self, fasta_file: Union[str, Path]) -> Dict[str, str]:
        """Load multiple sequence alignment from FASTA file"""
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
            
        return sequences
        
    def calculate_all_scores(self, alignment: Dict[str, str]) -> Dict:
        """Calculate all configured conservation scores"""
        if not alignment:
            raise ValueError("Empty alignment provided")
            
        seq_length = len(next(iter(alignment.values())))
        results = {
            'metadata': {
                'sequences': len(alignment),
                'length': seq_length,
                'score_types': self.config.score_types
            },
            'scores': {}
        }
        
        # Calculate each requested score type
        for score_type in self.config.score_types:
            self.logger.info(f"Calculating {score_type} conservation scores...")
            
            if score_type == 'shannon':
                results['scores'][score_type] = self.calculate_shannon_entropy(alignment)
            elif score_type == 'simpson':
                results['scores'][score_type] = self.calculate_simpson_index(alignment)
            elif score_type == 'property':
                results['scores'][score_type] = self.calculate_property_conservation(alignment)
            elif score_type == 'blosum':
                results['scores'][score_type] = self.calculate_blosum_conservation(alignment)
            else:
                self.logger.warning(f"Unknown score type: {score_type}")
                
        return results
        
    def calculate_shannon_entropy(self, alignment: Dict[str, str]) -> List[float]:
        """Calculate Shannon entropy-based conservation scores"""
        seq_length = len(next(iter(alignment.values())))
        entropy_scores = []
        
        for pos in range(seq_length):
            # Get amino acids at this position
            residues = []
            for seq in alignment.values():
                if pos < len(seq):
                    residue = seq[pos]
                    if residue != '-':  # Skip gaps
                        residues.append(residue)
                        
            if not residues:
                entropy_scores.append(0.0)
                continue
                
            # Calculate Shannon entropy
            residue_counts = Counter(residues)
            total = len(residues)
            entropy = 0.0
            
            for count in residue_counts.values():
                if count > 0:
                    prob = count / total
                    entropy -= prob * math.log2(prob + self.config.pseudocount)
                    
            # Normalize by maximum possible entropy (log2(20) for 20 amino acids)
            max_entropy = math.log2(20)
            conservation = 1.0 - (entropy / max_entropy)
            conservation = max(0.0, min(1.0, conservation))
            
            entropy_scores.append(conservation)
            
        return entropy_scores
        
    def calculate_simpson_index(self, alignment: Dict[str, str]) -> List[float]:
        """Calculate Simpson's diversity index for conservation"""
        seq_length = len(next(iter(alignment.values())))
        simpson_scores = []
        
        for pos in range(seq_length):
            residues = []
            for seq in alignment.values():
                if pos < len(seq):
                    residue = seq[pos]
                    if residue != '-':
                        residues.append(residue)
                        
            if not residues:
                simpson_scores.append(0.0)
                continue
                
            # Calculate Simpson's index
            residue_counts = Counter(residues)
            total = len(residues)
            simpson_sum = 0.0
            
            for count in residue_counts.values():
                simpson_sum += (count * (count - 1)) / (total * (total - 1))
                
            # Simpson's index ranges from 0 (max diversity) to 1 (no diversity)
            # Convert to conservation score (1 = conserved, 0 = diverse)
            simpson_scores.append(simpson_sum)
            
        return simpson_scores
        
    def calculate_property_conservation(self, alignment: Dict[str, str]) -> List[float]:
        """Calculate conservation based on amino acid property groups"""
        seq_length = len(next(iter(alignment.values())))
        property_scores = []
        
        # Create residue to property mapping
        residue_to_property = {}
        for prop, residues in self.config.property_groups.items():
            for residue in residues:
                residue_to_property[residue] = prop
                
        for pos in range(seq_length):
            residues = []
            for seq in alignment.values():
                if pos < len(seq):
                    residue = seq[pos]
                    if residue != '-':
                        residues.append(residue)
                        
            if not residues:
                property_scores.append(0.0)
                continue
                
            # Map to properties
            properties = []
            for residue in residues:
                prop = residue_to_property.get(residue, 'unknown')
                properties.append(prop)
                
            # Calculate property conservation
            prop_counts = Counter(properties)
            total = len(properties)
            
            if total == 0:
                property_scores.append(0.0)
                continue
                
            # Most frequent property fraction
            most_common_count = prop_counts.most_common(1)[0][1]
            conservation = most_common_count / total
            property_scores.append(conservation)
            
        return property_scores
        
    def calculate_blosum_conservation(self, alignment: Dict[str, str]) -> List[float]:
        """Calculate conservation based on BLOSUM62 substitution scores"""
        # Simplified BLOSUM62 scores for common substitutions
        blosum_similar = {
            ('A', 'S'): True, ('A', 'T'): True,
            ('V', 'I'): True, ('V', 'L'): True, ('I', 'L'): True,
            ('F', 'Y'): True, ('F', 'W'): True, ('Y', 'W'): True,
            ('K', 'R'): True, ('D', 'E'): True,
            ('N', 'Q'): True, ('S', 'T'): True
        }
        
        seq_length = len(next(iter(alignment.values())))
        blosum_scores = []
        
        for pos in range(seq_length):
            residues = []
            for seq in alignment.values():
                if pos < len(seq):
                    residue = seq[pos]
                    if residue != '-':
                        residues.append(residue)
                        
            if not residues:
                blosum_scores.append(0.0)
                continue
                
            # Calculate similarity score
            similar_pairs = 0
            total_pairs = 0
            
            for i in range(len(residues)):
                for j in range(i + 1, len(residues)):
                    aa1, aa2 = residues[i], residues[j]
                    total_pairs += 1
                    
                    if aa1 == aa2:
                        similar_pairs += 1
                    elif (aa1, aa2) in blosum_similar or (aa2, aa1) in blosum_similar:
                        similar_pairs += 0.5  # Partial credit for similar amino acids
                        
            if total_pairs == 0:
                conservation = 1.0
            else:
                conservation = similar_pairs / total_pairs
                
            blosum_scores.append(conservation)
            
        return blosum_scores
        
    def generate_conservation_report(self, results: Dict, output_file: Union[str, Path] = None) -> str:
        """Generate conservation analysis report"""
        lines = []
        
        # Header
        lines.extend([
            "=" * 70,
            "HIV SEQUENCE CONSERVATION ANALYSIS REPORT",
            "=" * 70,
            f"Sequences analyzed: {results['metadata']['sequences']}",
            f"Alignment length: {results['metadata']['length']}",
            f"Score types: {', '.join(results['metadata']['score_types'])}",
            ""
        ])
        
        # Summary statistics for each score type
        for score_type, scores in results['scores'].items():
            if not scores:
                continue
                
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            min_score = np.min(scores)
            max_score = np.max(scores)
            
            # Highly conserved positions (>0.8)
            highly_conserved = sum(1 for s in scores if s > 0.8)
            moderately_conserved = sum(1 for s in scores if 0.6 < s <= 0.8)
            variable = sum(1 for s in scores if s <= 0.6)
            
            lines.extend([
                f"{score_type.upper()} CONSERVATION:",
                f"  Mean: {mean_score:.3f} ± {std_score:.3f}",
                f"  Range: {min_score:.3f} - {max_score:.3f}",
                f"  Highly conserved positions (>0.8): {highly_conserved} ({highly_conserved/len(scores)*100:.1f}%)",
                f"  Moderately conserved (0.6-0.8): {moderately_conserved} ({moderately_conserved/len(scores)*100:.1f}%)",
                f"  Variable positions (≤0.6): {variable} ({variable/len(scores)*100:.1f}%)",
                ""
            ])
            
        # Top conserved regions
        lines.append("TOP 10 MOST CONSERVED REGIONS:")
        if 'shannon' in results['scores']:
            shannon_scores = results['scores']['shannon']
            # Find windows of high conservation
            window_size = 10
            best_windows = []
            
            for i in range(len(shannon_scores) - window_size + 1):
                window_scores = shannon_scores[i:i + window_size]
                avg_score = np.mean(window_scores)
                best_windows.append((i, i + window_size, avg_score))
                
            # Sort by conservation score
            best_windows.sort(key=lambda x: x[2], reverse=True)
            
            for i, (start, end, score) in enumerate(best_windows[:10]):
                lines.append(f"  {i+1:2d}. Position {start:3d}-{end:3d}: {score:.3f}")
                
        lines.append("")
        
        report = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            self.logger.info(f"Conservation report saved to: {output_file}")
            
        return report
        
    def export_scores(self, results: Dict, output_file: Union[str, Path]) -> None:
        """Export conservation scores to file"""
        output_path = Path(output_file)
        
        if self.config.output_format == "json":
            with open(output_path.with_suffix('.json'), 'w') as f:
                json.dump(results, f, indent=2)
                
        elif self.config.output_format == "csv":
            import csv
            with open(output_path.with_suffix('.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                header = ['Position'] + list(results['scores'].keys())
                writer.writerow(header)
                
                # Data rows
                seq_length = results['metadata']['length']
                for pos in range(seq_length):
                    row = [pos + 1]  # 1-based position
                    for score_type in results['scores']:
                        scores = results['scores'][score_type]
                        row.append(scores[pos] if pos < len(scores) else 'NA')
                    writer.writerow(row)
                    
        self.logger.info(f"Conservation scores exported to: {output_path}")

def main():
    """Main entry point for conservation scoring"""
    base_dir = Path(__file__).parent.parent
    
    print("🧬 HIV Sequence Conservation Scoring")
    print("=" * 45)
    
    # Check for aligned sequences
    aligned_file = base_dir / "data/processed/aligned/hiv1_env_sample_aligned.fasta"
    if not aligned_file.exists():
        print("❌ No aligned sequences found!")
        print("💡 Run mafft_wrapper.py first to create alignments")
        return 1
        
    # Configure conservation scoring
    config = ConservationConfig(
        score_types=['shannon', 'simpson', 'property'],
        output_format='json'
    )
    
    scorer = ConservationScorer(config)
    
    try:
        # Load alignment
        print(f"📖 Loading alignment: {aligned_file}")
        alignment = scorer.load_alignment(aligned_file)
        print(f"✅ Loaded {len(alignment)} sequences")
        
        # Calculate conservation scores
        print("🔍 Calculating conservation scores...")
        results = scorer.calculate_all_scores(alignment)
        
        # Create output directory
        output_dir = base_dir / "results/conservation"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export scores
        output_file = output_dir / "hiv_env_conservation"
        scorer.export_scores(results, output_file)
        
        # Generate report
        report_file = output_dir / "conservation_report.txt"
        report = scorer.generate_conservation_report(results, report_file)
        
        print("✅ Conservation analysis completed!")
        print(f"📊 Results: {output_file}.json")
        print(f"📋 Report: {report_file}")
        
        # Show summary
        print("\n📈 CONSERVATION SUMMARY:")
        if 'shannon' in results['scores']:
            shannon_scores = results['scores']['shannon']
            mean_conservation = np.mean(shannon_scores)
            highly_conserved = sum(1 for s in shannon_scores if s > 0.8)
            total_positions = len(shannon_scores)
            
            print(f"  Mean conservation: {mean_conservation:.3f}")
            print(f"  Highly conserved: {highly_conserved}/{total_positions} ({highly_conserved/total_positions*100:.1f}%)")
            
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())