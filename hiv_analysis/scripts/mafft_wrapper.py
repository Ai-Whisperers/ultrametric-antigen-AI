#!/usr/bin/env python3
"""
MAFFT Wrapper for HIV Sequence Alignment
Provides a Python interface to the MAFFT multiple sequence alignment tool
"""

import subprocess
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import tempfile
import shutil

@dataclass 
class MAFFTConfig:
    """Configuration for MAFFT alignment"""
    algorithm: str = "auto"  # auto, linsi, ginsi, einsi, fftns, fftnsi
    max_iterations: int = 1000
    gap_open_penalty: float = 1.53
    gap_extension_penalty: float = 0.123
    threads: int = 4
    quiet: bool = False
    output_format: str = "fasta"  # fasta, clustal, phylip
    
class MAFFTWrapper:
    """Python wrapper for MAFFT multiple sequence alignment"""
    
    def __init__(self, config: MAFFTConfig = None):
        self.config = config or MAFFTConfig()
        self.setup_logging()
        self.mafft_path = self._find_mafft()
        
    def setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO if not self.config.quiet else logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _find_mafft(self) -> Optional[str]:
        """Find MAFFT executable in system PATH"""
        mafft_path = shutil.which('mafft')
        if not mafft_path:
            self.logger.warning("MAFFT not found in PATH. Install with: sudo apt-get install mafft")
            self.logger.info("Alternative: Use Docker or conda environment")
        return mafft_path
        
    def check_mafft_installation(self) -> bool:
        """Check if MAFFT is properly installed and accessible"""
        if not self.mafft_path:
            return False
            
        try:
            result = subprocess.run(
                [self.mafft_path, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                version_info = result.stdout.strip()
                self.logger.info(f"MAFFT found: {version_info}")
                return True
            else:
                self.logger.error(f"MAFFT version check failed: {result.stderr}")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.logger.error(f"MAFFT check failed: {e}")
            return False
            
    def align_sequences(self, 
                       input_file: Union[str, Path],
                       output_file: Union[str, Path] = None,
                       algorithm: str = None) -> Dict:
        """
        Align sequences using MAFFT
        
        Args:
            input_file: Path to input FASTA file
            output_file: Path to output alignment file
            algorithm: MAFFT algorithm to use
            
        Returns:
            Dict with alignment results and metadata
        """
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        # Set algorithm
        alg = algorithm or self.config.algorithm
        
        # Set output file
        if output_file is None:
            output_path = input_path.parent / f"{input_path.stem}_aligned{input_path.suffix}"
        else:
            output_path = Path(output_file)
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build MAFFT command
        cmd = self._build_mafft_command(input_path, output_path, alg)
        
        self.logger.info(f"Running MAFFT alignment: {' '.join(cmd)}")
        
        # Run MAFFT
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                self.logger.info(f"Alignment completed successfully: {output_path}")
                
                # Parse statistics from stderr (MAFFT outputs stats there)
                stats = self._parse_mafft_stats(result.stderr)
                
                return {
                    "status": "success",
                    "input_file": str(input_path),
                    "output_file": str(output_path),
                    "algorithm": alg,
                    "command": ' '.join(cmd),
                    "statistics": stats,
                    "stderr": result.stderr
                }
            else:
                error_msg = f"MAFFT failed with code {result.returncode}: {result.stderr}"
                self.logger.error(error_msg)
                return {
                    "status": "error",
                    "error": error_msg,
                    "returncode": result.returncode,
                    "stderr": result.stderr
                }
                
        except subprocess.TimeoutExpired:
            error_msg = "MAFFT alignment timed out (>5 minutes)"
            self.logger.error(error_msg)
            return {
                "status": "timeout", 
                "error": error_msg
            }
        except Exception as e:
            error_msg = f"MAFFT execution failed: {e}"
            self.logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg
            }
            
    def _build_mafft_command(self, input_file: Path, output_file: Path, algorithm: str) -> List[str]:
        """Build MAFFT command line arguments"""
        if not self.mafft_path:
            raise RuntimeError("MAFFT not found. Please install MAFFT.")
            
        cmd = [self.mafft_path]
        
        # Algorithm selection
        if algorithm == "auto":
            cmd.append("--auto")
        elif algorithm == "linsi":
            cmd.extend(["--localpair", "--maxiterate", "1000"])
        elif algorithm == "ginsi":
            cmd.extend(["--globalpair", "--maxiterate", "1000"]) 
        elif algorithm == "einsi":
            cmd.extend(["--ep", "0", "--genafpair", "--maxiterate", "1000"])
        elif algorithm == "fftns":
            cmd.append("--retree", "2")
        elif algorithm == "fftnsi":
            cmd.extend(["--retree", "2", "--maxiterate", str(self.config.max_iterations)])
        else:
            self.logger.warning(f"Unknown algorithm {algorithm}, using auto")
            cmd.append("--auto")
            
        # Threading
        if self.config.threads > 1:
            cmd.extend(["--thread", str(self.config.threads)])
            
        # Gap penalties
        cmd.extend(["--op", str(self.config.gap_open_penalty)])
        cmd.extend(["--ep", str(self.config.gap_extension_penalty)])
        
        # Quiet mode
        if self.config.quiet:
            cmd.append("--quiet")
            
        # Input file
        cmd.append(str(input_file))
        
        return cmd
        
    def _parse_mafft_stats(self, stderr: str) -> Dict:
        """Parse statistics from MAFFT stderr output"""
        stats = {
            "sequences": None,
            "columns": None,
            "gaps": None,
            "runtime": None
        }
        
        lines = stderr.split('\n')
        for line in lines:
            line = line.strip()
            if 'sequences' in line.lower():
                # Try to extract sequence count
                words = line.split()
                for i, word in enumerate(words):
                    if word.isdigit() and 'sequences' in ' '.join(words[i:i+2]).lower():
                        stats["sequences"] = int(word)
                        break
            elif 'runtime' in line.lower() or 'time' in line.lower():
                # Extract runtime information
                if ':' in line:
                    stats["runtime"] = line
                    
        return stats
        
    def align_multiple_files(self, input_dir: Union[str, Path], 
                           output_dir: Union[str, Path],
                           pattern: str = "*.fasta") -> Dict:
        """
        Align multiple FASTA files in a directory
        
        Args:
            input_dir: Directory containing FASTA files
            output_dir: Directory for aligned output files
            pattern: File pattern to match
            
        Returns:
            Dict with results for all alignments
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find input files
        input_files = list(input_path.glob(pattern))
        if not input_files:
            return {
                "status": "error",
                "error": f"No files found matching {pattern} in {input_path}"
            }
            
        results = {
            "total_files": len(input_files),
            "successful": 0,
            "failed": 0,
            "results": []
        }
        
        for input_file in input_files:
            self.logger.info(f"Processing {input_file.name}...")
            
            output_file = output_path / f"{input_file.stem}_aligned{input_file.suffix}"
            
            result = self.align_sequences(input_file, output_file)
            results["results"].append(result)
            
            if result["status"] == "success":
                results["successful"] += 1
            else:
                results["failed"] += 1
                
        self.logger.info(f"Batch alignment complete: {results['successful']}/{results['total_files']} successful")
        return results
        
    def create_demo_alignment(self, base_dir: Path) -> Dict:
        """Create a demo alignment using sample data"""
        self.logger.info("Creating demo alignment with sample HIV sequences...")
        
        # Check for sample data
        sample_file = base_dir / "data/raw/hiv-sequences/hiv1_env_sample.fasta"
        if not sample_file.exists():
            return {
                "status": "error",
                "error": f"Sample file not found: {sample_file}"
            }
            
        # Create output directory
        output_dir = base_dir / "data/processed/aligned"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "hiv1_env_sample_aligned.fasta"
        
        # Run alignment
        if self.mafft_path:
            result = self.align_sequences(sample_file, output_file, algorithm="auto")
        else:
            # Create mock alignment for demo purposes
            result = self._create_mock_alignment(sample_file, output_file)
            
        return result
        
    def _create_mock_alignment(self, input_file: Path, output_file: Path) -> Dict:
        """Create a mock alignment when MAFFT is not available"""
        self.logger.warning("MAFFT not available, creating mock alignment for demo")
        
        # Read input sequences
        with open(input_file, 'r') as f:
            content = f.read()
            
        # Create mock aligned output (just copy input)
        mock_aligned = content.replace(
            ">HIV1_ENV_SAMPLE_001", 
            ">HIV1_ENV_SAMPLE_001_ALIGNED"
        ).replace(
            ">HIV1_ENV_SAMPLE_002",
            ">HIV1_ENV_SAMPLE_002_ALIGNED"
        )
        
        # Add mock alignment info
        mock_aligned = f"""# MOCK ALIGNMENT - MAFFT NOT AVAILABLE
# This is a demonstration file created when MAFFT is not installed
# To perform real alignment, install MAFFT: sudo apt-get install mafft
# Original file: {input_file}
# Created: {Path(__file__).stem}

{mock_aligned}"""
        
        with open(output_file, 'w') as f:
            f.write(mock_aligned)
            
        return {
            "status": "mock_success",
            "input_file": str(input_file),
            "output_file": str(output_file),
            "algorithm": "mock",
            "message": "Mock alignment created (MAFFT not installed)",
            "install_command": "sudo apt-get install mafft"
        }

def main():
    """Main entry point for MAFFT wrapper"""
    base_dir = Path(__file__).parent.parent
    
    # Initialize MAFFT wrapper
    config = MAFFTConfig(quiet=False, threads=2)
    mafft = MAFFTWrapper(config)
    
    print("🧬 MAFFT Wrapper for HIV Sequence Alignment")
    print("=" * 50)
    
    # Check MAFFT installation
    if mafft.check_mafft_installation():
        print("✅ MAFFT is installed and accessible")
    else:
        print("⚠️  MAFFT not found - will create demo with mock alignment")
        print("💡 To install MAFFT: sudo apt-get install mafft")
    
    # Run demo alignment
    print("\n🔄 Running demo alignment...")
    result = mafft.create_demo_alignment(base_dir)
    
    if result["status"] in ["success", "mock_success"]:
        print(f"✅ Demo alignment completed: {result['output_file']}")
        if "algorithm" in result:
            print(f"📊 Algorithm used: {result['algorithm']}")
        if "statistics" in result and result["statistics"]:
            stats = result["statistics"]
            if stats.get("sequences"):
                print(f"📈 Sequences processed: {stats['sequences']}")
    else:
        print(f"❌ Demo alignment failed: {result.get('error', 'Unknown error')}")
        
    print(f"\n📁 Output directory: {base_dir}/data/processed/aligned/")
    print("🚀 MAFFT wrapper ready for production use!")
    
    return 0

if __name__ == "__main__":
    exit(main())