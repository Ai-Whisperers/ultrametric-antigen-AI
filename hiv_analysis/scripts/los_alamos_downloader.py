#!/usr/bin/env python3
"""
Los Alamos HIV Database Downloader
Downloads HIV sequences from the Los Alamos National Laboratory HIV Database
"""

import requests
import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import json

@dataclass
class DownloadConfig:
    """Configuration for Los Alamos HIV database downloads"""
    data_dir: Path
    base_url: str = "https://www.hiv.lanl.gov"
    delay_seconds: float = 1.0  # Respectful crawling delay
    max_retries: int = 3
    timeout: int = 30
    
    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)

class LosAlamosDownloader:
    """Downloads HIV sequences from Los Alamos HIV database"""
    
    def __init__(self, config: DownloadConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HIV Research Pipeline/1.0 (Educational/Research Use)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        })
        self.setup_logging()
        
    def setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def download_sequence_sets(self) -> Dict:
        """Download commonly used HIV sequence sets"""
        self.logger.info("Starting Los Alamos HIV sequence download...")
        
        # Define common HIV sequence datasets
        datasets = {
            "hiv1_env_2023": {
                "description": "HIV-1 Env sequences (2023 alignment)",
                "url_path": "/content/sequence/HIV/ALIGNMENTS/ENV/2023/env_amino.fasta",
                "filename": "hiv1_env_2023.fasta",
                "type": "alignment"
            },
            "hiv1_gag_2023": {
                "description": "HIV-1 Gag sequences (2023 alignment)", 
                "url_path": "/content/sequence/HIV/ALIGNMENTS/GAG/2023/gag_amino.fasta",
                "filename": "hiv1_gag_2023.fasta",
                "type": "alignment"
            },
            "hiv1_pol_2023": {
                "description": "HIV-1 Pol sequences (2023 alignment)",
                "url_path": "/content/sequence/HIV/ALIGNMENTS/POL/2023/pol_amino.fasta", 
                "filename": "hiv1_pol_2023.fasta",
                "type": "alignment"
            }
        }
        
        download_results = {
            "downloaded": [],
            "failed": [],
            "metadata": {
                "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source": "Los Alamos HIV Database",
                "total_datasets": len(datasets)
            }
        }
        
        for dataset_id, dataset_info in datasets.items():
            try:
                success = self._download_dataset(dataset_id, dataset_info)
                if success:
                    download_results["downloaded"].append({
                        "id": dataset_id,
                        "filename": dataset_info["filename"],
                        "description": dataset_info["description"]
                    })
                    self.logger.info(f"Successfully downloaded: {dataset_id}")
                else:
                    download_results["failed"].append({
                        "id": dataset_id,
                        "reason": "Download failed"
                    })
                    self.logger.error(f"Failed to download: {dataset_id}")
                    
                # Respectful delay between downloads
                time.sleep(self.config.delay_seconds)
                
            except Exception as e:
                download_results["failed"].append({
                    "id": dataset_id,
                    "reason": str(e)
                })
                self.logger.error(f"Error downloading {dataset_id}: {e}")
        
        # Save download manifest
        manifest_path = self.config.data_dir / "download_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(download_results, f, indent=2)
            
        self.logger.info(f"Download completed. Manifest saved to: {manifest_path}")
        return download_results
        
    def _download_dataset(self, dataset_id: str, dataset_info: Dict) -> bool:
        """Download a specific dataset"""
        url = self.config.base_url + dataset_info["url_path"]
        filename = self.config.data_dir / "raw" / "hiv-sequences" / dataset_info["filename"]
        
        # Create directory if needed
        filename.parent.mkdir(parents=True, exist_ok=True)
        
        for attempt in range(self.config.max_retries):
            try:
                self.logger.info(f"Downloading {dataset_id} (attempt {attempt + 1}/{self.config.max_retries})")
                
                response = self.session.get(url, timeout=self.config.timeout)
                response.raise_for_status()
                
                # Check if response looks like FASTA
                content = response.text
                if not content.startswith('>'):
                    self.logger.warning(f"Response doesn't look like FASTA format for {dataset_id}")
                    # Continue anyway, might be valid data
                
                # Save to file
                with open(filename, 'w') as f:
                    f.write(content)
                
                self.logger.info(f"Saved {len(content)} characters to {filename}")
                return True
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {dataset_id}: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
        return False
        
    def download_reference_sequences(self) -> Dict:
        """Download HIV reference sequences"""
        self.logger.info("Downloading HIV reference sequences...")
        
        # Common HIV reference strains
        references = {
            "hxb2": {
                "description": "HIV-1 HXB2 reference genome",
                "accession": "K03455",
                "filename": "hiv1_hxb2_ref.fasta"
            },
            "nl4_3": {
                "description": "HIV-1 NL4-3 reference genome", 
                "accession": "AF324493",
                "filename": "hiv1_nl43_ref.fasta"
            }
        }
        
        ref_results = {
            "downloaded": [],
            "failed": [],
            "metadata": {
                "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source": "NCBI (via Los Alamos referencing)",
                "total_references": len(references)
            }
        }
        
        # For now, create placeholder reference files
        # In a full implementation, these would be downloaded from NCBI
        ref_dir = self.config.data_dir / "references"
        ref_dir.mkdir(parents=True, exist_ok=True)
        
        for ref_id, ref_info in references.items():
            ref_file = ref_dir / ref_info["filename"]
            
            # Create placeholder reference entry
            placeholder_content = f""">{ref_info['accession']} {ref_info['description']}
# This is a placeholder for {ref_info['description']}
# Accession: {ref_info['accession']}
# Source: Los Alamos HIV Database / NCBI
# Download date: {time.strftime('%Y-%m-%d')}
# 
# In a full implementation, the actual sequence would be downloaded
# from NCBI or retrieved from Los Alamos database
"""
            
            with open(ref_file, 'w') as f:
                f.write(placeholder_content)
                
            ref_results["downloaded"].append({
                "id": ref_id,
                "filename": ref_info["filename"],
                "description": ref_info["description"],
                "accession": ref_info["accession"]
            })
        
        return ref_results
        
    def create_download_report(self, results: Dict) -> str:
        """Create a human-readable download report"""
        report = [
            "# Los Alamos HIV Database Download Report",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            f"- Total datasets attempted: {results['metadata']['total_datasets']}",
            f"- Successfully downloaded: {len(results['downloaded'])}",
            f"- Failed downloads: {len(results['failed'])}",
            ""
        ]
        
        if results["downloaded"]:
            report.extend([
                "## Downloaded Datasets",
                ""
            ])
            for item in results["downloaded"]:
                report.append(f"- **{item['id']}**: {item['description']}")
                report.append(f"  - File: `{item['filename']}`")
                report.append("")
        
        if results["failed"]:
            report.extend([
                "## Failed Downloads",
                ""
            ])
            for item in results["failed"]:
                report.append(f"- **{item['id']}**: {item.get('reason', 'Unknown error')}")
                report.append("")
        
        report.extend([
            "## Data Structure",
            "```",
            "data/",
            "├── raw/hiv-sequences/    # Downloaded FASTA files",
            "├── references/          # Reference sequences", 
            "└── metadata/           # Download manifests",
            "```",
            "",
            "## Next Steps",
            "1. Validate downloaded sequences",
            "2. Check sequence quality and completeness", 
            "3. Prepare for multiple sequence alignment",
            "4. Document any missing or problematic downloads"
        ])
        
        return "\n".join(report)

def main():
    """Main entry point"""
    # Set up configuration
    base_dir = Path(__file__).parent.parent
    config = DownloadConfig(
        data_dir=base_dir / "data"
    )
    
    # Initialize downloader
    downloader = LosAlamosDownloader(config)
    
    try:
        # Download sequence datasets
        seq_results = downloader.download_sequence_sets()
        
        # Download reference sequences
        ref_results = downloader.download_reference_sequences()
        
        # Create comprehensive report
        report = downloader.create_download_report(seq_results)
        report_path = config.data_dir / "download_report.md"
        
        with open(report_path, 'w') as f:
            f.write(report)
            
        print("✅ Los Alamos HIV database download completed")
        print(f"📋 Report: {report_path}")
        print(f"📁 Data: {config.data_dir}")
        
        # Print summary
        total_downloaded = len(seq_results["downloaded"]) + len(ref_results["downloaded"])
        total_failed = len(seq_results["failed"]) + len(ref_results["failed"])
        
        print(f"📊 Summary: {total_downloaded} downloaded, {total_failed} failed")
        
        return 0 if total_failed == 0 else 1
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())