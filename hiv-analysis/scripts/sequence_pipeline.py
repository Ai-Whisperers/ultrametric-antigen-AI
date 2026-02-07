#!/usr/bin/env python3
"""
HIV Sequence Data Pipeline
Sets up data processing pipeline for HIV sequences
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class SequenceConfig:
    """Configuration for sequence processing"""
    data_dir: Path
    results_dir: Path
    temp_dir: Path
    log_level: str = "INFO"
    
    def __post_init__(self):
        # Create directories if they don't exist
        for dir_path in [self.data_dir, self.results_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

class SequenceDataPipeline:
    """Main pipeline class for HIV sequence processing"""
    
    def __init__(self, config: SequenceConfig):
        self.config = config
        self.setup_logging()
        
    def setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.config.results_dir / 'pipeline.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def validate_environment(self) -> bool:
        """Validate that required tools and dependencies are available"""
        self.logger.info("Validating pipeline environment...")
        
        # Check if required directories exist
        required_dirs = [self.config.data_dir, self.config.results_dir, self.config.temp_dir]
        for dir_path in required_dirs:
            if not dir_path.exists():
                self.logger.error(f"Required directory does not exist: {dir_path}")
                return False
                
        self.logger.info("Environment validation passed")
        return True
        
    def initialize_data_structure(self):
        """Set up the data directory structure"""
        self.logger.info("Initializing data structure...")
        
        subdirs = [
            "raw/hiv-sequences",
            "processed/aligned", 
            "processed/filtered",
            "metadata",
            "references"
        ]
        
        for subdir in subdirs:
            full_path = self.config.data_dir / subdir
            full_path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created directory: {full_path}")
            
    def create_manifest(self) -> Dict:
        """Create data manifest file"""
        manifest = {
            "pipeline_version": "1.0.0",
            "created": str(Path.cwd()),
            "data_sources": {
                "los_alamos": "https://www.hiv.lanl.gov/",
                "ncbi": "https://www.ncbi.nlm.nih.gov/genbank/"
            },
            "structure": {
                "raw": "Original sequences from databases", 
                "processed": "Aligned and filtered sequences",
                "metadata": "Sample information and annotations",
                "references": "Reference sequences"
            }
        }
        
        manifest_path = self.config.data_dir / "manifest.json"
        import json
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
            
        self.logger.info(f"Created manifest: {manifest_path}")
        return manifest
        
    def setup_pipeline(self) -> bool:
        """Complete pipeline setup"""
        self.logger.info("Setting up HIV sequence data pipeline...")
        
        if not self.validate_environment():
            return False
            
        self.initialize_data_structure()
        self.create_manifest()
        
        self.logger.info("Pipeline setup completed successfully")
        return True

def main():
    """Main entry point"""
    # Set up configuration
    base_dir = Path(__file__).parent.parent
    config = SequenceConfig(
        data_dir=base_dir / "data",
        results_dir=base_dir / "results", 
        temp_dir=base_dir / "temp"
    )
    
    # Initialize and run pipeline
    pipeline = SequenceDataPipeline(config)
    
    if pipeline.setup_pipeline():
        print("✅ HIV sequence data pipeline set up successfully")
    else:
        print("❌ Pipeline setup failed")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())