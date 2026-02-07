#!/usr/bin/env python3
"""
HIV Data Setup Script
Sets up HIV sequence data structure and provides guidance for manual downloads
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List

class HIVDataSetup:
    """Sets up HIV data structure for analysis"""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.data_dir = base_dir / "data"
        
    def create_data_structure(self) -> None:
        """Create the complete data directory structure"""
        directories = [
            "raw/hiv-sequences",
            "raw/reference-genomes", 
            "processed/aligned",
            "processed/filtered",
            "processed/annotated",
            "metadata/sequences",
            "metadata/downloads",
            "references/hxb2",
            "references/nl4-3",
            "references/subtype-refs"
        ]
        
        for dir_path in directories:
            full_path = self.data_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {full_path}")
            
    def create_sample_data(self) -> None:
        """Create sample HIV sequences for development and testing"""
        
        # Sample HIV-1 Env sequences (abbreviated for demo)
        sample_env = """>HIV1_ENV_SAMPLE_001 HIV-1 Env gp120 sample sequence
MRVKEKYQHLWRWGWRWGTMLLGMLMICSATEKLWVTVYYGVPVWKEATTTLFCASDAKAYDTEVHNVWATHACVPTDPNPQEVVLVNVTENFNMWKNDMVEQMHEDIISLWDQSLKPCVKLTPLCVSLKCTDLKNDTNTNSSSGRMIMEKGEIKNCSFNISTSIRGKVQKEYAFFYKLDIIPIDNDTTSYKLTSCNTSVITQACPKVSFEPIPIHYCAPAGFAILKCNNKTFNGTGPCTNVSTVQCTHGIRPVVSTQLLLNGSLAEEEVVIRSVNFTDNAKTIIVQLNTSVEINCTRPNNNTRKRIRIQRGPGRAFVTIGKIGNMRQAHCNISRAKWNNTLKQIASKLREQFGNNKTIIFKQSSGGDPEIVTHSFNCGGEFFYCNSTQLFNSTWFNSTWSTEGSNNTEGSDTITLPCRIKQIINMWQKVGKAMYAPPISGQIRCSSNITGLLLTRDGGNNNNNESEIFRPGGGDMRDNWRSELYKYKVVKIEPLGVAPTKAKRRVVQREKRAVGIGALFLGFLGAAGSTMGAASMTLTVQARQLLSGIVQQQNNLLRAIEAQQHLLQLTVWGIKQLQARILAVERYLKDQQLLGIWGCSGKLICTTAVPWNASWSNKSLEQIWNHTTWMEWDREINNYTSLIHSLIEESQNQQEKNEQELLELDKWASLWNWFNITNWLWYIKLFIMIVGGLVGLRIVFAVLSIVNRVRQGYSPLSFQTHLPTPRGPDRPEGIEEEGGERDRDRSIRLVNGSLALIWDDLRSLCLFSYHRLRDLLLIVTRIVELLGRRGWEALKYWWNLLQYWSQELKNSAVSLLNATAIAVAEGTDRVIEVVQGACRAIRHIPRRIRQGLERILL
>HIV1_ENV_SAMPLE_002 HIV-1 Env gp120 sample sequence variant
MRVKEKYQHLWRWGWRWGTMLLGMLMICSATEKLWVTVYYGVPVWKEATTTLFCASDAKAYDTEVHNVWATHACVPTDPNPQEVVLVNVTENFNMWKNDMVEQMHEDIISLWDQSLKPCVKLTPLCVSLKCTDLKNDTNTNSSSGRMIMEKGEIKNCSFNISTSIRGKVQKEYAFFYKLDIIPIDNDTTSYKLTSCNTSVITQACPKVSFEPIPIHYCAPAGFAILKCNNKTFNGTGPCTNVSTVQCTHGIRPVVSTQLLLNGSLAEEEVVIRSVNFTDNAKTIIVQLNTSVEINCTRPNNNTRKRIRIQRGPGRAFVTIGKIGNMRQAHCNISRAKWNNTLKQIASKLREQFGNNKTIIFKQSSGGDPEIVTHSFNCGGEFFYCNSTQLFNSTWFNSTWSTEGSNNTEGSDTITLPCRIKQIINMWQKVGKAMYAPPISGQIRCSSNITGLLLTRDGGNNNNNESEIFRPGGGDMRDNWRSELYKYKVVKIEPLGVAPTKAKRRVVQREKRAVGIGALFLGFLGAAGSTMGAASMTLTVQARQLLSGIVQQQNNLLRAIEAQQHLLQLTVWGIKQLQARILAVERYLKDQQLLGIWGCSGKLICTTAVPWNASWSNKSLEQIWNHTTWMEWDREINNYTSLIHSLIEESQNQQEKNEQELLELDKWASLWNWFNITNWLWYIKLFIMIVGGLVGLRIVFAVLSIVNRVRQGYSPLSFQTHLPTPRGPDRPEGIEEEGGERDRDRSIRLVNGSLALIWDDLRSLCLFSYHRLRDLLLIVTRIVELLGRRGWEALKYWWNLLQYWSQELKNSAVSLLNATAIAVAEGTDRVIEVVQGACRAIRHIPRRIRQGLERILL
"""
        
        # Save sample data
        sample_file = self.data_dir / "raw/hiv-sequences/hiv1_env_sample.fasta"
        with open(sample_file, 'w') as f:
            f.write(sample_env)
        print(f"✅ Created sample data: {sample_file}")
        
    def create_download_guide(self) -> None:
        """Create comprehensive guide for manual data downloads"""
        
        guide = """# HIV Sequence Download Guide

## Los Alamos HIV Database

The Los Alamos National Laboratory HIV Database is the primary source for HIV sequence data.

### Key Datasets to Download:

1. **HIV-1 Env Sequences**
   - URL: https://www.hiv.lanl.gov/content/sequence/HIV/ALIGNMENTS/ENV/
   - Files: Latest alignment files (amino acid and nucleotide)
   - Purpose: Envelope protein analysis, neutralization studies

2. **HIV-1 Gag Sequences** 
   - URL: https://www.hiv.lanl.gov/content/sequence/HIV/ALIGNMENTS/GAG/
   - Files: Latest alignment files
   - Purpose: Structural protein analysis, CTL epitopes

3. **HIV-1 Pol Sequences**
   - URL: https://www.hiv.lanl.gov/content/sequence/HIV/ALIGNMENTS/POL/
   - Files: Latest alignment files  
   - Purpose: Drug resistance analysis

4. **HIV-1 Complete Genomes**
   - URL: https://www.hiv.lanl.gov/content/sequence/HIV/COMPENDIUM/
   - Purpose: Comprehensive genomic analysis

### Manual Download Process:

1. Visit the Los Alamos HIV Database: https://www.hiv.lanl.gov/
2. Navigate to "Sequence Data" → "HIV-1 Alignments"
3. Select protein of interest (Env, Gag, Pol)
4. Download the latest year's alignment files
5. Save to `data/raw/hiv-sequences/` directory

### File Naming Convention:

- `hiv1_env_[year]_[type].fasta` (e.g., hiv1_env_2023_amino.fasta)
- `hiv1_gag_[year]_[type].fasta`
- `hiv1_pol_[year]_[type].fasta`

### Alternative Sources:

1. **NCBI HIV Database**
   - URL: https://www.ncbi.nlm.nih.gov/genome/viruses/retroviruses/hiv-1/
   - For specific strain sequences

2. **Stanford HIV Database**
   - URL: https://hivdb.stanford.edu/
   - For drug resistance data

3. **GISAID** (Registration required)
   - URL: https://www.gisaid.org/
   - For recent sequences and metadata

## Reference Sequences

Download these key reference genomes:

1. **HXB2** (K03455) - Primary HIV-1 reference
2. **NL4-3** (AF324493) - Laboratory reference strain
3. **Consensus sequences** by subtype from Los Alamos

## Data Verification

After downloading:
1. Check file sizes and sequence counts
2. Verify FASTA format integrity  
3. Compare with previous downloads for consistency
4. Document download date and source

## Automation Notes

The Los Alamos site has anti-automation measures. Manual download is recommended.
For bulk downloads, contact the database maintainers for API access.
"""

        guide_file = self.data_dir / "DOWNLOAD_GUIDE.md"
        with open(guide_file, 'w') as f:
            f.write(guide)
        print(f"✅ Created download guide: {guide_file}")
        
    def create_metadata_files(self) -> None:
        """Create metadata and configuration files"""
        
        # Data manifest
        manifest = {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "purpose": "HIV sequence analysis pipeline",
            "structure": {
                "raw/hiv-sequences": "Original sequences from databases",
                "raw/reference-genomes": "Reference genomes (HXB2, NL4-3, etc)",
                "processed/aligned": "Multiple sequence alignments",
                "processed/filtered": "Quality-filtered sequences",
                "processed/annotated": "Annotated sequences with metadata",
                "metadata/sequences": "Sequence metadata and annotations",
                "metadata/downloads": "Download logs and manifests",
                "references": "Reference sequences and consensus"
            },
            "sources": {
                "primary": "Los Alamos HIV Database (https://www.hiv.lanl.gov/)",
                "secondary": ["NCBI", "Stanford HIV DB", "GISAID"],
                "manual_download_required": True,
                "automation_blocked": "Anti-scraping measures present"
            },
            "next_steps": [
                "Manual download of sequence alignments",
                "Implementation of MAFFT alignment wrapper",
                "Creation of sequence alignment viewer",
                "Conservation scoring implementation"
            ]
        }
        
        manifest_file = self.data_dir / "manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"✅ Created data manifest: {manifest_file}")
        
        # Dataset registry
        datasets = {
            "available_datasets": {
                "hiv1_env_alignments": {
                    "description": "HIV-1 Envelope protein sequences",
                    "source": "Los Alamos HIV Database",
                    "update_frequency": "Annual",
                    "last_update": "2023",
                    "sequence_count": "~15000",
                    "file_pattern": "hiv1_env_*.fasta"
                },
                "hiv1_gag_alignments": {
                    "description": "HIV-1 Gag protein sequences", 
                    "source": "Los Alamos HIV Database",
                    "update_frequency": "Annual",
                    "last_update": "2023", 
                    "sequence_count": "~8000",
                    "file_pattern": "hiv1_gag_*.fasta"
                },
                "hiv1_pol_alignments": {
                    "description": "HIV-1 Pol protein sequences",
                    "source": "Los Alamos HIV Database", 
                    "update_frequency": "Annual",
                    "last_update": "2023",
                    "sequence_count": "~12000",
                    "file_pattern": "hiv1_pol_*.fasta"
                }
            },
            "status": "setup_complete_manual_download_required"
        }
        
        registry_file = self.data_dir / "metadata/datasets.json"
        with open(registry_file, 'w') as f:
            json.dump(datasets, f, indent=2)
        print(f"✅ Created dataset registry: {registry_file}")

def main():
    """Main setup function"""
    print("🔬 Setting up HIV sequence analysis data structure...")
    
    # Get base directory
    base_dir = Path(__file__).parent.parent
    setup = HIVDataSetup(base_dir)
    
    # Create structure and files
    setup.create_data_structure()
    setup.create_sample_data()
    setup.create_download_guide()
    setup.create_metadata_files()
    
    print("\n✅ HIV data setup completed!")
    print(f"📁 Data directory: {setup.data_dir}")
    print(f"📋 Download guide: {setup.data_dir}/DOWNLOAD_GUIDE.md")
    print("\n📌 Next Steps:")
    print("1. Follow the download guide to manually obtain HIV sequences")
    print("2. Run sequence validation and quality checks") 
    print("3. Proceed with T003: Implement MAFFT wrapper")
    
    return 0

if __name__ == "__main__":
    exit(main())