# HIV Sequence Download Guide

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
