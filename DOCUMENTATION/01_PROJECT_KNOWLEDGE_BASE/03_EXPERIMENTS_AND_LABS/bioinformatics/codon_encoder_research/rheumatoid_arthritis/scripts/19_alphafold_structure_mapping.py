#!/usr/bin/env python3
"""
AlphaFold2 Structure Mapping for Citrullination Sites

Map entropy change predictions onto 3D protein structures from AlphaFold2.
Analyze structural correlates of immunogenicity.

Output directory: results/structural/19_alphafold_mapping/

Version: 1.0
"""

import json
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ============================================================================
# CONFIGURATION
# ============================================================================

# AlphaFold DB API
ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/files"

# Key RA autoantigens with UniProt IDs
RA_AUTOANTIGENS = {
    'VIM': 'P08670',      # Vimentin
    'FGA': 'P02671',      # Fibrinogen alpha
    'FGB': 'P02675',      # Fibrinogen beta
    'FGG': 'P02679',      # Fibrinogen gamma
    'ENO1': 'P06733',     # Alpha-enolase
    'COL2A1': 'P02458',   # Collagen type II
    'FLG': 'P20930',      # Filaggrin
    'HIST1H2A': 'P0C0S8', # Histone H2A
    'HIST1H3A': 'P68431', # Histone H3
    'HIST1H4A': 'P62805', # Histone H4
    'TNC': 'P24821',      # Tenascin-C
    'FN1': 'P02751',      # Fibronectin
    'HSPA5': 'P11021',    # BiP/GRP78
    'CLU': 'P10909',      # Clusterin
}

# Output configuration
SCRIPT_NUM = "19"
OUTPUT_SUBDIR = f"{SCRIPT_NUM}_alphafold_mapping"


# ============================================================================
# DIRECTORY SETUP
# ============================================================================

def get_output_dir() -> Path:
    """Get output directory for this script."""
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results" / "structural" / OUTPUT_SUBDIR
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def get_pdb_dir() -> Path:
    """Get directory for downloaded PDB files."""
    output_dir = get_output_dir()
    pdb_dir = output_dir / "pdb_files"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    return pdb_dir


# ============================================================================
# ALPHAFOLD DOWNLOAD
# ============================================================================

def download_alphafold_structure(uniprot_id: str, output_dir: Path) -> Optional[Path]:
    """
    Download AlphaFold2 predicted structure for a protein.

    Returns path to downloaded PDB file or None if failed.
    """
    # Try different versions (v6 is current, fallback to v4)
    for version in ['v6', 'v4']:
        pdb_path = output_dir / f"AF-{uniprot_id}-F1-model_{version}.pdb"

        # Check if already downloaded
        if pdb_path.exists():
            return pdb_path

        # Download from AlphaFold DB
        url = f"{ALPHAFOLD_API}/AF-{uniprot_id}-F1-model_{version}.pdb"

        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(pdb_path, 'w') as f:
                    f.write(response.text)
                return pdb_path
        except Exception as e:
            continue

    # If loop completes without success
    print(f"  Failed to download {uniprot_id}: structure not available")
    return None


def download_alphafold_confidence(uniprot_id: str, output_dir: Path) -> Optional[Path]:
    """Download AlphaFold2 confidence scores (pLDDT)."""
    json_path = output_dir / f"AF-{uniprot_id}-F1-confidence_v4.json"

    if json_path.exists():
        with open(json_path, 'r') as f:
            return json.load(f)

    url = f"{ALPHAFOLD_API}/AF-{uniprot_id}-F1-confidence_v4.json"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            with open(json_path, 'w') as f:
                json.dump(data, f)
            return data
        else:
            return None
    except Exception as e:
        print(f"  Error downloading confidence for {uniprot_id}: {e}")
        return None


# ============================================================================
# PDB PARSING
# ============================================================================

def parse_pdb(pdb_path: Path) -> Dict:
    """
    Parse PDB file to extract residue information.

    Returns dict with:
        - residues: list of (resname, resnum, x, y, z, bfactor)
        - arginines: list of arginine positions
        - ca_coords: dict mapping resnum to CA coordinates
    """
    residues = []
    ca_coords = {}
    arginines = []

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                resname = line[17:20].strip()
                resnum = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                bfactor = float(line[60:66].strip()) if len(line) > 66 else 0.0

                if atom_name == 'CA':
                    ca_coords[resnum] = (x, y, z)
                    residues.append({
                        'resname': resname,
                        'resnum': resnum,
                        'x': x, 'y': y, 'z': z,
                        'bfactor': bfactor  # pLDDT in AlphaFold
                    })

                    if resname == 'ARG':
                        arginines.append(resnum)

    return {
        'residues': residues,
        'arginines': arginines,
        'ca_coords': ca_coords
    }


def compute_solvent_accessibility(ca_coords: Dict, resnum: int,
                                   neighbor_radius: float = 10.0) -> float:
    """
    Estimate relative solvent accessibility based on neighbor density.

    Lower neighbor count = more exposed = higher accessibility
    """
    if resnum not in ca_coords:
        return 0.0

    x, y, z = ca_coords[resnum]
    neighbor_count = 0

    for other_resnum, (ox, oy, oz) in ca_coords.items():
        if other_resnum == resnum:
            continue
        dist = np.sqrt((x-ox)**2 + (y-oy)**2 + (z-oz)**2)
        if dist < neighbor_radius:
            neighbor_count += 1

    # Normalize: fewer neighbors = more accessible
    # Typical range: 0-30 neighbors within 10A
    accessibility = max(0, 1 - neighbor_count / 30)
    return accessibility


def compute_secondary_structure_proxy(ca_coords: Dict, resnum: int) -> str:
    """
    Estimate secondary structure based on CA-CA distances.

    Helix: ~5.4A per 4 residues
    Sheet: ~6.8A per 2 residues
    Coil: variable
    """
    if resnum not in ca_coords:
        return 'unknown'

    # Check helix pattern (i to i+4)
    if resnum + 4 in ca_coords:
        x1, y1, z1 = ca_coords[resnum]
        x2, y2, z2 = ca_coords[resnum + 4]
        dist = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        if 5.0 < dist < 6.5:
            return 'helix'

    # Check sheet pattern (i to i+2)
    if resnum + 2 in ca_coords:
        x1, y1, z1 = ca_coords[resnum]
        x2, y2, z2 = ca_coords[resnum + 2]
        dist = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        if 6.0 < dist < 7.5:
            return 'sheet'

    return 'coil'


# ============================================================================
# EPITOPE DATA
# ============================================================================

def load_epitope_data() -> List[Dict]:
    """Load epitope data with entropy change values."""
    script_dir = Path(__file__).parent
    results_path = script_dir.parent / "results" / "hyperbolic" / "goldilocks_validation" / "goldilocks_validation_results.json"

    if not results_path.exists():
        print(f"  Warning: Goldilocks results not found at {results_path}")
        return []

    with open(results_path, 'r') as f:
        data = json.load(f)

    return data.get('epitope_details', [])


# ============================================================================
# STRUCTURAL ANALYSIS
# ============================================================================

def analyze_protein_structure(gene_name: str, uniprot_id: str,
                              epitopes: List[Dict], pdb_dir: Path) -> Optional[Dict]:
    """
    Analyze structure for a single protein.

    Maps epitopes onto structure and computes structural features.
    """
    # Download structure
    pdb_path = download_alphafold_structure(uniprot_id, pdb_dir)
    if pdb_path is None:
        return None

    # Parse structure
    structure = parse_pdb(pdb_path)

    # Get protein-specific epitopes
    protein_epitopes = [e for e in epitopes if e['epitope_id'].startswith(gene_name)]

    if not protein_epitopes:
        return None

    # Analyze each arginine site
    site_analyses = []

    for epitope in protein_epitopes:
        # Extract arginine position from epitope ID (e.g., "VIM_R71" -> 71)
        epitope_id = epitope['epitope_id']
        try:
            r_pos = int(epitope_id.split('_R')[-1].split('_')[0])
        except:
            # Handle special cases like "ENO1_CEP1"
            continue

        # Get structural features for this position
        if r_pos in structure['ca_coords']:
            accessibility = compute_solvent_accessibility(structure['ca_coords'], r_pos)
            ss = compute_secondary_structure_proxy(structure['ca_coords'], r_pos)

            # Get pLDDT (B-factor in AlphaFold PDB)
            plddt = 0
            for res in structure['residues']:
                if res['resnum'] == r_pos:
                    plddt = res['bfactor']
                    break

            site_analyses.append({
                'epitope_id': epitope_id,
                'r_position': r_pos,
                'entropy_change': epitope.get('mean_entropy_change', 0),
                'immunodominant': epitope.get('immunodominant', False),
                'acpa_reactivity': epitope.get('acpa_reactivity', 0),
                'solvent_accessibility': accessibility,
                'secondary_structure': ss,
                'plddt': plddt,
                'x': structure['ca_coords'][r_pos][0],
                'y': structure['ca_coords'][r_pos][1],
                'z': structure['ca_coords'][r_pos][2],
            })

    return {
        'gene_name': gene_name,
        'uniprot_id': uniprot_id,
        'total_residues': len(structure['residues']),
        'total_arginines': len(structure['arginines']),
        'analyzed_sites': site_analyses,
        'pdb_path': str(pdb_path),
    }


def generate_pymol_script(protein_analysis: Dict, output_dir: Path) -> Path:
    """
    Generate PyMOL script to visualize entropy change on structure.

    Colors arginines by entropy change:
    - Red: high entropy increase (immunodominant)
    - Blue: high entropy decrease (silent)
    - White: neutral
    """
    gene = protein_analysis['gene_name']
    script_path = output_dir / f"visualize_{gene}.pml"

    pdb_path = protein_analysis['pdb_path']
    sites = protein_analysis['analyzed_sites']

    script_lines = [
        f"# PyMOL visualization script for {gene}",
        f"# Generated by 19_alphafold_structure_mapping.py",
        "",
        f"load {pdb_path}, {gene}",
        f"hide all",
        f"show cartoon, {gene}",
        f"color gray80, {gene}",
        "",
        "# Color arginines by entropy change",
        "# Red = entropy increase (immunodominant)",
        "# Blue = entropy decrease (silent)",
        "",
    ]

    for site in sites:
        r_pos = site['r_position']
        entropy = site['entropy_change']
        imm = site['immunodominant']

        # Color scale: -0.15 (blue) to +0.05 (red)
        # Normalize to 0-1 range
        normalized = (entropy + 0.15) / 0.20
        normalized = max(0, min(1, normalized))

        # RGB interpolation: blue (0,0,1) to white (1,1,1) to red (1,0,0)
        if normalized < 0.5:
            r = 2 * normalized
            g = 2 * normalized
            b = 1.0
        else:
            r = 1.0
            g = 2 * (1 - normalized)
            b = 2 * (1 - normalized)

        color_name = f"entropy_{gene}_{r_pos}"
        script_lines.append(f"set_color {color_name}, [{r:.3f}, {g:.3f}, {b:.3f}]")
        script_lines.append(f"color {color_name}, {gene} and resi {r_pos}")
        script_lines.append(f"show sticks, {gene} and resi {r_pos}")

        # Label immunodominant sites
        if imm:
            script_lines.append(f"label {gene} and resi {r_pos} and name CA, \"R{r_pos}*\"")

    script_lines.extend([
        "",
        "# Styling",
        "bg_color white",
        "set ray_shadows, 0",
        "set antialias, 2",
        "set cartoon_fancy_helices, 1",
        f"zoom {gene}",
        "",
        f"# Save image",
        f"ray 1920, 1080",
        f"png {output_dir}/{gene}_entropy_map.png, dpi=300",
    ])

    with open(script_path, 'w') as f:
        f.write('\n'.join(script_lines))

    return script_path


def generate_html_viewer(protein_analysis: Dict, output_dir: Path) -> Path:
    """
    Generate interactive HTML viewer using 3Dmol.js
    """
    gene = protein_analysis['gene_name']
    html_path = output_dir / f"view_{gene}.html"

    pdb_path = Path(protein_analysis['pdb_path'])
    sites = protein_analysis['analyzed_sites']

    # Read PDB content
    with open(pdb_path, 'r') as f:
        pdb_content = f.read().replace('\n', '\\n').replace("'", "\\'")

    # Build site data for JavaScript
    site_js_data = []
    for site in sites:
        site_js_data.append({
            'resi': site['r_position'],
            'entropy': site['entropy_change'],
            'immunodominant': site['immunodominant'],
            'acpa': site['acpa_reactivity'],
        })

    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>{gene} - Entropy Change Mapping</title>
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
        #container {{ width: 800px; height: 600px; position: relative; }}
        #viewer {{ width: 100%; height: 100%; }}
        .legend {{ margin-top: 10px; }}
        .legend-item {{ display: inline-block; margin-right: 20px; }}
        .color-box {{ width: 20px; height: 20px; display: inline-block; vertical-align: middle; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f4f4f4; }}
        .imm {{ background-color: #ffcccc; }}
        .silent {{ background-color: #ccccff; }}
    </style>
</head>
<body>
    <h1>{gene} - Citrullination Site Entropy Mapping</h1>
    <p>UniProt: {protein_analysis['uniprot_id']} | Residues: {protein_analysis['total_residues']} | Arginines: {protein_analysis['total_arginines']}</p>

    <div id="container">
        <div id="viewer"></div>
    </div>

    <div class="legend">
        <span class="legend-item"><span class="color-box" style="background: #ff6666;"></span> High entropy (immunodominant)</span>
        <span class="legend-item"><span class="color-box" style="background: #ffffff; border: 1px solid #ccc;"></span> Neutral</span>
        <span class="legend-item"><span class="color-box" style="background: #6666ff;"></span> Low entropy (silent)</span>
    </div>

    <h2>Arginine Sites Analysis</h2>
    <table>
        <tr>
            <th>Position</th>
            <th>Entropy Change</th>
            <th>Type</th>
            <th>ACPA Reactivity</th>
            <th>Accessibility</th>
            <th>Structure</th>
        </tr>
        {''.join(f'''<tr class="{'imm' if s['immunodominant'] else 'silent'}">
            <td>R{s['r_position']}</td>
            <td>{s['entropy_change']:.4f}</td>
            <td>{'Immunodominant' if s['immunodominant'] else 'Silent'}</td>
            <td>{s['acpa_reactivity']:.2f}</td>
            <td>{s['solvent_accessibility']:.2f}</td>
            <td>{s['secondary_structure']}</td>
        </tr>''' for s in sites)}
    </table>

    <script>
        let viewer = $3Dmol.createViewer("viewer", {{backgroundColor: "white"}});

        let pdbData = '{pdb_content}';
        viewer.addModel(pdbData, "pdb");

        // Base style: cartoon in gray
        viewer.setStyle({{}}, {{cartoon: {{color: "gray"}}}});

        // Color arginines by entropy change
        let sites = {json.dumps(site_js_data)};

        sites.forEach(function(site) {{
            // Normalize entropy to color
            let norm = (site.entropy + 0.15) / 0.20;
            norm = Math.max(0, Math.min(1, norm));

            let r, g, b;
            if (norm < 0.5) {{
                r = Math.round(255 * 2 * norm);
                g = Math.round(255 * 2 * norm);
                b = 255;
            }} else {{
                r = 255;
                g = Math.round(255 * 2 * (1 - norm));
                b = Math.round(255 * 2 * (1 - norm));
            }}

            let color = "rgb(" + r + "," + g + "," + b + ")";

            viewer.setStyle({{resi: site.resi}}, {{
                cartoon: {{color: color}},
                stick: {{color: color}}
            }});

            // Label immunodominant sites
            if (site.immunodominant) {{
                viewer.addLabel("R" + site.resi, {{
                    position: {{resi: site.resi}},
                    backgroundColor: "rgba(255,0,0,0.7)",
                    fontColor: "white",
                    fontSize: 12
                }});
            }}
        }});

        viewer.zoomTo();
        viewer.render();
    </script>
</body>
</html>'''

    with open(html_path, 'w') as f:
        f.write(html_content)

    return html_path


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def analyze_structural_correlations(all_analyses: List[Dict], output_dir: Path) -> Dict:
    """
    Analyze correlations between structural features and immunogenicity.
    """
    # Collect all site data
    all_sites = []
    for analysis in all_analyses:
        if analysis and 'analyzed_sites' in analysis:
            all_sites.extend(analysis['analyzed_sites'])

    if not all_sites:
        return {}

    df = pd.DataFrame(all_sites)

    # Compute correlations
    correlations = {}

    # Entropy vs accessibility
    if 'solvent_accessibility' in df.columns and 'entropy_change' in df.columns:
        from scipy import stats
        r, p = stats.pearsonr(df['solvent_accessibility'], df['entropy_change'])
        correlations['entropy_vs_accessibility'] = {'r': r, 'p': p}

    # Entropy vs pLDDT
    if 'plddt' in df.columns and 'entropy_change' in df.columns:
        r, p = stats.pearsonr(df['plddt'], df['entropy_change'])
        correlations['entropy_vs_plddt'] = {'r': r, 'p': p}

    # Secondary structure distribution
    if 'secondary_structure' in df.columns and 'immunodominant' in df.columns:
        ss_dist = df.groupby(['immunodominant', 'secondary_structure']).size().unstack(fill_value=0)
        correlations['ss_distribution'] = ss_dist.to_dict()

    # Accessibility by immunodominance
    if 'solvent_accessibility' in df.columns and 'immunodominant' in df.columns:
        imm_acc = df[df['immunodominant']]['solvent_accessibility'].mean()
        silent_acc = df[~df['immunodominant']]['solvent_accessibility'].mean()
        correlations['mean_accessibility'] = {
            'immunodominant': imm_acc,
            'silent': silent_acc
        }

    # Generate correlation plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Entropy vs Accessibility
    ax = axes[0, 0]
    colors = ['#e53935' if imm else '#1e88e5' for imm in df['immunodominant']]
    ax.scatter(df['solvent_accessibility'], df['entropy_change'], c=colors, alpha=0.7, s=80)
    ax.set_xlabel('Solvent Accessibility (estimated)', fontsize=11)
    ax.set_ylabel('Entropy Change', fontsize=11)
    ax.set_title('Entropy Change vs Solvent Accessibility', fontsize=12, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # 2. Entropy vs pLDDT
    ax = axes[0, 1]
    ax.scatter(df['plddt'], df['entropy_change'], c=colors, alpha=0.7, s=80)
    ax.set_xlabel('AlphaFold pLDDT Score', fontsize=11)
    ax.set_ylabel('Entropy Change', fontsize=11)
    ax.set_title('Entropy Change vs Structure Confidence', fontsize=12, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # 3. Secondary structure distribution
    ax = axes[1, 0]
    ss_counts = df.groupby(['immunodominant', 'secondary_structure']).size().unstack(fill_value=0)
    ss_counts.plot(kind='bar', ax=ax, color=['#66b3ff', '#99ff99', '#ffcc99'])
    ax.set_xlabel('Immunodominant', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Secondary Structure Distribution', fontsize=12, fontweight='bold')
    ax.set_xticklabels(['Silent', 'Immunodominant'], rotation=0)
    ax.legend(title='Structure')

    # 4. Box plot of entropy by structure
    ax = axes[1, 1]
    ss_types = df['secondary_structure'].unique()
    data_by_ss = [df[df['secondary_structure'] == ss]['entropy_change'].values for ss in ss_types]
    bp = ax.boxplot(data_by_ss, labels=ss_types, patch_artist=True)
    colors_box = ['#66b3ff', '#99ff99', '#ffcc99']
    for patch, color in zip(bp['boxes'], colors_box[:len(ss_types)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel('Secondary Structure', fontsize=11)
    ax.set_ylabel('Entropy Change', fontsize=11)
    ax.set_title('Entropy Change by Secondary Structure', fontsize=12, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'structural_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save site data
    df.to_csv(output_dir / 'site_structural_analysis.csv', index=False)

    return correlations


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("ALPHAFOLD2 STRUCTURE MAPPING")
    print("Mapping entropy change predictions onto 3D protein structures")
    print("=" * 80)

    output_dir = get_output_dir()
    pdb_dir = get_pdb_dir()
    print(f"\nOutput directory: {output_dir}")
    print(f"PDB directory: {pdb_dir}")

    # Load epitope data
    print("\n[1] Loading epitope data...")
    epitopes = load_epitope_data()
    print(f"  Loaded {len(epitopes)} epitopes")

    # Download and analyze structures
    print("\n[2] Downloading AlphaFold2 structures...")
    all_analyses = []

    for gene, uniprot_id in RA_AUTOANTIGENS.items():
        print(f"\n  Processing {gene} ({uniprot_id})...")

        analysis = analyze_protein_structure(gene, uniprot_id, epitopes, pdb_dir)

        if analysis:
            all_analyses.append(analysis)
            print(f"    Structure: {analysis['total_residues']} residues, {analysis['total_arginines']} arginines")
            print(f"    Mapped sites: {len(analysis['analyzed_sites'])}")

            # Generate visualization scripts
            if analysis['analyzed_sites']:
                pymol_script = generate_pymol_script(analysis, output_dir)
                print(f"    PyMOL script: {pymol_script.name}")

                html_viewer = generate_html_viewer(analysis, output_dir)
                print(f"    HTML viewer: {html_viewer.name}")
        else:
            print(f"    No structure or epitopes found")

    # Structural correlation analysis
    print("\n[3] Analyzing structural correlations...")
    correlations = analyze_structural_correlations(all_analyses, output_dir)

    if correlations:
        print("\n  Correlation Results:")
        if 'entropy_vs_accessibility' in correlations:
            c = correlations['entropy_vs_accessibility']
            print(f"    Entropy vs Accessibility: r={c['r']:.3f}, p={c['p']:.4f}")
        if 'entropy_vs_plddt' in correlations:
            c = correlations['entropy_vs_plddt']
            print(f"    Entropy vs pLDDT: r={c['r']:.3f}, p={c['p']:.4f}")
        if 'mean_accessibility' in correlations:
            c = correlations['mean_accessibility']
            print(f"    Mean Accessibility - Immunodominant: {c['immunodominant']:.3f}, Silent: {c['silent']:.3f}")

    # Save summary
    print("\n[4] Saving summary...")
    summary = {
        'proteins_analyzed': len(all_analyses),
        'total_sites_mapped': sum(len(a['analyzed_sites']) for a in all_analyses if a),
        'correlations': correlations,
        'proteins': [
            {
                'gene': a['gene_name'],
                'uniprot': a['uniprot_id'],
                'residues': a['total_residues'],
                'arginines': a['total_arginines'],
                'sites_mapped': len(a['analyzed_sites'])
            }
            for a in all_analyses if a
        ]
    }

    # Convert numpy types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(i) for i in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    summary_path = output_dir / 'structure_mapping_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(convert_types(summary), f, indent=2)
    print(f"  Saved: {summary_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("STRUCTURE MAPPING COMPLETE")
    print("=" * 80)

    print(f"\n  Proteins analyzed: {len(all_analyses)}")
    print(f"  Total sites mapped: {summary['total_sites_mapped']}")

    print("\n  Generated files:")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            print(f"    {f.name}")

    print(f"\n  To view structures:")
    print(f"    - Open HTML files in browser for interactive 3D view")
    print(f"    - Run .pml files in PyMOL for publication-quality images")

    return all_analyses


if __name__ == '__main__':
    main()
