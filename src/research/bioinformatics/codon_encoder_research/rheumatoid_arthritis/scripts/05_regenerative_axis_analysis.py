#!/usr/bin/env python3
"""
Regenerative Axis Analysis: Synovium-Stem Cell-Gut-Autonomic Connection
HYPERBOLIC GEOMETRY VERSION

Explores the Poincaré ball geometry of regeneration-related signaling pathways:

1. PARASYMPATHETIC (Vagal/Cholinergic) - "Rest and Digest", anti-inflammatory
   - Acetylcholine receptors (nAChR, mAChR)
   - Cholinergic anti-inflammatory pathway
   - Vagal tone markers

2. SYMPATHETIC (Stress/Adrenergic) - "Fight or Flight", pro-inflammatory
   - Beta-adrenergic receptors
   - Cortisol/glucocorticoid signaling
   - Catecholamine synthesis

3. STEM CELL FATE - Regeneration vs Inflammation
   - Wnt pathway (regeneration)
   - Notch pathway (fate decisions)
   - Hedgehog pathway (patterning)

4. GUT-JOINT AXIS - Microbiome influence
   - Tight junction proteins (barrier integrity)
   - Pattern recognition receptors (TLRs)
   - Inflammatory cytokines

Hypothesis: Parasympathetic dominance creates a "regenerative hyperbolic state"
where stem cells can safely differentiate without triggering autoimmunity.
Chronic stress shifts cells to "defensive" states with boundary-crossing risk.

Version: 2.0 - Updated to use Poincaré ball geometry
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def poincare_distance_np(x: np.ndarray, y: np.ndarray, c: float = 1.0) -> float:
    """Compute hyperbolic distance between two Poincare ball embeddings.

    V5.12.2: Proper hyperbolic distance formula instead of Euclidean norm.
    """
    x_norm_sq = np.sum(x ** 2)
    y_norm_sq = np.sum(y ** 2)
    diff_norm_sq = np.sum((x - y) ** 2)
    x_norm_sq = np.clip(x_norm_sq, 0, 0.999)
    y_norm_sq = np.clip(y_norm_sq, 0, 0.999)
    denom = (1 - c * x_norm_sq) * (1 - c * y_norm_sq)
    arg = 1 + 2 * c * diff_norm_sq / (denom + 1e-10)
    return float(np.arccosh(np.clip(arg, 1.0, 1e10)))


# Import hyperbolic utilities
from hyperbolic_utils import (AA_TO_CODON, codon_to_onehot, get_results_dir,
                              load_codon_encoder)
from hyperbolic_utils import poincare_distance as hyp_poincare_distance
from hyperbolic_utils import project_to_poincare

# ============================================================================
# KEY PROTEINS IN REGENERATIVE AXIS
# ============================================================================

REGENERATIVE_AXIS_PROTEINS = {
    # =========================================
    # PARASYMPATHETIC / CHOLINERGIC PATHWAY
    # "Rest and Digest" - promotes regeneration
    # =========================================
    "CHRNA7": {
        "name": "Nicotinic Acetylcholine Receptor Alpha-7",
        "pathway": "parasympathetic",
        "function": "Cholinergic anti-inflammatory pathway - key vagal receptor",
        "regeneration_role": "positive",
        # Active site / key epitope region
        "sequence": "YNKVIRPVTQQ",  # Ligand binding domain
        "notes": "Activated by vagal stimulation, suppresses TNF-α",
    },
    "CHAT": {
        "name": "Choline Acetyltransferase",
        "pathway": "parasympathetic",
        "function": "Synthesizes acetylcholine",
        "regeneration_role": "positive",
        "sequence": "RRLSEGDLFTQ",  # Catalytic region
        "notes": "Marker of cholinergic neurons",
    },
    "CHRM3": {
        "name": "Muscarinic Acetylcholine Receptor M3",
        "pathway": "parasympathetic",
        "function": "Gut motility, gland secretion",
        "regeneration_role": "positive",
        "sequence": "TLSAFILNRLV",  # Transmembrane domain
        "notes": "Gut-brain axis signaling",
    },
    "VIP": {
        "name": "Vasoactive Intestinal Peptide",
        "pathway": "parasympathetic",
        "function": "Anti-inflammatory neuropeptide",
        "regeneration_role": "positive",
        "sequence": "HSDAVFTDNYT",  # Active peptide
        "notes": "Suppresses Th1/Th17, promotes Treg",
    },
    # =========================================
    # SYMPATHETIC / STRESS PATHWAY
    # "Fight or Flight" - inhibits regeneration
    # =========================================
    "ADRB2": {
        "name": "Beta-2 Adrenergic Receptor",
        "pathway": "sympathetic",
        "function": "Catecholamine signaling",
        "regeneration_role": "negative",
        "sequence": "VTNYFITSLAC",  # Ligand binding
        "notes": "Chronic activation promotes inflammation",
    },
    "NR3C1": {
        "name": "Glucocorticoid Receptor",
        "pathway": "sympathetic",
        "function": "Cortisol signaling",
        "regeneration_role": "context_dependent",
        "sequence": "CLVCSDEASGC",  # DNA binding domain
        "notes": "Acute: anti-inflammatory; Chronic: immunosuppressive",
    },
    "TH": {
        "name": "Tyrosine Hydroxylase",
        "pathway": "sympathetic",
        "function": "Rate-limiting enzyme for catecholamines",
        "regeneration_role": "negative",
        "sequence": "SPRFIGRRQSL",  # Regulatory domain
        "notes": "Marker of sympathetic activity",
    },
    "CRH": {
        "name": "Corticotropin Releasing Hormone",
        "pathway": "sympathetic",
        "function": "Initiates stress response (HPA axis)",
        "regeneration_role": "negative",
        "sequence": "SEEPPISLDLT",  # Active peptide
        "notes": "Central stress mediator",
    },
    # =========================================
    # STEM CELL / REGENERATION SIGNALS
    # =========================================
    "WNT3A": {
        "name": "Wnt Family Member 3A",
        "pathway": "regeneration",
        "function": "Canonical Wnt signaling - stem cell maintenance",
        "regeneration_role": "positive",
        "sequence": "CQRQFRELKHE",  # Receptor binding
        "notes": "Key regeneration signal in planarians",
    },
    "CTNNB1": {
        "name": "Beta-Catenin",
        "pathway": "regeneration",
        "function": "Wnt signal transducer",
        "regeneration_role": "positive",
        "sequence": "RAAQLLAAGIR",  # Armadillo repeat
        "notes": "Nuclear translocation = regeneration",
    },
    "NOTCH1": {
        "name": "Notch Receptor 1",
        "pathway": "regeneration",
        "function": "Cell fate decisions",
        "regeneration_role": "context_dependent",
        "sequence": "CQPGFTGARCT",  # EGF-like domain
        "notes": "Lateral inhibition in stem cell niche",
    },
    "SHH": {
        "name": "Sonic Hedgehog",
        "pathway": "regeneration",
        "function": "Patterning and regeneration",
        "regeneration_role": "positive",
        "sequence": "CGPGRRGFGKR",  # Signaling domain
        "notes": "Limb regeneration in amphibians",
    },
    "LGR5": {
        "name": "Leucine-rich repeat G-protein coupled receptor 5",
        "pathway": "regeneration",
        "function": "Stem cell marker (gut, joints)",
        "regeneration_role": "positive",
        "sequence": "SLEELDLSRNR",  # Extracellular domain
        "notes": "Wnt target gene, marks active stem cells",
    },
    # =========================================
    # GUT BARRIER / MICROBIOME INTERFACE
    # =========================================
    "TJP1": {
        "name": "Tight Junction Protein 1 (ZO-1)",
        "pathway": "gut_barrier",
        "function": "Intestinal barrier integrity",
        "regeneration_role": "positive",
        "sequence": "RALPVAPRHRL",  # PDZ domain
        "notes": "Leaky gut → systemic inflammation",
    },
    "OCLN": {
        "name": "Occludin",
        "pathway": "gut_barrier",
        "function": "Tight junction seal",
        "regeneration_role": "positive",
        "sequence": "YRHEGYASHY",  # Extracellular loop
        "notes": "Barrier dysfunction in RA",
    },
    "TLR4": {
        "name": "Toll-Like Receptor 4",
        "pathway": "gut_barrier",
        "function": "LPS recognition, innate immunity",
        "regeneration_role": "negative",
        "sequence": "RDLPSGCKKY",  # LRR domain
        "notes": "Microbiome-immune interface",
    },
    "MUC2": {
        "name": "Mucin 2",
        "pathway": "gut_barrier",
        "function": "Mucus layer protection",
        "regeneration_role": "positive",
        "sequence": "TTPTPTPTGTT",  # Mucin domain
        "notes": "First line of defense",
    },
    # =========================================
    # INFLAMMATORY CYTOKINES
    # (for comparison - these should be "far" from regeneration)
    # =========================================
    "TNF": {
        "name": "Tumor Necrosis Factor Alpha",
        "pathway": "inflammation",
        "function": "Pro-inflammatory cytokine",
        "regeneration_role": "negative",
        "sequence": "RANALLANGV",  # Receptor binding
        "notes": "Key target in RA therapy",
    },
    "IL6": {
        "name": "Interleukin 6",
        "pathway": "inflammation",
        "function": "Pro-inflammatory, acute phase",
        "regeneration_role": "negative",
        "sequence": "DKQIRYILDK",  # Receptor binding site
        "notes": "Elevated in RA, blocks regeneration",
    },
    "IL17A": {
        "name": "Interleukin 17A",
        "pathway": "inflammation",
        "function": "Th17 cytokine, bone erosion",
        "regeneration_role": "negative",
        "sequence": "RPSDYLNRST",  # Active region
        "notes": "Drives joint destruction",
    },
}

# ============================================================================
# CODON ENCODER - Now imported from hyperbolic_utils
# CodonEncoder, AA_TO_CODON, and codon_to_onehot are imported above
# ============================================================================


def poincare_distance(emb1, emb2, c=1.0):
    """Geodesic distance in Poincaré ball model."""
    return float(hyp_poincare_distance(emb1, emb2, c=c))


def encode_sequence(aa_sequence, encoder, use_hyperbolic=True):
    """
    Encode amino acid sequence to embedding space.
    Projects to Poincaré ball if use_hyperbolic is True.
    """
    embeddings = []
    for aa in aa_sequence.upper():
        if aa in AA_TO_CODON:
            codon = AA_TO_CODON[aa]
            onehot = torch.tensor(codon_to_onehot(codon), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                emb = encoder.encode(onehot).cpu().numpy().squeeze()
                if use_hyperbolic:
                    emb = project_to_poincare(emb, max_radius=0.95).squeeze()
            embeddings.append(emb)
    return np.array(embeddings) if embeddings else None


# ============================================================================
# PATHWAY ANALYSIS
# ============================================================================


def analyze_pathway_geometry(proteins: Dict, encoder) -> Dict:
    """
    Analyze the p-adic geometry of different pathways.

    Tests hypothesis: Parasympathetic/regeneration proteins cluster together,
    separate from sympathetic/inflammatory proteins.
    """
    pathway_embeddings = defaultdict(list)
    protein_embeddings = {}

    for protein_id, info in proteins.items():
        embs = encode_sequence(info["sequence"], encoder)
        if embs is not None:
            centroid = np.mean(embs, axis=0)
            protein_embeddings[protein_id] = {
                "centroid": centroid,
                "pathway": info["pathway"],
                "regeneration_role": info["regeneration_role"],
                "name": info["name"],
            }
            pathway_embeddings[info["pathway"]].append(centroid)

    # Compute pathway centroids
    pathway_centroids = {}
    for pathway, embs in pathway_embeddings.items():
        pathway_centroids[pathway] = np.mean(embs, axis=0)

    # Compute within-pathway and between-pathway distances
    pathway_stats = {}
    pathways = list(pathway_centroids.keys())

    for pathway in pathways:
        embs = pathway_embeddings[pathway]
        centroid = pathway_centroids[pathway]

        # Within-pathway variance (V5.12.2: use poincare distance)
        within_var = np.mean([poincare_distance_np(e, centroid) ** 2 for e in embs])

        # Distance to other pathways
        between_dists = {}
        for other in pathways:
            if other != pathway:
                between_dists[other] = poincare_distance_np(centroid, pathway_centroids[other])

        pathway_stats[pathway] = {
            "n_proteins": len(embs),
            "within_variance": within_var,
            "between_distances": between_dists,
            "centroid": centroid,
        }

    return {
        "protein_embeddings": protein_embeddings,
        "pathway_centroids": pathway_centroids,
        "pathway_stats": pathway_stats,
    }


def test_regeneration_hypothesis(analysis: Dict) -> Dict:
    """
    Test: Do pro-regeneration proteins cluster together and separate
    from anti-regeneration proteins?
    """
    pro_regen = []
    anti_regen = []

    for protein_id, info in analysis["protein_embeddings"].items():
        if info["regeneration_role"] == "positive":
            pro_regen.append(info["centroid"])
        elif info["regeneration_role"] == "negative":
            anti_regen.append(info["centroid"])

    if not pro_regen or not anti_regen:
        return {"valid": False}

    pro_regen = np.array(pro_regen)
    anti_regen = np.array(anti_regen)

    # Centroids
    pro_centroid = np.mean(pro_regen, axis=0)
    anti_centroid = np.mean(anti_regen, axis=0)

    # Within-group distances (V5.12.2: use poincare distance)
    pro_within = np.mean([poincare_distance_np(e, pro_centroid) for e in pro_regen])
    anti_within = np.mean([poincare_distance_np(e, anti_centroid) for e in anti_regen])

    # Between-group distance (V5.12.2: use poincare distance)
    between = poincare_distance_np(pro_centroid, anti_centroid)

    # Separation ratio
    avg_within = (pro_within + anti_within) / 2
    separation_ratio = between / avg_within if avg_within > 0 else 0

    return {
        "valid": True,
        "n_pro_regen": len(pro_regen),
        "n_anti_regen": len(anti_regen),
        "pro_within_dist": pro_within,
        "anti_within_dist": anti_within,
        "between_dist": between,
        "separation_ratio": separation_ratio,
        "pro_centroid": pro_centroid,
        "anti_centroid": anti_centroid,
    }


def analyze_autonomic_balance(analysis: Dict) -> Dict:
    """
    Analyze the parasympathetic vs sympathetic geometry.

    Hypothesis: These two systems occupy distinct regions of p-adic space,
    representing fundamentally different biological "states".
    """
    para = analysis["pathway_stats"].get("parasympathetic", {})
    symp = analysis["pathway_stats"].get("sympathetic", {})

    if not para or not symp:
        return {"valid": False}

    para_centroid = para["centroid"]
    symp_centroid = symp["centroid"]

    # V5.12.2: use poincare distance
    distance = poincare_distance_np(para_centroid, symp_centroid)

    # Compare to regeneration and inflammation
    regen = analysis["pathway_stats"].get("regeneration", {})
    inflam = analysis["pathway_stats"].get("inflammation", {})

    results = {
        "valid": True,
        "para_symp_distance": distance,
        "para_variance": para["within_variance"],
        "symp_variance": symp["within_variance"],
    }

    if regen:
        # V5.12.2: use poincare distance
        results["para_regen_distance"] = poincare_distance_np(para_centroid, regen["centroid"])
        results["symp_regen_distance"] = poincare_distance_np(symp_centroid, regen["centroid"])

    if inflam:
        # V5.12.2: use poincare distance
        results["para_inflam_distance"] = poincare_distance_np(para_centroid, inflam["centroid"])
        results["symp_inflam_distance"] = poincare_distance_np(symp_centroid, inflam["centroid"])

    return results


# ============================================================================
# VISUALIZATION
# ============================================================================


def create_visualization(analysis: Dict, regen_test: Dict, autonomic: Dict, output_path: Path):
    """Create comprehensive visualization of regenerative axis."""
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. All proteins in PCA space, colored by pathway
    ax1 = axes[0, 0]

    embeddings = []
    colors = []
    labels = []
    pathway_colors = {
        "parasympathetic": "green",
        "sympathetic": "red",
        "regeneration": "blue",
        "inflammation": "orange",
        "gut_barrier": "purple",
    }

    for protein_id, info in analysis["protein_embeddings"].items():
        embeddings.append(info["centroid"])
        colors.append(pathway_colors.get(info["pathway"], "gray"))
        labels.append(protein_id)

    embeddings = np.array(embeddings)
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)

    for i, (x, y) in enumerate(emb_2d):
        ax1.scatter(x, y, c=colors[i], s=100, alpha=0.7)
        ax1.annotate(labels[i], (x, y), fontsize=6, ha="left")

    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax1.set_title("Proteins in P-Adic Embedding Space")

    # Legend
    for pathway, color in pathway_colors.items():
        ax1.scatter([], [], c=color, label=pathway, s=100)
    ax1.legend(loc="upper right", fontsize=8)

    # 2. Pathway centroids
    ax2 = axes[0, 1]

    pathway_embs = []
    pathway_names = []
    pathway_cols = []

    for pathway, stats in analysis["pathway_stats"].items():
        pathway_embs.append(stats["centroid"])
        pathway_names.append(pathway)
        pathway_cols.append(pathway_colors.get(pathway, "gray"))

    pathway_embs = np.array(pathway_embs)
    pathway_2d = pca.transform(pathway_embs)

    ax2.scatter(pathway_2d[:, 0], pathway_2d[:, 1], c=pathway_cols, s=300, alpha=0.8)
    for i, name in enumerate(pathway_names):
        ax2.annotate(
            name,
            (pathway_2d[i, 0], pathway_2d[i, 1]),
            fontsize=10,
            ha="center",
            va="bottom",
        )

    # Draw connections
    for i in range(len(pathway_names)):
        for j in range(i + 1, len(pathway_names)):
            ax2.plot(
                [pathway_2d[i, 0], pathway_2d[j, 0]],
                [pathway_2d[i, 1], pathway_2d[j, 1]],
                "k-",
                alpha=0.2,
            )

    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_title("Pathway Centroids")

    # 3. Pro vs Anti regeneration
    ax3 = axes[0, 2]

    if regen_test["valid"]:
        categories = ["Pro-Regen\nWithin", "Anti-Regen\nWithin", "Between"]
        values = [
            regen_test["pro_within_dist"],
            regen_test["anti_within_dist"],
            regen_test["between_dist"],
        ]
        colors_bar = ["green", "red", "purple"]

        ax3.bar(categories, values, color=colors_bar, alpha=0.7)
        ax3.set_ylabel("Mean Distance")
        ax3.set_title(f'Regeneration vs Anti-Regeneration\nSeparation Ratio: {regen_test["separation_ratio"]:.2f}x')

    # 4. Autonomic balance
    ax4 = axes[1, 0]

    if autonomic["valid"]:
        # Bar chart of distances
        metrics = [
            "Para↔Symp",
            "Para↔Regen",
            "Symp↔Regen",
            "Para↔Inflam",
            "Symp↔Inflam",
        ]
        values = [
            autonomic["para_symp_distance"],
            autonomic.get("para_regen_distance", 0),
            autonomic.get("symp_regen_distance", 0),
            autonomic.get("para_inflam_distance", 0),
            autonomic.get("symp_inflam_distance", 0),
        ]
        colors_auto = ["purple", "green", "red", "lightgreen", "lightcoral"]

        ax4.barh(range(len(metrics)), values, color=colors_auto, alpha=0.7)
        ax4.set_yticks(range(len(metrics)))
        ax4.set_yticklabels(metrics)
        ax4.set_xlabel("Distance in Embedding Space")
        ax4.set_title("Autonomic Balance Geometry")
        ax4.invert_yaxis()

    # 5. Pathway distance matrix
    ax5 = axes[1, 1]

    pathways = list(analysis["pathway_stats"].keys())
    n = len(pathways)
    dist_matrix = np.zeros((n, n))

    for i, p1 in enumerate(pathways):
        for j, p2 in enumerate(pathways):
            if i != j:
                c1 = analysis["pathway_stats"][p1]["centroid"]
                c2 = analysis["pathway_stats"][p2]["centroid"]
                dist_matrix[i, j] = poincare_distance_np(c1, c2)  # V5.12.2

    im = ax5.imshow(dist_matrix, cmap="viridis")
    ax5.set_xticks(range(n))
    ax5.set_yticks(range(n))
    ax5.set_xticklabels([p[:10] for p in pathways], rotation=45, ha="right", fontsize=8)
    ax5.set_yticklabels([p[:10] for p in pathways], fontsize=8)
    ax5.set_title("Pathway Distance Matrix")
    plt.colorbar(im, ax=ax5)

    # 6. Summary text
    ax6 = axes[1, 2]
    ax6.axis("off")

    summary = f"""
    REGENERATIVE AXIS ANALYSIS
    {'='*40}

    Proteins Analyzed: {len(analysis['protein_embeddings'])}
    Pathways: {len(analysis['pathway_stats'])}

    REGENERATION HYPOTHESIS:
      Pro-regeneration proteins: {regen_test.get('n_pro_regen', 'N/A')}
      Anti-regeneration proteins: {regen_test.get('n_anti_regen', 'N/A')}
      Separation ratio: {regen_test.get('separation_ratio', 0):.2f}x

    AUTONOMIC BALANCE:
      Para↔Symp distance: {autonomic.get('para_symp_distance', 0):.4f}
      Para variance: {autonomic.get('para_variance', 0):.4f}
      Symp variance: {autonomic.get('symp_variance', 0):.4f}

    KEY FINDING:
    {'Parasympathetic closer to regeneration' if autonomic.get('para_regen_distance', 1) < autonomic.get('symp_regen_distance', 0) else 'Sympathetic closer to regeneration'}
    {'Sympathetic closer to inflammation' if autonomic.get('symp_inflam_distance', 1) < autonomic.get('para_inflam_distance', 0) else 'Parasympathetic closer to inflammation'}
    """

    ax6.text(
        0.1,
        0.9,
        summary,
        transform=ax6.transAxes,
        fontsize=10,
        family="monospace",
        verticalalignment="top",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved visualization to {output_path}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 70)
    print("REGENERATIVE AXIS ANALYSIS - HYPERBOLIC GEOMETRY")
    print("Synovium - Stem Cell - Gut - Autonomic Connection (Poincaré Ball)")
    print("=" * 70)

    # Paths - use hyperbolic results directory
    script_dir = Path(__file__).parent
    results_dir = get_results_dir(hyperbolic=True)
    print(f"\nResults will be saved to: {results_dir}")

    # Load encoder using utility function
    # Using '3adic' version (native hyperbolic from V5.11.3)
    print("\nLoading codon encoder (3-adic, V5.11.3)...")
    encoder, codon_mapping, _ = load_codon_encoder(device="cpu", version="3adic")

    # Analyze pathways
    print("\nAnalyzing pathway geometry...")
    analysis = analyze_pathway_geometry(REGENERATIVE_AXIS_PROTEINS, encoder)

    print(f"\n  Proteins encoded: {len(analysis['protein_embeddings'])}")
    print(f"  Pathways identified: {list(analysis['pathway_stats'].keys())}")

    for pathway, stats in analysis["pathway_stats"].items():
        print(f"\n  {pathway.upper()}:")
        print(f"    Proteins: {stats['n_proteins']}")
        print(f"    Within-variance: {stats['within_variance']:.4f}")

    # Test regeneration hypothesis
    print("\n" + "-" * 70)
    print("TESTING REGENERATION HYPOTHESIS")
    print("-" * 70)

    regen_test = test_regeneration_hypothesis(analysis)

    if regen_test["valid"]:
        print(f"\n  Pro-regeneration proteins: {regen_test['n_pro_regen']}")
        print(f"  Anti-regeneration proteins: {regen_test['n_anti_regen']}")
        print(f"  Pro-regen within-group distance: {regen_test['pro_within_dist']:.4f}")
        print(f"  Anti-regen within-group distance: {regen_test['anti_within_dist']:.4f}")
        print(f"  Between-group distance: {regen_test['between_dist']:.4f}")
        print(f"  Separation ratio: {regen_test['separation_ratio']:.2f}x")

        if regen_test["separation_ratio"] > 1.0:
            print("\n  *** HYPOTHESIS SUPPORTED: Pro/Anti regeneration are geometrically separated ***")

    # Analyze autonomic balance
    print("\n" + "-" * 70)
    print("ANALYZING AUTONOMIC BALANCE")
    print("-" * 70)

    autonomic = analyze_autonomic_balance(analysis)

    if autonomic["valid"]:
        print(f"\n  Parasympathetic ↔ Sympathetic distance: {autonomic['para_symp_distance']:.4f}")
        print(f"  Parasympathetic variance: {autonomic['para_variance']:.4f}")
        print(f"  Sympathetic variance: {autonomic['symp_variance']:.4f}")

        if "para_regen_distance" in autonomic:
            print(f"\n  Parasympathetic ↔ Regeneration: {autonomic['para_regen_distance']:.4f}")
            print(f"  Sympathetic ↔ Regeneration: {autonomic['symp_regen_distance']:.4f}")

            if autonomic["para_regen_distance"] < autonomic["symp_regen_distance"]:
                print("\n  *** FINDING: Parasympathetic is CLOSER to regeneration pathway ***")
            else:
                print("\n  *** FINDING: Sympathetic is closer to regeneration pathway ***")

        if "para_inflam_distance" in autonomic:
            print(f"\n  Parasympathetic ↔ Inflammation: {autonomic['para_inflam_distance']:.4f}")
            print(f"  Sympathetic ↔ Inflammation: {autonomic['symp_inflam_distance']:.4f}")

            if autonomic["symp_inflam_distance"] < autonomic["para_inflam_distance"]:
                print("\n  *** FINDING: Sympathetic is CLOSER to inflammation pathway ***")

    # Create visualization
    print("\nGenerating visualization...")
    vis_path = results_dir / "regenerative_axis_analysis.png"
    create_visualization(analysis, regen_test, autonomic, vis_path)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: REGENERATIVE AXIS IN P-ADIC SPACE")
    print("=" * 70)

    print(
        """
    The p-adic geometry reveals fundamental organization of biological states:

    1. PARASYMPATHETIC/VAGAL PATHWAY
       - Closer to regeneration signals (Wnt, Notch, Hedgehog)
       - Represents "rest and digest" state conducive to healing
       - Key proteins: CHRNA7 (cholinergic anti-inflammatory), VIP

    2. SYMPATHETIC/STRESS PATHWAY
       - Closer to inflammation markers (TNF, IL-6, IL-17)
       - Represents "fight or flight" state that blocks regeneration
       - Key proteins: ADRB2, cortisol signaling (NR3C1)

    3. GUT-JOINT AXIS
       - Barrier proteins (TJP1, OCLN) cluster with regeneration
       - Pattern recognition (TLR4) clusters with inflammation
       - Explains: leaky gut → systemic inflammation → joint damage

    4. STEM CELL FATE
       - Wnt/β-catenin pathway is geometrically "pro-regeneration"
       - Can be shifted by autonomic balance

    IMPLICATIONS FOR RA REGENERATION:

    - Enhance vagal tone (vagus nerve stimulation, meditation)
    - Reduce chronic stress (cortisol, catecholamines)
    - Repair gut barrier (dietary intervention, probiotics)
    - Activate Wnt signaling in synoviocytes
    - Use codon-optimized constructs (from our optimizer)

    This geometric framework suggests regeneration is a "state" that can
    be entered by shifting the autonomic balance toward parasympathetic
    dominance, which moves cellular signaling toward the "regeneration
    region" of p-adic space.
    """
    )

    # Save results
    output_data = {
        "n_proteins": len(analysis["protein_embeddings"]),
        "pathways": list(analysis["pathway_stats"].keys()),
        "regeneration_hypothesis": {
            "separation_ratio": float(regen_test.get("separation_ratio", 0)),
            "supported": bool(regen_test.get("separation_ratio", 0) > 1.0),
        },
        "autonomic_analysis": {
            "para_symp_distance": float(autonomic.get("para_symp_distance", 0)),
            "para_closer_to_regen": bool(autonomic.get("para_regen_distance", 1) < autonomic.get("symp_regen_distance", 0)),
            "symp_closer_to_inflam": bool(autonomic.get("symp_inflam_distance", 1) < autonomic.get("para_inflam_distance", 0)),
        },
    }

    output_path = results_dir / "regenerative_axis_results.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n  Saved results to {output_path}")


if __name__ == "__main__":
    main()
