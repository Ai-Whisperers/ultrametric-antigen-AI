# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Hemolysis Predictor for Antimicrobial Peptides.

Predicts hemolytic activity (HC50) of peptides based on sequence features.
Hemolysis is the lysis of red blood cells and is a major toxicity concern
for therapeutic peptides.

Usage:
    from shared.hemolysis_predictor import HemolysisPredictor

    predictor = HemolysisPredictor()
    result = predictor.predict("GIGKFLHSAKKFGKAFVGEIMNS")
    print(f"HC50: {result['hc50_predicted']:.1f} μM")
    print(f"Hemolytic risk: {result['risk_category']}")
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .constants import CHARGES, HYDROPHOBICITY, VOLUMES, AMINO_ACIDS
from .peptide_utils import compute_peptide_properties, compute_ml_features

# Try to import sklearn
try:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# Curated hemolysis data from literature
# Format: (name, sequence, HC50_uM, is_hemolytic)
# HC50 < 50 μM is considered hemolytic
# Sources: DBAASP, HemoPI database, primary literature
CURATED_HEMOLYSIS_DATA = [
    # Highly hemolytic peptides (HC50 < 50 μM)
    ("Melittin", "GIGAVLKVLTTGLPALISWIKRKRQQ", 1.5, True),
    ("Mastoparan", "INLKALAALAKKIL", 15.0, True),
    ("Crabrolin", "FLPLILRKIVTAL", 20.0, True),
    ("Pardaxin", "GFFALIPKIISSPLFKTLLSAVGSALSSSGGQE", 8.0, True),
    ("Polybia-MP1", "IDWKKLLDAAKQIL", 25.0, True),
    ("Delta-lysin", "MAQDIISTIGDLVKWIIDTVNKFTKK", 5.0, True),
    ("Gramicidin S", "VOLFPVOLFP", 12.0, True),
    ("Alamethicin", "APAAAAQAVAGLAPVAAEQ", 18.0, True),

    # Moderately hemolytic (50-200 μM)
    ("Magainin 2", "GIGKFLHSAKKFGKAFVGEIMNS", 100.0, False),
    ("Indolicidin", "ILPWKWPWWPWRR", 80.0, False),
    ("LL-37", "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES", 150.0, False),
    ("Temporin A", "FLPLIGRVLSGIL", 60.0, False),
    ("Aurein 1.2", "GLFDIIKKIAESF", 70.0, False),

    # Low/No hemolysis (HC50 > 200 μM)
    ("Pexiganan (MSI-78)", "GIGKFLKKAKKFGKAFVKILKK", 250.0, False),
    ("Cecropin A", "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK", 400.0, False),
    ("Cecropin B", "KWKVFKKIEKMGRNIRNGIVKAGPAIAVLGEAKAL", 350.0, False),
    ("Buforin II", "TRSSRAGLQFPVGRVHRLLRK", 500.0, False),
    ("PR-39", "RRRPRPPYLPRPRPPPFFPPRLPPRIPPGFPPRFPPRFP", 600.0, False),
    ("Defensin HNP-1", "ACYCRIPACIAGERRYGTCIYQGRLWAFCC", 300.0, False),
    ("Nisin", "ITSISLCTPGCKTGALMGCNMKTATCHCSIHVSK", 400.0, False),
    ("Pleurocidin", "GWGSFFKKAAHVGKHVGKAALTHYL", 280.0, False),
    ("Drosocin", "GKPRPYSPRPTSHPRPIRV", 800.0, False),
    ("Pyrrhocoricin", "VDKGSYLPRPTPPRPIYNRN", 900.0, False),
    ("Apidaecin IA", "GNNRPVYIPQPRPPHPRI", 1000.0, False),

    # Additional data points for model training
    ("Brevinin-1", "FLPVLAGIAAKVVPALFCKITKKC", 45.0, True),
    ("Esculentin-1", "GIFSKLGRKKIKNLLISGLKNVGKEVGMDVVRTGIDIAGCKIKGEC", 180.0, False),
    ("Dermaseptin S1", "ALWKTMLKKLGTMALHAGKAALGAAADTISQGTQ", 120.0, False),
    ("Ranalexin", "FLGGLIKIVPAMICAVTKKC", 55.0, False),
    ("BMAP-27", "GRFKRFRKKFKKLFKKLSPVIPLLHL", 90.0, False),
    ("BMAP-28", "GGLRSLGRKILRAWKKYGPIIVPIIRI", 110.0, False),
    ("Protegrin-1", "RGGRLCYCRRRFCVCVGR", 35.0, True),
    ("Tachyplesin I", "KWCFRVCYRGICYRRCR", 40.0, True),
    ("Polyphemusin I", "RRWCFRVCYRGFCYRKCR", 38.0, True),
    ("Gomesin", "QCRRLCYKQRCVTYCRGR", 65.0, False),
    ("Arenicin-1", "RWCVYAYVRVRGVLVRYRRCW", 30.0, True),
    ("Lactoferricin B", "FKCRRWQWRMKKLGAPSITCVRRAF", 85.0, False),
    ("Histatin 5", "DSHAKRHHGYKRKFHEKHHSHRGY", 500.0, False),
    ("Cathelicidin BF", "KRFKKFFKKLKNSVKKRAKKFFKKPRVIGVSIPF", 75.0, False),
    ("WLBU2", "RRWVRRVRRWVRRVVRVVRRWVRR", 95.0, False),
    ("Omiganan", "ILRWPWWPWRRK", 100.0, False),
]


class HemolysisPredictor:
    """Predict hemolytic activity of peptides.

    Uses machine learning to predict HC50 (hemolytic concentration 50%)
    and classify peptides as hemolytic or non-hemolytic.
    """

    def __init__(self):
        """Initialize the hemolysis predictor."""
        self.regressor = None
        self.classifier = None
        self.scaler = None
        self._is_trained = False

        # Train on initialization
        self._train_models()

    def _compute_hemolysis_features(self, sequence: str) -> np.ndarray:
        """Compute features relevant for hemolysis prediction.

        Features include:
        - Basic properties (length, charge, hydrophobicity)
        - Amphipathicity-related features
        - Hydrophobic moment proxy
        - Tryptophan content (membrane insertion)
        - Proline content (helix disruption)

        Args:
            sequence: Amino acid sequence

        Returns:
            Feature array
        """
        props = compute_peptide_properties(sequence)
        n = len(sequence) if sequence else 1

        # Basic features
        length = len(sequence)
        charge = props["net_charge"]
        hydro = props["hydrophobicity"]
        hydro_ratio = props["hydrophobic_ratio"]
        cationic_ratio = props["cationic_ratio"]

        # Special residue content
        trp_content = sequence.count("W") / n
        phe_content = sequence.count("F") / n
        leu_content = sequence.count("L") / n
        ile_content = sequence.count("I") / n
        pro_content = sequence.count("P") / n
        gly_content = sequence.count("G") / n

        # Aromatic content (membrane interaction)
        aromatic_content = sum(1 for aa in sequence if aa in "FWY") / n

        # Aliphatic content
        aliphatic_content = sum(1 for aa in sequence if aa in "AILV") / n

        # Charge density
        charge_density = charge / max(length, 1)

        # Hydrophobic moment proxy (simplified)
        # Real HM requires 3D structure, use local window approach
        hydro_moment_proxy = 0
        window_size = min(7, n)
        if n >= window_size:
            for i in range(n - window_size + 1):
                window = sequence[i:i + window_size]
                window_hydro = [HYDROPHOBICITY.get(aa, 0) for aa in window]
                # Variance in window as proxy for amphipathicity
                hydro_moment_proxy += np.std(window_hydro)
            hydro_moment_proxy /= (n - window_size + 1)

        # Net hydrophobicity (for membrane penetration)
        net_hydrophobicity = sum(HYDROPHOBICITY.get(aa, 0) for aa in sequence)

        # Cysteine content (disulfide bonds stabilize structure)
        cys_content = sequence.count("C") / n

        return np.array([
            length,
            charge,
            hydro,
            hydro_ratio,
            cationic_ratio,
            trp_content,
            phe_content,
            leu_content,
            ile_content,
            pro_content,
            gly_content,
            aromatic_content,
            aliphatic_content,
            charge_density,
            hydro_moment_proxy,
            net_hydrophobicity,
            cys_content,
        ])

    def _train_models(self):
        """Train the hemolysis prediction models."""
        if not SKLEARN_AVAILABLE:
            print("Warning: scikit-learn not available, using rule-based predictions")
            return

        # Prepare training data
        X = []
        y_hc50 = []
        y_class = []

        for name, seq, hc50, is_hemolytic in CURATED_HEMOLYSIS_DATA:
            features = self._compute_hemolysis_features(seq)
            X.append(features)
            y_hc50.append(np.log10(hc50))  # Log-transform HC50
            y_class.append(1 if is_hemolytic else 0)

        X = np.array(X)
        y_hc50 = np.array(y_hc50)
        y_class = np.array(y_class)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train regressor for HC50 prediction
        self.regressor = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            min_samples_leaf=2,
            random_state=42,
        )
        self.regressor.fit(X_scaled, y_hc50)

        # Train classifier for hemolytic/non-hemolytic
        self.classifier = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            min_samples_leaf=2,
            random_state=42,
        )
        self.classifier.fit(X_scaled, y_class)

        self._is_trained = True

    def predict(self, sequence: str) -> dict:
        """Predict hemolytic activity of a peptide.

        Args:
            sequence: Amino acid sequence

        Returns:
            Dictionary with:
                - hc50_predicted: Predicted HC50 in μM
                - is_hemolytic: Boolean classification
                - hemolytic_probability: Probability of being hemolytic
                - risk_category: 'High', 'Moderate', or 'Low'
                - therapeutic_index_note: Interpretation note
        """
        features = self._compute_hemolysis_features(sequence)

        if self._is_trained and SKLEARN_AVAILABLE:
            features_scaled = self.scaler.transform(features.reshape(1, -1))

            # Predict HC50
            log_hc50 = self.regressor.predict(features_scaled)[0]
            hc50 = 10 ** log_hc50

            # Predict class probability
            proba = self.classifier.predict_proba(features_scaled)[0]
            hemolytic_prob = proba[1] if len(proba) > 1 else 0.5
            is_hemolytic = hemolytic_prob > 0.5
        else:
            # Rule-based fallback
            hc50 = self._rule_based_hc50(sequence, features)
            is_hemolytic = hc50 < 50
            hemolytic_prob = 0.8 if is_hemolytic else 0.2

        # Determine risk category
        if hc50 < 50:
            risk_category = "High"
            note = "Strong hemolytic activity predicted. Consider modifications to reduce toxicity."
        elif hc50 < 200:
            risk_category = "Moderate"
            note = "Moderate hemolytic activity. May need optimization for therapeutic use."
        else:
            risk_category = "Low"
            note = "Low hemolytic activity predicted. Favorable for therapeutic development."

        return {
            "sequence": sequence,
            "hc50_predicted": float(hc50),
            "is_hemolytic": bool(is_hemolytic),
            "hemolytic_probability": float(hemolytic_prob),
            "risk_category": risk_category,
            "therapeutic_index_note": note,
        }

    def _rule_based_hc50(self, sequence: str, features: np.ndarray) -> float:
        """Rule-based HC50 estimation (fallback).

        Args:
            sequence: Amino acid sequence
            features: Precomputed features

        Returns:
            Estimated HC50 in μM
        """
        length = features[0]
        hydro = features[2]
        hydro_ratio = features[3]
        trp_content = features[5]
        aromatic_content = features[11]
        aliphatic_content = features[12]

        # Base HC50 estimate
        base_hc50 = 200.0

        # High hydrophobicity increases hemolysis
        if hydro > 1.0:
            base_hc50 /= (1 + hydro * 0.5)

        # High aromatic/aliphatic content increases hemolysis
        if aromatic_content > 0.2:
            base_hc50 /= (1 + aromatic_content * 3)

        if aliphatic_content > 0.4:
            base_hc50 /= (1 + (aliphatic_content - 0.3) * 2)

        # Tryptophan particularly increases membrane disruption
        if trp_content > 0.1:
            base_hc50 /= (1 + trp_content * 5)

        # Short peptides tend to be less hemolytic
        if length < 15:
            base_hc50 *= 1.5

        # Very long peptides may be more hemolytic
        if length > 30:
            base_hc50 /= 1.3

        return max(1.0, min(1000.0, base_hc50))

    def predict_batch(self, sequences: list[str]) -> list[dict]:
        """Predict hemolytic activity for multiple peptides.

        Args:
            sequences: List of amino acid sequences

        Returns:
            List of prediction dictionaries
        """
        return [self.predict(seq) for seq in sequences]

    def compute_therapeutic_index(
        self,
        sequence: str,
        mic_value: float,
    ) -> dict:
        """Compute therapeutic index (HC50/MIC).

        A higher therapeutic index indicates better selectivity
        for bacteria over mammalian cells.

        Args:
            sequence: Amino acid sequence
            mic_value: Minimum inhibitory concentration in μM

        Returns:
            Dictionary with therapeutic index metrics
        """
        hemo_result = self.predict(sequence)
        hc50 = hemo_result["hc50_predicted"]

        if mic_value <= 0:
            therapeutic_index = 0.0
            interpretation = "Invalid MIC value"
        else:
            therapeutic_index = hc50 / mic_value

            if therapeutic_index > 10:
                interpretation = "Excellent selectivity"
            elif therapeutic_index > 5:
                interpretation = "Good selectivity"
            elif therapeutic_index > 2:
                interpretation = "Moderate selectivity"
            else:
                interpretation = "Poor selectivity - high toxicity risk"

        return {
            "sequence": sequence,
            "hc50": hc50,
            "mic": mic_value,
            "therapeutic_index": float(therapeutic_index),
            "interpretation": interpretation,
            **hemo_result,
        }


def get_hemolysis_predictor() -> HemolysisPredictor:
    """Get a hemolysis predictor instance.

    Returns:
        Trained HemolysisPredictor
    """
    return HemolysisPredictor()
