"""
Escape Mutation Predictor for HIV.

Machine learning model to predict immune and drug escape mutations
using hyperbolic geometry, structural features, and evolutionary signals.

Key features:
1. Feature extraction from sequence and structure
2. Multi-task learning for CTL and antibody escape
3. Gradient boosting and neural network models
4. Uncertainty quantification
5. Interpretable feature importance

Based on papers:
- Barton et al. 2016: Fitness landscape of HIV-1
- Ferguson et al. 2013: Translating HIV sequences
- Louie et al. 2018: Escape prediction

Requirements:
    pip install scikit-learn numpy pandas

Author: Research Team
Date: December 2025
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# Lazy imports for optional dependencies
_SKLEARN_AVAILABLE = False
_TORCH_AVAILABLE = False

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        roc_auc_score,
    )
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:
    pass

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    pass


# Amino acid properties for feature extraction
AMINO_ACID_PROPERTIES = {
    # (hydrophobicity, volume, charge, polarity)
    "A": (1.8, 88.6, 0, 0),
    "R": (-4.5, 173.4, 1, 1),
    "N": (-3.5, 114.1, 0, 1),
    "D": (-3.5, 111.1, -1, 1),
    "C": (2.5, 108.5, 0, 0),
    "Q": (-3.5, 143.8, 0, 1),
    "E": (-3.5, 138.4, -1, 1),
    "G": (-0.4, 60.1, 0, 0),
    "H": (-3.2, 153.2, 0.5, 1),
    "I": (4.5, 166.7, 0, 0),
    "L": (3.8, 166.7, 0, 0),
    "K": (-3.9, 168.6, 1, 1),
    "M": (1.9, 162.9, 0, 0),
    "F": (2.8, 189.9, 0, 0),
    "P": (-1.6, 112.7, 0, 0),
    "S": (-0.8, 89.0, 0, 1),
    "T": (-0.7, 116.1, 0, 1),
    "W": (-0.9, 227.8, 0, 0),
    "Y": (-1.3, 193.6, 0, 1),
    "V": (4.2, 140.0, 0, 0),
}


@dataclass
class EscapeFeatures:
    """Features for escape prediction."""

    # Sequence features
    position: int
    wild_type_aa: str
    mutant_aa: str
    conservation_score: float
    entropy: float

    # Structural features
    surface_accessibility: float
    secondary_structure: str  # H, E, C
    distance_to_active_site: float

    # Hyperbolic geometry features
    hyperbolic_distance: float
    radial_position_wt: float
    radial_position_mut: float
    boundary_crossing: bool

    # Evolutionary features
    substitution_rate: float
    selection_coefficient: float
    fitness_cost: float

    # Immune features (optional)
    hla_restriction: Optional[str] = None
    epitope_position: Optional[str] = None  # anchor, flanking, processing
    ic50_impact: Optional[float] = None


def extract_aa_features(wt_aa: str, mut_aa: str) -> dict:
    """
    Extract amino acid property change features.

    Args:
        wt_aa: Wild-type amino acid
        mut_aa: Mutant amino acid

    Returns:
        Dictionary of property change features
    """
    wt_props = AMINO_ACID_PROPERTIES.get(wt_aa.upper(), (0, 0, 0, 0))
    mut_props = AMINO_ACID_PROPERTIES.get(mut_aa.upper(), (0, 0, 0, 0))

    return {
        "hydrophobicity_change": mut_props[0] - wt_props[0],
        "volume_change": mut_props[1] - wt_props[1],
        "charge_change": mut_props[2] - wt_props[2],
        "polarity_change": mut_props[3] - wt_props[3],
        "hydrophobicity_wt": wt_props[0],
        "hydrophobicity_mut": mut_props[0],
        "volume_wt": wt_props[1],
        "volume_mut": mut_props[1],
    }


def extract_positional_features(
    position: int,
    sequence_length: int,
    epitope_start: int = 0,
    epitope_end: int = 0,
) -> dict:
    """
    Extract position-based features.

    Args:
        position: Position in sequence
        sequence_length: Total sequence length
        epitope_start: Start of epitope (if applicable)
        epitope_end: End of epitope (if applicable)

    Returns:
        Dictionary of positional features
    """
    relative_position = position / sequence_length

    # Position within epitope
    if epitope_start <= position <= epitope_end:
        epitope_relative = (position - epitope_start) / max(1, epitope_end - epitope_start)
        in_epitope = 1
    else:
        epitope_relative = -1
        in_epitope = 0

    # Common anchor positions (positions 2 and C-terminus for HLA class I)
    is_anchor = 1 if position in [epitope_start + 1, epitope_end - 1, epitope_end] else 0

    return {
        "relative_position": relative_position,
        "in_epitope": in_epitope,
        "epitope_relative_position": epitope_relative,
        "is_anchor_position": is_anchor,
        "distance_from_n_term": position,
        "distance_from_c_term": sequence_length - position,
    }


def build_feature_vector(
    features: EscapeFeatures,
    include_aa_properties: bool = True,
) -> np.ndarray:
    """
    Build feature vector from EscapeFeatures.

    Args:
        features: EscapeFeatures dataclass
        include_aa_properties: Whether to include AA property changes

    Returns:
        Feature vector as numpy array
    """
    vec = [
        features.conservation_score,
        features.entropy,
        features.surface_accessibility,
        features.hyperbolic_distance,
        features.radial_position_wt,
        features.radial_position_mut,
        1.0 if features.boundary_crossing else 0.0,
        features.substitution_rate,
        features.selection_coefficient,
        features.fitness_cost,
        features.distance_to_active_site,
    ]

    # Add secondary structure one-hot
    ss_map = {"H": [1, 0, 0], "E": [0, 1, 0], "C": [0, 0, 1]}
    vec.extend(ss_map.get(features.secondary_structure, [0, 0, 1]))

    if include_aa_properties:
        aa_feats = extract_aa_features(features.wild_type_aa, features.mutant_aa)
        vec.extend([
            aa_feats["hydrophobicity_change"],
            aa_feats["volume_change"],
            aa_feats["charge_change"],
            aa_feats["polarity_change"],
        ])

    return np.array(vec, dtype=np.float32)


class EscapePredictor:
    """
    Machine learning model for escape mutation prediction.

    Supports multiple model types and provides uncertainty estimates.
    """

    def __init__(
        self,
        model_type: str = "gradient_boosting",
        random_state: int = 42,
    ):
        """
        Initialize escape predictor.

        Args:
            model_type: 'gradient_boosting', 'random_forest', or 'logistic'
            random_state: Random seed for reproducibility
        """
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")

        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False

    def _create_model(self):
        """Create the underlying ML model."""
        if self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state,
            )
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
            )
        elif self.model_type == "logistic":
            return LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list[str]] = None,
        test_size: float = 0.2,
    ) -> dict:
        """
        Train the escape predictor.

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Labels (0 = no escape, 1 = escape), shape (n_samples,)
            feature_names: Optional list of feature names
            test_size: Fraction for test set

        Returns:
            Dictionary with training metrics
        """
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        # Cross-validation
        cv_scores = cross_val_score(
            self._create_model(),
            self.scaler.transform(X),
            y,
            cv=5,
            scoring="roc_auc",
        )

        self.is_trained = True

        return {
            "accuracy": accuracy,
            "auc": auc,
            "cv_auc_mean": np.mean(cv_scores),
            "cv_auc_std": np.std(cv_scores),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict escape probability.

        Args:
            X: Feature matrix, shape (n_samples, n_features)

        Returns:
            Predicted labels, shape (n_samples,)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict escape probability with confidence.

        Args:
            X: Feature matrix, shape (n_samples, n_features)

        Returns:
            Probability of escape, shape (n_samples,)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance rankings.

        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importance = np.abs(self.model.coef_[0])
        else:
            importance = np.zeros(len(self.feature_names))

        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        })

        return df.sort_values("importance", ascending=False)


class CTLEscapePredictor(EscapePredictor):
    """
    Specialized predictor for CTL escape mutations.

    Adds HLA-specific features and epitope processing signals.
    """

    # Known HLA anchor preferences
    HLA_ANCHORS = {
        "A*02:01": {1: ["L", "M", "I", "V"], -1: ["V", "L", "I"]},
        "A*03:01": {1: ["L", "V", "I"], -1: ["K", "R"]},
        "B*27:05": {1: ["R"], -1: ["L", "F", "K"]},
        "B*57:01": {1: ["A", "S", "T"], -1: ["W", "F", "Y"]},
    }

    def add_hla_features(
        self,
        features: np.ndarray,
        hla_type: str,
        epitope_sequence: str,
        mutation_position: int,
    ) -> np.ndarray:
        """
        Add HLA-specific features to feature vector.

        Args:
            features: Base feature vector
            hla_type: HLA allele (e.g., "A*02:01")
            epitope_sequence: Epitope amino acid sequence
            mutation_position: Position within epitope (0-indexed)

        Returns:
            Extended feature vector
        """
        anchors = self.HLA_ANCHORS.get(hla_type, {})

        # Check if mutation is at anchor position
        is_anchor = mutation_position in [1, len(epitope_sequence) - 1]

        # Check if mutation disrupts anchor preference
        if is_anchor and mutation_position + 1 in anchors:
            preferred = anchors[mutation_position + 1]
            current_aa = epitope_sequence[mutation_position]
            disrupts_anchor = 0 if current_aa in preferred else 1
        else:
            disrupts_anchor = 0

        # Estimate processing score (simplified)
        # Mutations near N/C terminus affect proteasomal processing
        n_term_proximity = 1.0 / (mutation_position + 1)
        c_term_proximity = 1.0 / (len(epitope_sequence) - mutation_position)

        hla_features = np.array([
            is_anchor,
            disrupts_anchor,
            n_term_proximity,
            c_term_proximity,
        ])

        return np.concatenate([features, hla_features])


class AntibodyEscapePredictor(EscapePredictor):
    """
    Specialized predictor for antibody escape mutations.

    Adds structural and binding features.
    """

    def add_structural_features(
        self,
        features: np.ndarray,
        distance_to_epitope_center: float,
        glycan_nearby: bool,
        buried_surface_area: float,
        hydrogen_bonds: int,
    ) -> np.ndarray:
        """
        Add structural features for antibody escape prediction.

        Args:
            features: Base feature vector
            distance_to_epitope_center: Distance from mutation to epitope center
            glycan_nearby: Whether glycosylation site is nearby
            buried_surface_area: Area buried upon antibody binding
            hydrogen_bonds: Number of H-bonds with antibody

        Returns:
            Extended feature vector
        """
        struct_features = np.array([
            distance_to_epitope_center,
            1.0 if glycan_nearby else 0.0,
            buried_surface_area,
            hydrogen_bonds,
        ])

        return np.concatenate([features, struct_features])


class EnsembleEscapePredictor:
    """
    Ensemble of multiple escape predictors.

    Combines predictions from different model types for better accuracy.
    """

    def __init__(self, random_state: int = 42):
        """Initialize ensemble predictor."""
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required")

        self.predictors = [
            EscapePredictor("gradient_boosting", random_state),
            EscapePredictor("random_forest", random_state + 1),
            EscapePredictor("logistic", random_state + 2),
        ]
        self.weights = None
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Train all models in ensemble."""
        metrics = {}

        for i, predictor in enumerate(self.predictors):
            pred_metrics = predictor.train(X, y)
            metrics[f"model_{i}_{predictor.model_type}"] = pred_metrics

        # Set weights based on cross-validation AUC
        aucs = [
            metrics[f"model_{i}_{p.model_type}"]["cv_auc_mean"]
            for i, p in enumerate(self.predictors)
        ]
        total_auc = sum(aucs)
        self.weights = [auc / total_auc for auc in aucs]

        self.is_trained = True

        # Calculate ensemble metrics
        y_prob_ensemble = self.predict_proba(X)
        metrics["ensemble_auc"] = roc_auc_score(y, y_prob_ensemble)
        metrics["weights"] = dict(zip(
            [p.model_type for p in self.predictors],
            self.weights
        ))

        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get weighted ensemble probability."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained first")

        probs = np.zeros(len(X))
        for predictor, weight in zip(self.predictors, self.weights):
            probs += weight * predictor.predict_proba(X)

        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Get ensemble predictions."""
        return (self.predict_proba(X) >= threshold).astype(int)


def generate_synthetic_training_data(
    n_samples: int = 1000,
    n_features: int = 18,
    escape_rate: float = 0.3,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data for testing.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        escape_rate: Fraction of escape mutations
        random_state: Random seed

    Returns:
        Tuple of (X, y) arrays
    """
    np.random.seed(random_state)

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Make some features more predictive
    # Higher entropy = more likely to escape (less constrained)
    entropy_idx = 1
    # Lower conservation = more likely to escape
    conservation_idx = 0
    # Higher hyperbolic distance = more likely to escape
    hyp_dist_idx = 3

    # Generate labels with some structure
    escape_score = (
        0.3 * X[:, entropy_idx] -
        0.4 * X[:, conservation_idx] +
        0.3 * X[:, hyp_dist_idx] +
        np.random.randn(n_samples) * 0.5
    )

    # Convert to probabilities
    probs = 1 / (1 + np.exp(-escape_score))

    # Adjust to match target escape rate
    threshold = np.percentile(probs, (1 - escape_rate) * 100)
    y = (probs > threshold).astype(int)

    return X, y


def analyze_escape_patterns(
    predictor: EscapePredictor,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> dict:
    """
    Analyze patterns in escape predictions.

    Args:
        predictor: Trained EscapePredictor
        X: Feature matrix
        y: True labels
        feature_names: Feature names

    Returns:
        Dictionary with pattern analysis
    """
    probs = predictor.predict_proba(X)

    # High confidence escapes
    high_conf_escape = (probs > 0.8) & (y == 1)
    high_conf_non_escape = (probs < 0.2) & (y == 0)

    # Feature statistics for high-confidence cases
    escape_features = X[high_conf_escape].mean(axis=0) if any(high_conf_escape) else np.zeros(X.shape[1])
    non_escape_features = X[high_conf_non_escape].mean(axis=0) if any(high_conf_non_escape) else np.zeros(X.shape[1])

    feature_diff = escape_features - non_escape_features

    # Get feature importance
    importance = predictor.get_feature_importance()

    return {
        "n_high_conf_escape": int(sum(high_conf_escape)),
        "n_high_conf_non_escape": int(sum(high_conf_non_escape)),
        "top_features": importance.head(5).to_dict(),
        "escape_non_escape_diff": dict(zip(feature_names, feature_diff)),
        "mean_escape_prob_true_escape": float(probs[y == 1].mean()),
        "mean_escape_prob_true_non_escape": float(probs[y == 0].mean()),
    }


def generate_prediction_report(
    predictor: EscapePredictor,
    X: np.ndarray,
    mutations: list[str],
) -> str:
    """
    Generate prediction report for a set of mutations.

    Args:
        predictor: Trained predictor
        X: Feature matrix for mutations
        mutations: List of mutation names (e.g., ["D30N", "M46I"])

    Returns:
        Formatted report string
    """
    probs = predictor.predict_proba(X)
    preds = predictor.predict(X)

    lines = [
        "=" * 60,
        "ESCAPE MUTATION PREDICTION REPORT",
        "=" * 60,
        "",
        f"Mutations analyzed: {len(mutations)}",
        "",
        "PREDICTIONS:",
        "-" * 40,
    ]

    # Sort by probability
    sorted_idx = np.argsort(probs)[::-1]

    for idx in sorted_idx:
        risk = "HIGH" if probs[idx] > 0.7 else "MEDIUM" if probs[idx] > 0.4 else "LOW"
        lines.append(
            f"  {mutations[idx]:10} | P(escape): {probs[idx]:.3f} | Risk: {risk}"
        )

    # Summary
    high_risk = sum(probs > 0.7)
    medium_risk = sum((probs > 0.4) & (probs <= 0.7))
    low_risk = sum(probs <= 0.4)

    lines.extend([
        "",
        "SUMMARY:",
        f"  High risk (>70%): {high_risk}",
        f"  Medium risk (40-70%): {medium_risk}",
        f"  Low risk (<40%): {low_risk}",
        "",
        "=" * 60,
    ])

    return "\n".join(lines)


# Example usage
if __name__ == "__main__":
    print("Testing Escape Mutation Predictor Module")
    print("=" * 50)

    if not _SKLEARN_AVAILABLE:
        print("scikit-learn not available. Install with: pip install scikit-learn")
        exit(0)

    # Generate synthetic data
    print("\nGenerating synthetic training data...")
    X, y = generate_synthetic_training_data(n_samples=500, escape_rate=0.3)
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Escape rate: {y.mean():.1%}")

    # Feature names (example)
    feature_names = [
        "conservation", "entropy", "surface_accessibility",
        "hyperbolic_distance", "radial_wt", "radial_mut",
        "boundary_crossing", "substitution_rate", "selection_coef",
        "fitness_cost", "active_site_dist", "ss_helix", "ss_sheet",
        "ss_coil", "hydro_change", "volume_change", "charge_change",
        "polarity_change",
    ]

    # Test single predictor
    print("\nTraining gradient boosting predictor...")
    predictor = EscapePredictor(model_type="gradient_boosting")
    metrics = predictor.train(X, y, feature_names=feature_names)

    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  AUC: {metrics['auc']:.3f}")
    print(f"  CV AUC: {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}")

    # Feature importance
    print("\nTop 5 Important Features:")
    importance = predictor.get_feature_importance()
    for _, row in importance.head(5).iterrows():
        print(f"  {row['feature']:25} | {row['importance']:.4f}")

    # Test ensemble
    print("\n" + "=" * 50)
    print("Training ensemble predictor...")
    ensemble = EnsembleEscapePredictor()
    ensemble_metrics = ensemble.train(X, y)

    print(f"  Ensemble AUC: {ensemble_metrics['ensemble_auc']:.3f}")
    print(f"  Model weights: {ensemble_metrics['weights']}")

    # Pattern analysis
    print("\nAnalyzing escape patterns...")
    patterns = analyze_escape_patterns(predictor, X, y, feature_names)
    print(f"  High confidence escapes: {patterns['n_high_conf_escape']}")
    print(f"  Mean prob for true escapes: {patterns['mean_escape_prob_true_escape']:.3f}")

    # Generate report for hypothetical mutations
    print("\n" + "=" * 50)
    mutations = ["D30N", "M46I", "V82A", "L90M", "I84V"]
    X_test = np.random.randn(5, X.shape[1])  # Synthetic features
    print(generate_prediction_report(predictor, X_test, mutations))

    print("Module testing complete!")
