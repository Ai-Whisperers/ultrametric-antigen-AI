Uncertainty Quantification
==========================

This tutorial covers how to use uncertainty quantification in drug resistance predictions.

Overview
--------

Uncertainty quantification helps answer: "How confident is the model in its prediction?"

The framework supports three uncertainty methods:

1. **MC Dropout**: Multiple forward passes with dropout enabled
2. **Evidential**: Single-pass uncertainty using evidential deep learning
3. **Ensemble**: Multiple model predictions combined

Quick Start
-----------

.. code-block:: python

    from src.diseases.uncertainty_aware_analyzer import (
        UncertaintyAwareAnalyzer,
        UncertaintyConfig,
        UncertaintyMethod,
    )
    from src.diseases.hiv_analyzer import HIVAnalyzer

    # Create base analyzer
    base_analyzer = HIVAnalyzer()

    # Configure uncertainty
    config = UncertaintyConfig(
        method=UncertaintyMethod.EVIDENTIAL,
        confidence_level=0.95,
        calibrate=True,
        decompose=True,  # Split into epistemic/aleatoric
    )

    # Wrap with uncertainty
    analyzer = UncertaintyAwareAnalyzer(
        base_analyzer,
        config=config,
        model=your_trained_model,
    )

    # Analyze with uncertainty
    results = analyzer.analyze_with_uncertainty(
        sequences=["MKTEFPSASLY..."],
        encodings=encoded_sequences,
    )

    # Access uncertainty
    for drug, data in results["drug_resistance"].items():
        print(f"{drug}:")
        print(f"  Prediction: {data['scores']}")
        print(f"  Confidence Interval: {data['uncertainty']['lower']} - {data['uncertainty']['upper']}")
        print(f"  Epistemic (model): {data['uncertainty']['epistemic']}")
        print(f"  Aleatoric (data): {data['uncertainty']['aleatoric']}")


Uncertainty Methods
-------------------

MC Dropout
~~~~~~~~~~

Monte Carlo Dropout runs multiple forward passes with dropout enabled:

.. code-block:: python

    config = UncertaintyConfig(
        method=UncertaintyMethod.MC_DROPOUT,
        n_samples=50,  # Number of forward passes
        confidence_level=0.95,
    )

**Pros**: Works with any model with dropout layers
**Cons**: Slow (multiple forward passes)


Evidential Deep Learning
~~~~~~~~~~~~~~~~~~~~~~~~

Evidential models output distribution parameters directly:

.. code-block:: python

    config = UncertaintyConfig(
        method=UncertaintyMethod.EVIDENTIAL,
        decompose=True,  # Get epistemic/aleatoric split
    )

**Pros**: Single forward pass, natural uncertainty decomposition
**Cons**: Requires specially trained model


Ensemble
~~~~~~~~

Combines predictions from multiple trained models:

.. code-block:: python

    config = UncertaintyConfig(
        method=UncertaintyMethod.ENSEMBLE,
        n_models=5,  # Number of models in ensemble
    )

**Pros**: Most robust uncertainty estimates
**Cons**: Requires training multiple models


Calibration
-----------

Uncertainty estimates should be calibrated - a 95% confidence interval should contain the true value 95% of the time.

.. code-block:: python

    # Calibrate on validation data
    analyzer.calibrate(
        x_val=validation_encodings,
        y_val=validation_targets,
    )

    # Check calibration quality
    metrics = analyzer.evaluate_uncertainty_quality(
        x_test=test_encodings,
        y_test=test_targets,
    )
    print(f"Coverage at 95%: {metrics['coverage_95']}")  # Should be ~0.95
    print(f"Calibration Error: {metrics['calibration_error']}")  # Lower is better


Uncertainty Decomposition
-------------------------

Understanding the source of uncertainty:

- **Epistemic Uncertainty**: Model uncertainty (reducible with more data)
- **Aleatoric Uncertainty**: Data uncertainty (irreducible noise)

.. code-block:: python

    config = UncertaintyConfig(
        method=UncertaintyMethod.EVIDENTIAL,
        decompose=True,
    )

    results = analyzer.analyze_with_uncertainty(sequences, encodings=x)

    # High epistemic = model is unsure (need more training data)
    # High aleatoric = inherent data variability


Clinical Decision Support
-------------------------

Use uncertainty for clinical decision-making:

.. code-block:: python

    def should_recommend_drug(prediction, uncertainty):
        """Recommend drug only if high confidence of resistance < 0.5."""
        # Resistance prediction
        if prediction > 0.7:
            return "AVOID - High resistance predicted"

        # Check uncertainty
        upper_bound = prediction + 1.96 * uncertainty
        if upper_bound > 0.5:
            return "CAUTION - Uncertain prediction, consider alternative"

        return "RECOMMEND - Low resistance with high confidence"


Best Practices
--------------

1. **Always calibrate** on held-out validation data
2. **Use evidential** for production (fast, single forward pass)
3. **Use ensemble** when accuracy is critical
4. **Flag high uncertainty** predictions for expert review
5. **Monitor calibration** over time as data distribution shifts


See Also
--------

- :doc:`/api/diseases` for full API reference
- :doc:`hiv_resistance` for HIV-specific examples
- :doc:`meta_learning` for few-shot uncertainty
