Disease Analyzers API
=====================

This module provides unified drug resistance and escape prediction across 11 disease domains.

Base Classes
------------

DiseaseAnalyzer
~~~~~~~~~~~~~~~

.. automodule:: src.diseases.base
   :members:
   :undoc-members:
   :show-inheritance:


Uncertainty-Aware Analyzer
--------------------------

Integrates uncertainty quantification (MC Dropout, Evidential, Ensemble) into disease prediction.

.. automodule:: src.diseases.uncertainty_aware_analyzer
   :members:
   :undoc-members:
   :show-inheritance:


Disease Registry
----------------

.. automodule:: src.diseases.registry
   :members:
   :undoc-members:
   :show-inheritance:


Viral Disease Analyzers
-----------------------

HIV Analyzer
~~~~~~~~~~~~

23 antiretroviral drugs, transfer learning, 0.89 Spearman correlation.

.. automodule:: src.diseases.hiv_analyzer
   :members:
   :undoc-members:
   :show-inheritance:


SARS-CoV-2 Analyzer
~~~~~~~~~~~~~~~~~~~

Paxlovid resistance, Spike protein escape prediction.

.. automodule:: src.diseases.sars_cov2_analyzer
   :members:
   :undoc-members:
   :show-inheritance:


Influenza Analyzer
~~~~~~~~~~~~~~~~~~

Neuraminidase inhibitors, baloxavir, vaccine strain selection.

.. automodule:: src.diseases.influenza_analyzer
   :members:
   :undoc-members:
   :show-inheritance:


HCV Analyzer
~~~~~~~~~~~~

Direct-acting antivirals (NS3/NS5A/NS5B RAS).

.. automodule:: src.diseases.hcv_analyzer
   :members:
   :undoc-members:
   :show-inheritance:


HBV Analyzer
~~~~~~~~~~~~

Nucleos(t)ide analogues, S-gene overlap analysis.

.. automodule:: src.diseases.hbv_analyzer
   :members:
   :undoc-members:
   :show-inheritance:


RSV Analyzer
~~~~~~~~~~~~

Nirsevimab and palivizumab escape prediction.

.. automodule:: src.diseases.rsv_analyzer
   :members:
   :undoc-members:
   :show-inheritance:


Bacterial Disease Analyzers
---------------------------

Tuberculosis Analyzer
~~~~~~~~~~~~~~~~~~~~~

13 drugs, MDR/XDR classification.

.. automodule:: src.diseases.tuberculosis_analyzer
   :members:
   :undoc-members:
   :show-inheritance:


MRSA Analyzer
~~~~~~~~~~~~~

mecA/mecC detection, MDR profiling.

.. automodule:: src.diseases.mrsa_analyzer
   :members:
   :undoc-members:
   :show-inheritance:


Fungal Disease Analyzers
------------------------

Candida Analyzer
~~~~~~~~~~~~~~~~

Pan-resistance alerts for Candida auris.

.. automodule:: src.diseases.candida_analyzer
   :members:
   :undoc-members:
   :show-inheritance:


Parasitic Disease Analyzers
---------------------------

Malaria Analyzer
~~~~~~~~~~~~~~~~

K13 artemisinin resistance, ACT combinations.

.. automodule:: src.diseases.malaria_analyzer
   :members:
   :undoc-members:
   :show-inheritance:


Oncology Analyzers
------------------

Cancer Analyzer
~~~~~~~~~~~~~~~

EGFR/BRAF/KRAS/ALK tyrosine kinase inhibitor resistance.

.. automodule:: src.diseases.cancer_analyzer
   :members:
   :undoc-members:
   :show-inheritance:
