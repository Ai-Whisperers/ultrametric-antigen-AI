Models API
==========

Core VAE Models
---------------

.. automodule:: src.models.ternary_vae
   :members:
   :undoc-members:
   :show-inheritance:


BaseVAE
-------

Abstract base class for all VAE variants, reducing code duplication across 19+ models.

.. automodule:: src.models.base_vae
   :members:
   :undoc-members:
   :show-inheritance:


Structure-Aware VAE
-------------------

VAE with integrated 3D protein structure encoding using AlphaFold2 predictions.

.. automodule:: src.models.structure_aware_vae
   :members:
   :undoc-members:
   :show-inheritance:


SwarmVAE
--------

.. automodule:: src.models.swarm_vae
   :members:
   :undoc-members:
   :show-inheritance:


Epistasis Module
----------------

Mutation interaction modeling for epistatic effects in drug resistance.

.. automodule:: src.models.epistasis_module
   :members:
   :undoc-members:
   :show-inheritance:


Model Components
----------------

Homeostasis Controller
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: src.models.homeostasis
   :members:
   :undoc-members:
   :show-inheritance:


Curriculum Learning
~~~~~~~~~~~~~~~~~~~

.. automodule:: src.models.curriculum
   :members:
   :undoc-members:
   :show-inheritance:
