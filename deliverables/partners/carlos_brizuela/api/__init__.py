"""AMP Design API - Classical endpoints for mechanism-based design.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! DISCLAIMER: PREMATURE MOCK - NOT PRODUCTION READY                 !!
!! C5 hold-out generalization test NOT RUN. R2 constraint violated.  !!
!! See amp_design_api.py for full disclaimer.                        !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""

from .mechanism_service import get_mechanism_service, MechanismDesignService
from .amp_design_api import app, create_app

__all__ = [
    "get_mechanism_service",
    "MechanismDesignService",
    "app",
    "create_app",
]
