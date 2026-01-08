"""AMP Design API - Classical endpoints for mechanism-based design.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! C5 FALSIFIED - PATHOGEN METADATA PROVIDES NO IMPROVEMENT          !!
!! Endpoints /design/rules and /predict/pathogen-rank are MISLEADING !!
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
