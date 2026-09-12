"""
Bias Evaluation and Mitigation Package
Provides core detection, scoring, and mitigation engines for the Responsible AI Suite.
"""

from bias.detection import (
    BiasDetector,
    calculate_contextual_bias,
    explain_contextual_bias,
    is_prompt_biased,
)
from bias.mitigation import MitigationEngine
from bias.scoring import (
    calculate_reduction,
    calculate_score,
    category_wise_stats,
    get_severity_label,
)

__all__ = [
    "BiasDetector",
    "MitigationEngine",
    "calculate_score",
    "calculate_contextual_bias",
    "explain_contextual_bias",
    "is_prompt_biased",
    "get_severity_label",
    "calculate_reduction",
    "category_wise_stats",
]
