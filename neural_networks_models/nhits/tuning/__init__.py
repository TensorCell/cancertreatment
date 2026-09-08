"""nhits.tuning — Ray Tune hyperparameter search for N-HiTS.

Public API
----------
search_space
    Default Ray Tune search-space dict.
tune_nhits
    Ray Tune trainable function (one trial = one call).
run_tuning
    Entry-point for a full hyperparameter search.
"""

from nhits.tuning.search_space import default_search_space
from nhits.tuning.trainable import tune_nhits

__all__ = ["default_search_space", "tune_nhits"]
