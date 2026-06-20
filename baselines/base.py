"""
Shared placeholder utilities for third-party baseline adapters.

The public TiDAL-Net release does not redistribute third-party baseline code.
Each baseline file contains an adapter skeleton so that users can plug in the
corresponding official implementation after obtaining it from the original authors.
"""
from __future__ import annotations


class RightsRestrictedBaselineError(NotImplementedError):
    pass


class ExternalBaselineAdapter:
    baseline_name = "ExternalBaseline"
    citation_hint = "Please consult the original paper and official repository."

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def _raise(self):
        raise RightsRestrictedBaselineError(
            f"{self.baseline_name} is a third-party comparison method and is not redistributed in this public repository. "
            f"Please read the original paper and obtain the official implementation from the authors before completing this adapter. "
            f"Citation hint: {self.citation_hint}"
        )

    def fit(self, *args, **kwargs):
        self._raise()

    def predict(self, *args, **kwargs):
        self._raise()

    def fit_predict(self, *args, **kwargs):
        self._raise()
