"""Adapter skeleton for Autoformer.

This file intentionally does not contain the official Autoformer source code. The method is a
third-party baseline and may be subject to the original authors' license, copyright,
or redistribution restrictions. To reproduce the comparison, please read the original
paper and official repository, install the authorized implementation locally, and then
map its train/test API inside this adapter.
"""
from __future__ import annotations

try:
    from .base import ExternalBaselineAdapter
except ImportError:  # support direct execution for quick inspection
    from base import ExternalBaselineAdapter


class AutoformerAdapter(ExternalBaselineAdapter):
    baseline_name = "Autoformer"
    citation_hint = "single-temporal-scale baseline for long-term time-series modeling. Please cite and follow the original authors' license."


def build_model(*args, **kwargs):
    return AutoformerAdapter(*args, **kwargs)
