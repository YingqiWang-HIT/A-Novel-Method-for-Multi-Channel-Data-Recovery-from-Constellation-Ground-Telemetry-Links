"""Adapter skeleton for COOL.

This file intentionally does not contain the official COOL source code. The method is a
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


class CoolAdapter(ExternalBaselineAdapter):
    baseline_name = "COOL"
    citation_hint = "dynamic spatio-temporal baseline. Please cite and follow the original authors' license."


def build_model(*args, **kwargs):
    return CoolAdapter(*args, **kwargs)
