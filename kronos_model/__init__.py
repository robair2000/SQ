"""Vendored copy of Kronos (https://github.com/shiyu-coder/Kronos, MIT License, commit 67b630e).

Changes: the absolute import `from model.module import *` in kronos.py is now relative, and pandas is
imported lazily (the inference path in kronos_infer.py does not need it, which saves memory).
"""
from .kronos import KronosTokenizer, Kronos, KronosPredictor  # noqa: F401
