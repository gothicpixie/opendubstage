"""Restoration: RX-style spectral repair (``specrepair``) and gap-learned LF rumble removal (``memofix``).

``memofix`` is both a submodule and its main function; use ``from opendubstage.restore import memofix`` for the
module and ``memofix.memofix(x, sr, ...)`` for the function.
"""
from . import memofix, specrepair
from .memofix import MemoParams
from .specrepair import (Params, Patch, detect_thumps, eq_match, level_db, plan_file, repair_array, repair_event,
                         repair_file, repair_thumps, spectral_attenuate, write_patched)

__all__ = ["memofix", "specrepair", "MemoParams", "Params", "Patch", "detect_thumps", "eq_match", "level_db",
           "plan_file", "repair_array", "repair_event", "repair_file", "repair_thumps", "spectral_attenuate",
           "write_patched"]
