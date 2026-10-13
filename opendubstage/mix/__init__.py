"""Mixing: speech-keyed music carve and posts (``carve``) and cue-sheet transitions (``transitions``).

The submodules share their main function's name, so call them through the module:
``from opendubstage.mix import carve; carve.carve(mx, dx, sr)`` and
``from opendubstage.mix import transitions; transitions.transitions(cues, chapters)``.
"""
from . import carve, transitions
from .carve import band_energy_ratio_db, carve_blocks, carve_pad, phrase_safe_end, posts
from .transitions import ProbeCache, Settings, bridge_gaps, check_transitions

__all__ = ["carve", "transitions", "band_energy_ratio_db", "carve_blocks", "carve_pad", "phrase_safe_end", "posts",
           "ProbeCache", "Settings", "bridge_gaps", "check_transitions"]
