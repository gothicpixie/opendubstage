"""opendubstage.foley (formerly stillfoley): documentary-grade sound for still photographs and archival footage."""
from .analysis import KeywordAnalyzer
from .model import Cue, Interval, Layer, MusicSegment, NarrationSegment, SceneAnalysis, Sequence, Still
from .rules import CueSheet, Rules, check, plan

__version__ = "0.1.0.dev0"
__all__ = ["KeywordAnalyzer", "Cue", "CueSheet", "Interval", "Layer", "MusicSegment", "NarrationSegment",
           "Rules", "SceneAnalysis", "Sequence", "Still", "check", "plan"]
