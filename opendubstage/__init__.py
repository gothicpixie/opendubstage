"""OpenDubStage: open tools for documentary and film sound post-production.

Subpackages
-----------
restore   spectral repair (room-tone Replace, spectral Attenuate, LF thumps) and car/room rumble removal
match     long-term-average-spectrum dialogue EQ matching
mix       speech-keyed music carve, music posts, phrase-safe music outs, J/L-cut transitions
qc        public-radio-style mix QC (loudness, LRA, true peak, dead air, voice jumps, tonal balance)
foley     restrained, rule-based sound placement for stills and archival footage

The audio cores need only numpy, scipy and soundfile; ``foley`` is stdlib-only. Heavy model backends are optional
extras and are never imported unless asked for.
"""
__version__ = "0.1.0.dev0"
