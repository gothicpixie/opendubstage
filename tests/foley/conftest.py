from __future__ import annotations

import pytest

from foley_helpers import mk_still

from opendubstage.foley.analysis import KeywordAnalyzer
from opendubstage.foley.model import NarrationSegment, Sequence


@pytest.fixture
def analyzer():
    return KeywordAnalyzer()


@pytest.fixture
def street_sequence() -> Sequence:
    caps = [
        "a busy main street with cars parked along the sidewalk",
        "portrait of a woman looking at the camera",
        "a freight train passing a small station across open country",
        "two men walking down a gravel road",
        "a boat landing on a lake with rowboats at the dock",
        "people raising wine glasses in a toast at a long table",
    ]
    stills = [mk_still(i, c, start=i * 6.0) for i, c in enumerate(caps)]
    return Sequence(stills=stills, narration=[NarrationSegment(10.0, 16.0)], seed="t")
