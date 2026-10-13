"""On-device picture analysis with Apple's Vision framework (macOS only, no model download).

`VNClassifyImageRequest` gives scene/object labels; `VNDetectFaceRectanglesRequest`
tells us when a face dominates the frame (a portrait: no spot, the face is the
subject). The labels are folded into the caption text and passed through the
same precision-first keyword matcher, so both paths share one taxonomy.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .analysis import KeywordAnalyzer
from .model import SceneAnalysis, Still


@dataclass
class VisionResult:
    labels: list[tuple[str, float]] = field(default_factory=list)
    largest_face_area: float = 0.0   # fraction of the frame


def run_vision(path: str, min_conf: float = 0.25) -> VisionResult:  # pragma: no cover - needs macOS
    import Vision
    from Foundation import NSURL

    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(NSURL.fileURLWithPath_(path), {})
    cls = Vision.VNClassifyImageRequest.alloc().init()
    faces = Vision.VNDetectFaceRectanglesRequest.alloc().init()
    ok, err = handler.performRequests_error_([cls, faces], None)
    if not ok:
        raise RuntimeError(f"Vision failed on {path}: {err}")
    labels = sorted(((o.identifier(), float(o.confidence())) for o in cls.results() if o.confidence() >= min_conf),
                    key=lambda x: -x[1])
    area = 0.0
    for f in faces.results() or []:
        bb = f.boundingBox()
        area = max(area, float(bb.size.width * bb.size.height))
    return VisionResult(labels, area)


def labels_to_text(labels: list[tuple[str, float]]) -> str:
    return " ".join(lbl.replace("_", " ") for lbl, _ in labels)


@dataclass
class VisionAnalyzer:
    """Caption (if any) + Vision labels -> SceneAnalysis. Portrait detection by face size."""

    min_conf: float = 0.25
    portrait_face_area: float = 0.03   # a face covering >= 3% of the frame makes it a portrait
    base: KeywordAnalyzer = field(default_factory=KeywordAnalyzer)
    runner: object = None              # injectable for tests: callable(path, min_conf) -> VisionResult

    def analyze(self, still: Still) -> SceneAnalysis:
        if not still.path:
            return self.base.analyze(still)
        vr = (self.runner or run_vision)(still.path, self.min_conf)
        tags = list(still.tags)
        if vr.largest_face_area >= self.portrait_face_area:
            tags.append("portrait")
        caption = (still.caption + " " + labels_to_text(vr.labels)).strip()
        out = self.base.analyze(replace(still, caption=caption, tags=tags))
        for c in out.candidates:
            c.source = "vision" if not still.caption else "caption+vision"
        return out
