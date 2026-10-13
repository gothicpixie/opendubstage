"""Sound classes, bed selectors and no-sound classes.

This is deliberately small and conservative. Real photo collections have far
fewer spot-worthy subjects than one expects (in the collection that prompted
this project, roughly one still in five or six); the taxonomy should make
silence the default, not invent sounds.

Keyword patterns are matched against a lower-cased caption plus tags. They are
the fallback analyser; a vision model can emit the same class keys directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .model import Layer
from .prompts import PromptParts, build_prompt


@dataclass(frozen=True)
class SoundClass:
    key: str
    layer: Layer
    parts: PromptParts
    keywords: tuple[str, ...]
    dur: float = 2.0
    exclude: tuple[str, ...] = field(default_factory=tuple)

    @property
    def prompt(self) -> str:
        return build_prompt(self.parts, self.layer.value)


P = PromptParts
SOUND_CLASSES: dict[str, SoundClass] = {c.key: c for c in [
    SoundClass("glass_clink", Layer.SPOT, P("single clink of two wine glasses", "", "at a table outdoors"),
               (r"\b(wine ?glass(es)?|glasses raised|champagne|toast(ing)?|clinking)\b",), 1.5),
    SoundClass("steps_leaves", Layer.SPOT, P("three soft footsteps", "on dry leaves and pine needles", "in a forest"),
               (r"\b(walk\w*|stroll\w*|hik\w*)\b.*\b(forest|woods|trail|trees|pines?)\b",
                r"\b(forest|woods|trail|pines?)\b.*\b(walk\w*|stroll\w*|hik\w*)\b"), 2.0),
    SoundClass("steps_gravel", Layer.SPOT, P("a few footsteps", "on gravel", "outdoors"),
               (r"\b(walk\w*|stroll\w*|arriv\w*)\b.*\b(road|path|gravel|driveway|street)\b",
                r"\b(road|path|gravel|driveway)\b.*\b(walk\w*|stroll\w*)\b"), 2.0),
    SoundClass("steps_wood", Layer.SPOT, P("a few footsteps", "on a wooden floor", "indoors"),
               (r"\b(walk\w*)\b.*\b(porch|hall|aisle|deck|boardwalk)\b",), 2.0),
    SoundClass("car_far", Layer.SPOT, P("a car passing slowly", "", "on a quiet road", "distant"),
               (r"\b(car|cars|truck|automobile|traffic|highway|motorcar)\b",), 3.0),
    SoundClass("horse_cart", Layer.SPOT, P("horse hooves and a wooden cart", "on a dirt road", "", "distant"),
               (r"\b(horse[- ]drawn|wagon|buggy|carriage|cart)\b",), 3.0),
    SoundClass("camera_shutter", Layer.SPOT, P("one mechanical camera shutter click", "", "outdoors"),
               (r"\b(photographer|taking a (photo|picture)|holding a camera)\b",), 1.0),
    SoundClass("chair_creak", Layer.SPOT, P("a wooden chair creaking once", "", "on a porch"),
               (r"\b(rocking chair|porch swing|sitting on (a|the) (porch|bench))\b",), 1.5),
    SoundClass("door_screen", Layer.SPOT, P("an old screen door closing softly with a spring", "", "", "medium distance"),
               (r"\b(screen door|doorway|front door)\b",), 2.0),
    SoundClass("typewriter", Layer.SPOT, P("a few keystrokes on a manual typewriter", "", "in a quiet office"),
               (r"\btypewriter\b",), 2.0),
    SoundClass("train_far", Layer.TEXTURE, P("a steam train passing", "", "across open country", "far away"),
               (r"\b(train|locomotive|railroad|railway)\b",), 12.0),
    SoundClass("laughter_far", Layer.TEXTURE, P("soft laughter of a small group", "", "outdoors", "far away, unintelligible"),
               (r"\blaugh\w*\b",), 10.0),
    SoundClass("crowd_murmur", Layer.TEXTURE, P("murmur of a crowd", "", "in a large room", "far away, no clear words"),
               (r"\b(crowd|audience|gathering|guests|reception|meeting)\b",), 15.0),
    SoundClass("kids_far", Layer.TEXTURE, P("children playing", "", "in a field", "far away, no clear words"),
               (r"\b(children|kids|boys|girls)\b.*\b(play\w*|run\w*)\b",), 10.0),
    SoundClass("water_lap", Layer.TEXTURE, P("gentle water lapping", "against a wooden dock", "", "close"),
               (r"\b(lake|dock|pier|shore|boat)\b",), 15.0),
    SoundClass("creek", Layer.TEXTURE, P("a small creek trickling", "over stones", "", "medium distance"),
               (r"\b(creek|stream|brook|river)\b",), 15.0),
    SoundClass("birds", Layer.TEXTURE, P("a few songbirds", "", "in trees", "far away"),
               (r"\bbirds?\b",), 12.0),
    SoundClass("fire_crackle", Layer.TEXTURE, P("a small wood fire crackling", "", "outdoors", "medium distance"),
               (r"\b(campfire|bonfire|fireplace|fire pit|hearth)\b",), 12.0),
    SoundClass("machinery_far", Layer.TEXTURE, P("a factory machine running steadily", "", "in a large hall", "far away"),
               (r"\b(factory|mill|machinery|assembly line|workshop|forge)\b",), 15.0),
]}

# Hand-requested transitions only (Still.transition). Never chosen from keywords.
TRANSITION_CLASSES: dict[str, SoundClass] = {c.key: c for c in [
    SoundClass("door_close", Layer.TRANSITION, P("a wooden door closing", "", "in a quiet house"), (), 1.5),
    SoundClass("shutter_cut", Layer.TRANSITION, P("one mechanical camera shutter click", "", "indoors", "close"), (), 0.6),
    SoundClass("car_pass", Layer.TRANSITION, P("a car passing by", "", "on a country road"), (), 3.0),
]}

# Beds: (key, patterns). First match wins; time of day picks a variant.
BED_SELECTORS: list[tuple[str, tuple[str, ...]]] = [
    ("city_street", (r"\b(street|city|downtown|avenue|sidewalk|storefront)\b",)),
    ("interior_room", (r"\b(kitchen|bedroom|living room|interior|indoors|cabin|office|parlou?r)\b",)),
    ("crowd_hall", (r"\b(reception|banquet|ballroom|hall|church)\b",)),
    ("shore", (r"\b(beach|ocean|sea|coast|surf)\b",)),
    ("water_edge", (r"\b(lake|river|creek|pond)\b",)),
    ("mountain_wind", (r"\b(mountain|ridge|summit|snow|peak|overlook)\b",)),
    ("forest", (r"\b(forest|woods|pines?|trees)\b",)),
    ("field", (r"\b(field|farm|meadow|prairie|pasture|barn)\b",)),
]
DEFAULT_BED = "room_tone"

BED_PROMPTS: dict[str, str] = {
    "city_street": "distant city street traffic and footsteps, no horns",
    "interior_room": "quiet room tone, faint clock, wood settling",
    "crowd_hall": "large hall, distant murmur, no clear words",
    "shore": "small waves on a beach, light wind",
    "water_edge": "calm water edge, faint lapping, light breeze",
    "mountain_wind": "high mountain wind, thin air, no birds",
    "forest": "light wind in tall trees, a few distant songbirds",
    "field": "open field, light breeze in grass, distant insects",
    "room_tone": "neutral quiet room tone",
}
TOD_SUFFIX = {"dusk": ", dusk, crickets beginning", "night": ", night, crickets, no birds"}

# Never a spot or texture: the picture has no sound source, or the face is the subject.
NO_SOUND_PATTERNS: list[tuple[str, str]] = [
    (r"\b(portrait|headshot|close[- ]?up|selfie|posed|posing|looking at (the )?camera)\b", "portrait or close-up"),
    (r"\b(detail of|still life|document|letter|certificate|map|newspaper clipping)\b", "detail or document"),
    (r"\b(blurr?y|out of focus|overexposed|underexposed|very dark)\b", "technically weak picture"),
]
NO_SOUND_TAGS = {"no_sound", "portrait", "detail", "document"}
QUALITY_FLAGS = {"blurry", "dark", "overexposed", "underexposed"}

# Rooms used for worldizing (post chain); keyed by bed.
ROOM_FOR_BED = {
    "forest": "outdoor_trees", "mountain_wind": "outdoor_open", "field": "outdoor_open",
    "shore": "outdoor_open", "water_edge": "outdoor_open", "city_street": "street",
    "interior_room": "small_room", "crowd_hall": "hall", "room_tone": "small_room",
}


def bed_prompt(bed: str, time_of_day: str | None = None) -> str:
    base = BED_PROMPTS.get(bed, BED_PROMPTS[DEFAULT_BED])
    return base + TOD_SUFFIX.get(time_of_day or "", "") + ", no music, no voices, ambience loop"
