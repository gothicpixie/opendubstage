# Transcription check of every narration cut

Purpose: catch clipped first or last words, fades that eat a syllable, wrong source offsets
and mid-sentence cuts, by listening to the **mix** with an ASR model and comparing with the
word-timed transcript. It is binding: a failing cue blocks the build.

## Inputs

- The finished mix (or the radio cut), and the cue list: for each narration cue its source
  in-point `ss`, duration `dur`, and its film time `at`.
- A word-timed transcript of the source with optional speaker labels (`w`, `t`, `e`, `spk`)
  and audio-event tags (`[laughs]` etc.).
- An ASR CLI (e.g. whisper.cpp `whisper-cli` with a large-v3-turbo model, `-mc 0` so long
  audio does not drop lines; 16 kHz mono input).

## Procedure

```python
import re, difflib, subprocess

FILLERS = {"um", "uh", "er", "ah", "hmm", "mm"}

def clean(text):
    """Words for comparison: lower-case, no punctuation, no fillers, no false starts ("bea-"),
    no stutter repeats. ASR smooths disfluencies; the transcript keeps them; neither is a clipped word."""
    toks = [t for t in re.sub(r"[^a-z'\- ]", " ", text.lower()).split()
            if t not in FILLERS and not t.endswith("-")]
    toks = [t.strip("-") for t in toks if t.strip("-")]
    return [t for i, t in enumerate(toks) if i == 0 or t != toks[i - 1]]

def hear(mix, t0, dur, tmpdir):
    wav = f"{tmpdir}/a.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.3f}", "-t", f"{dur:.3f}", "-i", mix,
                    "-ar", "16000", "-ac", "1", wav], check=True)
    subprocess.run(["whisper-cli", "-m", MODEL, "-f", wav, "-t", "8", "-mc", "0", "-otxt", "-of", f"{tmpdir}/a"],
                   capture_output=True)
    return clean(open(f"{tmpdir}/a.txt").read())

def check_cue(words_in_cue, got):
    """words_in_cue: transcript words whose times fall inside [ss, ss+dur]."""
    want_all = clean(" ".join(w["w"] for w in words_in_cue))
    # Cross-talk is optional: ASR drops the other speaker. Diarisation labels are unreliable,
    # so build the main-speaker expectation too and pass if EITHER matches.
    main = max(set(w.get("spk") for w in words_in_cue), key=[w.get("spk") for w in words_in_cue].count)
    want_main = clean(" ".join(w["w"] for w in words_in_cue if w.get("spk") == main))
    r = lambda a, b: difflib.SequenceMatcher(None, a, b).ratio()
    head = max(r(want_main[:6], got[:8]), r(want_all[:6], got[:8]))
    tail = max(r(want_main[-6:], got[-8:]), r(want_all[-6:], got[-8:]))
    return head >= 0.5 and tail >= 0.5, head, tail
```

For each narration cue: `got = hear(mix, at, dur)`; `ok, head, tail = check_cue(words, got)`;
print `id ok/BAD head tail "first five … last five"`; append a problem for every BAD cue with
what was heard and what was expected. A real clip fails both comparisons; cross-talk or a
mislabelled speaker fails only one.

For live scenes, transcribe the whole scene once and check that each key phrase is present
with a fuzzy match ≥ 0.75 over a sliding window of the phrase's length (so "dearest friends
and family" matches "dear friends and family").

## Why these choices

- **Every cue, not a sample.** In practice a 12-cue sample passed while a model reviewer heard dangling "and--" endings elsewhere.
- **Out of the mix, not the source.** The failure modes (fade placement, offset errors, a bed masking a syllable) only exist in the mix.
- **Normalise disfluencies.** A kept stammer ("bea- beautiful") is a feature of the recording (Third Coast / Peabody pattern), not a clipped word; without normalisation it fails the head check.
- **Either comparison passes.** Diarisation on a two-person recording merges and splits labels; requiring the main-speaker match alone produced false failures, requiring the all-speaker match alone failed whenever ASR dropped an interjection.
- **Thresholds.** 0.5 on six words tolerates one ASR substitution and one dropped function word; a clipped first or last word reliably drives the ratio below 0.4 in practice.

## Cost

One ASR call per cue (a few seconds each on a laptop); a 130-cue film takes under ten minutes.
Run it in the audio gate before the picture render, alongside `opendubstage qc`.
