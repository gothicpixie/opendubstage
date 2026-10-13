"""opendubstage.mix.carve: music-under-voice tools for documentary and radio mixing. These are pure numpy functions with no
fixed paths and no I/O beyond an optional JSON song-map load. Every setting is a keyword parameter; the
module-level constants are only defaults.

carve(mx, dx, sr, start=, pad=)  speech-keyed dynamic EQ: the music's 1–4 kHz band dips CUT_DB while dialogue is
                                 active. It is an STFT (2048 / hop 512, sqrt-Hann WOLA) on an ABSOLUTE frame grid and
                                 is stateless. The caller hands over each block with `pad` samples of real context on
                                 each side (carve_pad()), so blocks join with no seams and match a single pass.
posts(env, activity)             frame-rate duck envelope (linear gain per 10 ms by default). The
                                 music sits at a shallower DEPTH under speech, rises to a "post" in narration gaps of
                                 2.5 s or more, and pre-dips just before the next line.
phrase_safe_end(song_map, t)     nearest section, 4-bar phrase or downbeat end to a wanted music out-point, plus a
                                 fade length, so music is not cut mid-phrase.

Why these numbers
- Speech intelligibility lives in about 1–4 kHz. The ANSI S3.5 Speech Intelligibility Index band-importance
  weights peak at roughly 1.5–3.5 kHz; consonant cues and the "presence" region sit there. That is where
  sustained strings and piano mask a voice.
- CUT_DB = -6. This is the top of the usual 3–6 dB dynamic-EQ pocket. Below about 3 dB the change in masking is
  marginal. Beyond about 6 dB, strings and piano turn audibly hollow ("phasey") each time the narrator speaks,
  which is just pumping moved into one band. The look-ahead attack and the 0.2 s hold keep the pocket open across
  the gaps between words, so it does not flutter.
- With the pocket in place, the full-band duck can be about 3 dB shallower (default DEPTH = -14, versus a typical
  -17 duck without a carve). In the band that matters, the voice gains about 3 dB of clearance
  (-14 - 6 = -20 dB at 1–4 kHz, versus -17). Below 1 kHz, where music carries its warmth and harmony and masks
  speech far less, the bed is 3 dB fuller, and the smaller full-band swing breathes ("pumps") less. Measured on a
  narration-over-orchestra excerpt, a -4.5 / -13 setting only held the 1–4 kHz masking level (+0.3 dB better),
  so the defaults trade that for a real gain in intelligibility.
- Posts sit at POST_DB = -6 relative to music-up. A post is an audible lift that marks a beat without competing
  with the next line, and the narrow -14 -> -6 swing (8 dB) suits a 1 s rise. Gaps of 8 s or more rise fully.
- Posts follow the public-radio practice of letting music "post up" in the clear between narration lines. NPR
  Training's music guidance, Transom.org scoring pieces and Jessica Abel's *Out on the Wire* (2015) all describe
  music that comes up in the clear and drops just before the next voice. I am paraphrasing these from memory and
  have not quoted or verified them here. A post shorter than about 1.5 s reads as pumping, not as a musical
  moment, so posts happen only in gaps of 2.5 s or more and never last less than `min_post` (1.5 s).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import fft as sfft

# --- carve -----------------------------------------------------------------------------------------------------------
N_FFT, HOP = 2048, 512                 # 42.7 ms frames at 48 kHz, 75 % overlap (sqrt-Hann² = Hann, sums to 2)
CUT_DB = -6.0                          # presence-band dip under speech
BAND = (1000.0, 4000.0)                # full cut inside; raised-cosine half-octave tapers outside
ATTACK, HOLD, RELEASE = 0.05, 0.20, 0.30   # s; attack is look-ahead, so the pocket is open AT the first syllable
GATE_DB = -45.0                        # dialogue 300 Hz–4 kHz frame RMS (dBFS) above which the carve engages fully
GATE_SOFT = 6.0                        # dB below GATE_DB over which it fades in (soft knee, no chatter)
# --- posts -----------------------------------------------------------------------------------------------------------
DEPTH = -14.0                          # full-band duck under speech when carve is on (vs ~-17 without carve)
POST_DB = -6.0                         # post level, relative to music-up
FR = 100                               # envelope frames per second


def carve_pad(sr, attack=ATTACK, hold=HOLD, release=RELEASE, n_fft=N_FFT):
    """Context (samples) needed each side of a block for seamless block processing: one STFT frame plus the
    control kernel's reach (look-ahead and hold+release), plus one hop of slack. Pass the same timing
    parameters you give carve()."""
    hop = n_fft // 4
    return n_fft + hop + int(math.ceil((attack + hold + release) * sr / hop)) * hop


def _window(n_fft):
    n = np.arange(n_fft)
    return np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * n / n_fft)).astype(np.float32)


def _band_weight(sr, band, n_fft):
    f = np.fft.rfftfreq(n_fft, 1.0 / sr)
    lo, hi = band
    w = np.zeros_like(f)
    w[(f >= lo) & (f <= hi)] = 1.0
    a = (f > lo / math.sqrt(2)) & (f < lo)                         # half-octave taper below
    w[a] = 0.5 - 0.5 * np.cos(np.pi * np.log2(f[a] / (lo / math.sqrt(2))) / 0.5)
    b = (f > hi) & (f < hi * math.sqrt(2))                         # half-octave taper above
    w[b] = 0.5 + 0.5 * np.cos(np.pi * np.log2(f[b] / hi) / 0.5)
    return w.astype(np.float32)


def _frames(x, j0, nfr, base, n_fft):
    """(nfr, n_fft) view of x (1-D) for absolute frames j0.. at hop n_fft/4, where x[0] is absolute sample `base`."""
    hop = n_fft // 4
    s = j0 * hop - base
    st = x.strides[0]
    return np.lib.stride_tricks.as_strided(x[s:], shape=(nfr, n_fft), strides=(hop * st, st), writeable=False)


def _amount(gate, attack_f, hold_f, rel_f):
    """Non-recursive attack/hold/release follower: amt[j] = max_k gate[j-k]·w[k]. The look-ahead ramp covers
    k in [-attack_f, 0], the hold k in (0, hold_f] and the linear release after that. It has finite support, so
    block processing with enough context gives the same result as one pass."""
    amt = gate.copy()
    nf = len(gate)
    for k in range(-attack_f, hold_f + rel_f + 1):
        if k == 0: continue
        if k < 0: w = 1.0 + k / (attack_f + 1)
        elif k <= hold_f: w = 1.0
        else: w = 1.0 - (k - hold_f) / (rel_f + 1)
        if w <= 0: continue
        if k > 0: np.maximum(amt[k:], gate[:nf - k] * w, out=amt[k:])
        else: np.maximum(amt[:nf + k], gate[-k:] * w, out=amt[:nf + k])
    return amt


def carve(mx, dx, sr, start=0, pad=0, cut_db=CUT_DB, band=BAND, gate_db=GATE_DB, gate_soft=GATE_SOFT,
          sidechain_band=(300.0, 4000.0), attack=ATTACK, hold=HOLD, release=RELEASE, n_fft=N_FFT,
          return_amount=False):
    """Speech-keyed dynamic EQ on a music block.

    mx, dx : (n, ch) or (n,) float arrays, same length, sample-aligned. dx is the dialogue sidechain (all speech
             stems, before any gain envelope). Both include `pad` samples of real context at each end (zeros where
             the programme has none).
    start  : absolute sample index of mx[0] on the timeline. Frames sit on a global grid (multiples of HOP),
             so results do not depend on how the programme is blocked.
    pad    : context samples at each end. Use 0 for a one-shot call on a whole signal, and >= carve_pad(sr, ...) for
             blocks (same attack/hold/release/n_fft).
    cut_db, band        : dip depth (dB, negative) and the fully-cut band (Hz); half-octave raised-cosine tapers outside.
    gate_db, gate_soft  : sidechain band RMS (dBFS) for a full carve; the carve fades in over gate_soft dB below it.
    attack, hold, release (s) : look-ahead ramp in, hold, and linear release of the carve amount.
    n_fft  : STFT size (hop = n_fft/4, sqrt-Hann analysis and synthesis).
    Returns the carved music for the inner range mx[pad:n-pad], float32, same channel layout. With
    return_amount=True it also returns (frame_times_s, amount 0..1) for the output frames.
    """
    mx = np.asarray(mx, np.float32); dx = np.asarray(dx, np.float32)
    one_d = mx.ndim == 1
    if one_d: mx = mx[:, None]
    if dx.ndim == 2: dx = dx.mean(1)
    n, ch = mx.shape
    assert len(dx) == n, "mx and dx must be the same length"
    HOP, N_FFT = n_fft // 4, n_fft
    need = carve_pad(sr, attack, hold, release, n_fft)
    if pad and pad < need: raise ValueError(f"pad {pad} < carve_pad(...)={need}")
    o0, o1 = start + pad, start + n - pad                       # absolute output range
    if o1 <= o0: return (mx[:0, 0] if one_d else mx[:0]).copy()
    E = need                                                    # internal zero extension (only matters at file edges)
    base = start - E
    xm = np.zeros((n + 2 * E, ch), np.float32); xm[E:E + n] = mx
    xd = np.zeros(n + 2 * E, np.float32); xd[E:E + n] = dx
    j0 = -((-base) // HOP)                                      # ceil(base / HOP)
    j1 = (base + len(xd) - N_FFT) // HOP                        # last frame fully inside the extended buffer
    win = _window(n_fft)

    # control: per-frame dialogue level in the speech band, then soft gate, then attack/hold/release
    nfr = j1 - j0 + 1
    fd = _frames(xd, j0, nfr, base, n_fft)
    f = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    sb = (f >= sidechain_band[0]) & (f <= sidechain_band[1])
    P = np.abs(sfft.rfft(fd * win, axis=1)[:, sb]) ** 2
    ms = 2.0 * P.sum(1) / (N_FFT * float((win ** 2).sum()))     # mean-square of the band within the frame
    lvl = 10 * np.log10(ms + 1e-20)
    gate = np.clip((lvl - gate_db) / gate_soft + 1.0, 0.0, 1.0).astype(np.float32)
    amt = _amount(gate, int(round(attack * sr / HOP)), int(round(hold * sr / HOP)), int(round(release * sr / HOP)))

    # frames that touch the output range
    k0 = max(j0, (o0 - N_FFT) // HOP + 1); k1 = min(j1, -((-o1) // HOP) - 1)
    a = amt[k0 - j0:k1 - j0 + 1]
    times = (np.arange(k0, k1 + 1) * HOP + N_FFT / 2) / sr
    if a.max() <= 0.0:                                          # nothing to do: bit-exact pass-through
        out = mx[pad:n - pad].copy()
        out = out[:, 0] if one_d else out
        return (out, times, a) if return_amount else out
    wband = _band_weight(sr, band, n_fft)
    G = (10.0 ** (cut_db * a[:, None] * wband[None, :] / 20.0)).astype(np.float32)   # (frames, bins)

    nk = k1 - k0 + 1
    span0 = k0 * HOP                                            # absolute start of the OLA buffer
    buf = np.zeros((nk * HOP + N_FFT, ch), np.float32)
    for c in range(ch):
        fr = _frames(np.ascontiguousarray(xm[:, c]), k0, nk, base, n_fft)
        Y = sfft.rfft(fr * win, axis=1); Y *= G
        y = (sfft.irfft(Y, n=N_FFT, axis=1) * win).astype(np.float32)
        for r in range(4):                                      # 4 phases at hop N/4 tile without overlap
            blk = y[r::4]
            s = r * HOP
            buf[s:s + blk.shape[0] * N_FFT, c] += blk.reshape(-1)
    buf *= 0.5                                                  # Σ hann over hop N/4 = 2
    out = buf[o0 - span0:o1 - span0]
    out = out[:, 0] if one_d else out
    return (out, times, a) if return_amount else out


def carve_blocks(mx, dx, sr, block, **kw):
    """Reference block driver (what a block mixer does): process mx in `block`-sample blocks with carve_pad context."""
    mx = np.asarray(mx, np.float32); dx = np.asarray(dx, np.float32)
    P = carve_pad(sr, kw.get("attack", ATTACK), kw.get("hold", HOLD), kw.get("release", RELEASE), kw.get("n_fft", N_FFT))
    n = len(mx); outs = []
    for b0 in range(0, n, block):
        b1 = min(n, b0 + block)
        sl = lambda x: np.concatenate([np.zeros((max(0, P - b0),) + x.shape[1:], np.float32),
                                       x[max(0, b0 - P):min(n, b1 + P)],
                                       np.zeros((max(0, b1 + P - n),) + x.shape[1:], np.float32)])
        outs.append(carve(sl(mx), sl(dx), sr, start=b0 - P, pad=P, **kw))
    return np.concatenate(outs)


# --- posts -----------------------------------------------------------------------------------------------------------
def _ramp(db, fr, t0, t1, L0, L1):
    """Raised-cosine (S-curve) move from L0 to L1 dB over [t0, t1) s, written into db."""
    i0, i1 = int(round(t0 * fr)), int(round(t1 * fr))
    if i1 <= i0:
        return
    t = (np.arange(i0, i1) - i0 + 0.5) / (i1 - i0)
    v = L0 + (L1 - L0) * (0.5 - 0.5 * np.cos(np.pi * t))
    a, b = max(0, i0), min(len(db), i1)
    if b > a: db[a:b] = v[a - i0:b - i0]


def _fill(db, fr, t0, t1, L):
    a, b = max(0, int(round(t0 * fr))), min(len(db), int(round(t1 * fr)))
    if b > a: db[a:b] = L


def posts(env, activity, depth=DEPTH, post_db=POST_DB, fr=FR, min_gap=2.5, full_gap=8.0, min_post=1.5,
          hang=0.15, rise=1.0, dip=0.5, lead=0.25, full_rise=2.0, full_dip=0.8, return_db=False):
    """Duck envelope with music posts: a linear gain per 1/fr s (a frame-rate envelope a block mixer interpolates).

    env      : an existing envelope array, or an int frame count; used only for its length.
    activity : speech intervals [(a, b), ...] in seconds on the same timeline (all speech that should duck music).
    Behaviour: intervals closer than `min_gap` merge into one line. Under a line, plus `hang` after it, the level
    is `depth`. A gap g with min_gap <= g < full_gap gets a post: an S-curve rise over `rise` s starting `hang`
    after the line, a hold at `post_db`, and a `dip` s S-curve back to depth ending `lead` s before the next word.
    A gap of full_gap or more rises to 0 dB (music up) over `full_rise` and drops over `full_dip`. A post window
    (rise start to dip end) shorter than `min_post` is not taken; the level stays at depth.
    If post_db <= depth (e.g. ambience at -5 dB), the gaps just stay at depth.
    """
    n = env if isinstance(env, (int, np.integer)) else len(env)
    db = np.zeros(n, np.float64)
    lines = []
    for a, b in sorted(activity):
        if lines and a - lines[-1][1] < min_gap: lines[-1][1] = max(lines[-1][1], b)
        else: lines.append([a, b])
    post = max(post_db, depth)
    prev_level = 0.0                                            # music-up before the first line
    for i, (a, b) in enumerate(lines):
        # approach: dip from the previous gap level to depth, ending `lead` before the first word
        d = dip if prev_level != 0.0 else full_dip
        _ramp(db, fr, a - lead - d, a - lead, prev_level, depth)
        _fill(db, fr, a - lead, b + hang, depth)
        nxt = lines[i + 1][0] if i + 1 < len(lines) else math.inf
        g = nxt - b
        if g >= full_gap:
            level, r, d_next = 0.0, full_rise, full_dip
        elif post > depth and (g - hang - lead) >= min_post:
            level, r, d_next = post, rise, dip
        else:
            level, r, d_next = depth, 0.0, 0.0
        if level == depth:
            _fill(db, fr, b + hang, nxt - lead, depth); prev_level = depth; continue
        t_rise = b + hang
        t_dip = nxt - lead - d_next                             # dip starts here (inf for the last line)
        r = min(r, max(0.0, t_dip - t_rise))
        _ramp(db, fr, t_rise, t_rise + r, depth, level)
        _fill(db, fr, t_rise + r, min(t_dip, n / fr + 1), level)
        prev_level = level
    return db if return_db else (10 ** (db / 20)).astype(np.float32)


# --- phrase-safe music outs ------------------------------------------------------------------------------------------
def _load_map(song_map):
    """A song map: a dict, or a path to JSON, with keys bpm, dur, downbeats [s], sections [{start, end, ...}]."""
    if isinstance(song_map, dict): return song_map
    return json.loads(Path(song_map).read_text())


def phrase_safe_end(song_map, t_end, search=6.0, bars_per_phrase=4):
    """Nearest musically safe out-point to `t_end` (seconds in SONG time, i.e. cue ss + offset).

    The search order is section boundary, then a `bars_per_phrase`-bar phrase boundary, then any downbeat, each
    within ±`search` s (the nearest downbeat is the final fallback). It returns a dict:
      end     : the boundary time (song s); let the music play through this downbeat
      fade    : fade-out length (s), starting AT `end` ("button" ring-out): one bar, clipped to 1.5–4 s
      cue_end : end + fade, the song time at which the cue should stop (fade the last `fade` s)
      kind    : "section" | "phrase" | "downbeat" | "none"
    """
    m = _load_map(song_map)
    bpm = float(m.get("bpm") or 0) or 100.0
    bar = 4 * 60.0 / bpm
    fade = float(np.clip(bar, 1.5, 4.0))
    db = np.asarray(m.get("downbeats") or [], float)
    secs = m.get("sections") or []

    def pick(cands):
        c = [x for x in cands if abs(x - t_end) <= search]
        return min(c, key=lambda x: abs(x - t_end)) if c else None

    sec_b = sorted({s["end"] for s in secs} | {s["start"] for s in secs if s["start"] > 0})
    if db.size:  # snap section boundaries to the nearest downbeat (song maps quantise loosely)
        sec_b = [float(db[np.argmin(np.abs(db - x))]) if np.min(np.abs(db - x)) < 0.6 * bar else x for x in sec_b]
    e = pick(sec_b); kind = "section"
    if e is None and db.size:
        phr = []
        starts = [s["start"] for s in secs] or [float(db[0])]
        for i, d in enumerate(db):                              # count bars from the start of the section holding d
            s0 = max([s for s in starts if s <= d + 1e-3], default=float(db[0]))
            k0 = int(np.argmin(np.abs(db - s0)))
            if (i - k0) % bars_per_phrase == 0: phr.append(float(d))
        e = pick(phr); kind = "phrase"
        if e is None:
            e = float(db[np.argmin(np.abs(db - t_end))]); kind = "downbeat"
    if e is None:
        e, kind = float(t_end), "none"
    dur = float(m.get("dur") or math.inf)
    if e + fade > dur: fade = max(0.5, dur - e)
    return {"end": round(e, 3), "fade": round(fade, 3), "cue_end": round(e + fade, 3), "kind": kind}


# --- masking proxy (used by the demo and tests) ----------------------------------------------------------------------
def band_energy_ratio_db(mus, spk, sr, intervals, band=BAND):
    """Music-to-speech energy ratio (dB) in `band`, over the given (a, b) second intervals (relative to array start)."""
    from scipy.signal import butter, sosfilt
    sos = butter(4, band, btype="band", fs=sr, output="sos")
    m = mus.mean(1) if mus.ndim == 2 else mus
    s = spk.mean(1) if spk.ndim == 2 else spk
    mb, sbf = sosfilt(sos, m), sosfilt(sos, s)
    em = es = 0.0
    for a, b in intervals:
        i0, i1 = max(0, int(a * sr)), min(len(m), int(b * sr))
        em += float((mb[i0:i1] ** 2).sum()); es += float((sbf[i0:i1] ** 2).sum())
    return 10 * math.log10((em + 1e-20) / (es + 1e-20))


# --- command line ----------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    import argparse
    import soundfile as sf
    ap = argparse.ArgumentParser(prog="opendubstage carve",
                                 description="Speech-keyed dynamic EQ: dip the music's 1-4 kHz band while dialogue "
                                             "is active. MUSIC and DIALOGUE must share a sample rate and start time.")
    ap.add_argument("music"); ap.add_argument("dialogue"); ap.add_argument("out")
    ap.add_argument("--cut-db", type=float, default=CUT_DB)
    ap.add_argument("--band", type=float, nargs=2, default=BAND, metavar=("LO", "HI"))
    ap.add_argument("--gate-db", type=float, default=GATE_DB)
    ap.add_argument("--block-s", type=float, default=60.0, help="processing block length (bounded memory)")
    a = ap.parse_args(argv)
    mx, sr = sf.read(a.music, dtype="float32", always_2d=False)
    dx, sr2 = sf.read(a.dialogue, dtype="float32", always_2d=False)
    if sr != sr2:
        raise SystemExit(f"sample rates differ: {sr} vs {sr2}")
    if dx.ndim == 2:
        dx = dx.mean(1)
    dx = dx[:len(mx)] if len(dx) >= len(mx) else np.pad(dx, (0, len(mx) - len(dx)))
    y = carve_blocks(mx, dx, sr, int(a.block_s * sr), cut_db=a.cut_db, band=tuple(a.band), gate_db=a.gate_db)
    sf.write(a.out, y, sr, subtype="FLOAT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
