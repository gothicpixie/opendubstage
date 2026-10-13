"""specrepair: an open, RX-style spectral repair for speech recordings (numpy / scipy / soundfile only).

It does two things:

* Replace (room-tone fill). This is for a non-verbal event in a pause, such as a cough, throat clear, sneeze or
  handling noise. The event's boundaries are refined from frame energy against the local noise floor, inside the
  gap the protected intervals (words, sounds you want to keep) leave. Each loud cluster is then replaced by room
  tone from the nearest quiet, non-protected stretches of the same recording. The tone is spliced with
  equal-power crossfades, EQ'd to the local noise spectrum (1/3-oct smoothed, clamped to +/-6 dB) and
  level-matched. This is roughly RX "Spectral Repair > Replace" plus "Ambience Match".
* Attenuate (spectral). This is for an event that overlaps speech, or for a low-frequency thump. In the STFT,
  each bin's reference magnitude is taken from the frames around the region, either as a median or as a linear
  interpolation between the left and right context medians. Bins that exceed the reference are pulled down to
  it, optionally only below `fmax`, and the original phase is kept. This is roughly RX "Spectral Repair >
  Attenuate".

Samples outside the patched spans are never modified. The file-level API (`repair_file`) streams integer PCM, so
the output is bit-exact to the input outside the regions it reports.

Times are in seconds. Intervals are (start, end) tuples.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter, uniform_filter

FR, HOP = 0.020, 0.010  # energy-analysis frame / hop (s)


@dataclass
class Params:
    guard: float = 0.010          # never come closer than this to a protected interval (s)
    xfade: float = 0.030          # equal-power crossfade length (s)
    search: float = 0.5           # widen the event's nominal span by this much when refining (s)
    active_db: float = 6.0        # frame counts as active above floor + this
    peak_db: float = 10.0         # a cluster is an event only if it peaks above floor + this
    merge_gap: float = 0.25       # merge active runs closer than this (s)
    context: float = 30.0         # room-tone search radius (s)
    tone_max_db: float = 3.0      # tone candidates must be within floor + this
    tone_min_len: float = 0.25    # shortest usable tone stretch (s)
    eq_clamp_db: float = 6.0
    # thumps
    thump_band: tuple = (20.0, 150.0)
    thump_jump_db: float = 12.0   # band energy over its running median
    thump_median_s: float = 2.0
    thump_max_len: float = 0.5
    thump_rise_db: float = 9.0    # sharp onset requirement
    thump_fmax: float = 250.0     # attenuate only bins up to this
    thump_lf_over_mid_db: float = -3.0  # at the peak, LF band must be within this of the 300-3000 Hz band
                                         # (speech is mid-heavy; a thump is LF-heavy) -> keeps voiced speech out


@dataclass
class Patch:
    """A replacement for samples [start_n, start_n + len(new)) of the source, plus a report dict."""
    start_n: int
    new: np.ndarray
    report: dict = field(default_factory=dict)


# ---------------------------------------------------------------- small helpers
def level_db(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(10 * np.log10(np.mean(x ** 2) + 1e-20)) if len(x) else float("nan")


def frame_db(x, sr):
    n, h = int(FR * sr), int(HOP * sr)
    if len(x) < n:
        return np.array([level_db(x)])
    fr = np.lib.stride_tricks.sliding_window_view(x, n)[::h]
    return 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-20)


def band_db(x, sr, lo, hi, nfft=4096) -> float:
    if len(x) < 256:
        return float("nan")
    f, p = signal.welch(x, sr, nperseg=min(nfft, len(x)))
    s = (f >= lo) & (f < hi)
    return float(10 * np.log10(p[s].sum() + 1e-20))


def runs(mask):
    d = np.diff(np.concatenate([[0], np.asarray(mask).astype(int), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def frames_in(n_frames, t0, ivs, pad=0.0):
    """Boolean mask over energy frames (frame i centred at t0 + i*HOP + FR/2) inside any interval (+pad)."""
    tc = t0 + np.arange(n_frames) * HOP + FR / 2
    m = np.zeros(n_frames, bool)
    for a, b in ivs:
        m |= (tc >= a - pad) & (tc <= b + pad)
    return m


def merge_intervals(ivs, gap=0.0):
    out = []
    for a, b in sorted(ivs):
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(v) for v in out]


def eqp_concat(pieces, xf):
    """Concatenate arrays with equal-power (sin/cos) crossfades of xf samples."""
    out = pieces[0].copy()
    for p in pieces[1:]:
        k = min(xf, len(out) // 2, len(p) // 2)
        if k < 8:
            out = np.concatenate([out, p]); continue
        t = np.linspace(0, np.pi / 2, k)
        out = np.concatenate([out[:-k], out[-k:] * np.cos(t) + p[:k] * np.sin(t), p[k:]])
    return out


def eq_match(tone, ref, sr, clamp_db=6.0, nfft=2048):
    """EQ `tone` toward the long-term spectrum of `ref` (~1/3-oct smoothed, clamped), then level-match it (+/-6 dB)."""
    def lvl(y):
        g = 10 ** ((level_db(ref) - level_db(y)) / 20)
        return y * min(max(g, 0.5), 2.0)
    if len(ref) < nfft or len(tone) < nfft:
        return lvl(tone) if len(ref) else tone
    f, pr = signal.welch(ref, sr, nperseg=nfft)
    _, pt = signal.welch(tone, sr, nperseg=nfft)
    r = 10 * np.log10((pr + 1e-20) / (pt + 1e-20))
    sm = np.empty_like(r)
    for i, fi in enumerate(f):
        sel = (f >= fi / 2 ** (1 / 6)) & (f <= fi * 2 ** (1 / 6)) if fi > 0 else (f <= 30)
        sm[i] = r[sel].mean()
    sm = np.clip(sm, -clamp_db, clamp_db)
    nov = nfft * 3 // 4
    _, _, Z = signal.stft(tone, sr, nperseg=nfft, noverlap=nov)
    _, y = signal.istft(Z * (10 ** (sm / 20))[:, None], sr, nperseg=nfft, noverlap=nov)
    y = np.pad(y[:len(tone)], (0, max(0, len(tone) - len(y))))
    return lvl(y)


# ---------------------------------------------------------------- Attenuate
def spectral_attenuate(x, sr, a, b, fmax=None, nfft=2048, hop=256, ctx=0.5, thresh_db=0.0, mode="interp"):
    """Attenuate x[a:b] (sample indices) to the spectral level of its surroundings.

    mode='interp': per-bin reference interpolated linearly from the left-context median to the right-context median
    (closest to RX Attenuate). mode='median': per-bin median of all context frames (good for short transients).
    Only bins above reference*10^(thresh_db/20) are reduced; bins above fmax are untouched; the phase is kept.
    The result is spliced over [a, b] with 10 ms linear ramps just outside; nothing else changes. Returns a new array.
    """
    c0, c1 = max(0, a - int(ctx * sr)), min(len(x), b + int(ctx * sr))
    seg = x[c0:c1]
    nov = nfft - hop
    f, tt, Z = signal.stft(seg, sr, nperseg=nfft, noverlap=nov, boundary="even", padded=True)
    tabs = c0 + tt * sr
    ev = (tabs >= a - nfft / 2) & (tabs <= b + nfft / 2)
    L, R = (~ev) & (tabs < a), (~ev) & (tabs > b)
    if L.sum() < 2 and R.sum() < 2 or not ev.any():
        return x.copy()
    M = np.abs(Z)
    idx = np.flatnonzero(ev)
    if mode == "median":
        ref = np.repeat(np.median(M[:, ~ev], axis=1)[:, None], len(idx), axis=1)
    else:
        mL = np.median(M[:, L], axis=1) if L.sum() >= 2 else np.median(M[:, R], axis=1)
        mR = np.median(M[:, R], axis=1) if R.sum() >= 2 else mL
        w = np.linspace(0, 1, len(idx)) if len(idx) > 1 else np.array([0.5])
        ref = mL[:, None] * (1 - w) + mR[:, None] * w
    ref = ref * 10 ** (thresh_db / 20)
    g = np.minimum(1.0, ref / (M[:, idx] + 1e-20))
    if fmax is not None:
        g[f > fmax, :] = 1.0
    gd = uniform_filter(20 * np.log10(g + 1e-9), size=(3, 3), mode="nearest")   # tame musical noise
    g = np.minimum(1.0, 10 ** (gd / 20))
    if fmax is not None:
        g[f > fmax * 1.15, :] = 1.0
    Z2 = Z.copy()
    Z2[:, idx] *= g
    _, y = signal.istft(Z2, sr, nperseg=nfft, noverlap=nov, boundary=True)
    y = y[:len(seg)]
    out = x.copy()
    r = int(0.010 * sr)
    s0, s1 = max(c0, a - r), min(c1, b + r)
    w = np.ones(s1 - s0)
    k0, k1 = a - s0, s1 - b
    if k0 > 0: w[:k0] = np.linspace(0, 1, k0)
    if k1 > 0: w[-k1:] = np.linspace(1, 0, k1)
    out[s0:s1] = x[s0:s1] * (1 - w) + y[s0 - c0:s1 - c0] * w
    return out


# ---------------------------------------------------------------- Replace (room tone)
def noise_floor(fdb, t0, protected, excluded):
    m = frames_in(len(fdb), t0, protected, pad=0.15) | frames_in(len(fdb), t0, excluded, pad=0.3)
    pool = fdb[~m]
    return float(np.percentile(pool if len(pool) > 50 else fdb, 25))


def gap_around(protected, t, e):
    prev = max([b for a, b in protected if b <= t + 0.02], default=-np.inf)
    nxt = min([a for a, b in protected if a >= e - 0.02], default=np.inf)
    return prev, nxt


def repair_event(x, sr, t0, span, protected=(), excluded=(), p: Params = Params()):
    """Repair one event whose nominal span is `span` (s, absolute) in x (which starts at absolute time t0; give it
    >= p.context of surrounding audio). `protected`: intervals that must not be touched (words, kept sounds).
    `excluded`: other event spans not to borrow room tone from. Returns a list of Patch (start_n relative to x)."""
    t, e = span
    prev, nxt = gap_around(protected, t, e)
    overlaps = [(a, b) for a, b in protected if a < e - 0.02 and b > t + 0.02]
    fdb = frame_db(x, sr)
    floor = noise_floor(fdb, t0, protected, list(excluded) + [span])
    base = dict(span=[round(t, 4), round(e, 4)], gap=[round(prev, 4), round(nxt, 4)], floor_db=round(floor, 2))
    if overlaps:
        a, b = int(round((t - t0) * sr)), int(round((e - t0) * sr))
        y = spectral_attenuate(x, sr, a, b, nfft=2048, hop=256, ctx=0.3, thresh_db=1.0, mode="interp")
        r = int(0.010 * sr)
        rep = dict(base, method="spectral_attenuate", start=round(t0 + (a - r) / sr, 4), end=round(t0 + (b + r) / sr, 4),
                   event_window=[round(t, 4), round(e, 4)], before_db=round(level_db(x[a:b]), 2),
                   after_db=round(level_db(y[a:b]), 2), overlapped=[list(o) for o in overlaps])
        rep["reduction_db"] = round(rep["before_db"] - rep["after_db"], 2)
        return [Patch(a - r, y[a - r:b + r], rep)]
    A0, A1 = max(prev + p.guard, t - p.search, t0), min(nxt - p.guard, e + p.search, t0 + len(x) / sr)
    if A1 - A0 < 0.06:
        return [Patch(0, np.zeros(0), dict(base, skipped="gap too short"))]
    i0 = int(np.ceil((A0 - t0) / HOP)); i1 = int((A1 - t0 - FR) / HOP)
    sub = fdb[i0:i1 + 1]
    cl = []
    for r0, r1 in runs(sub > floor + p.active_db):
        if cl and (r0 - cl[-1][1]) * HOP < p.merge_gap:
            cl[-1][1] = r1
        else:
            cl.append([r0, r1])
    cl = [(r0, r1) for r0, r1 in cl if sub[r0:r1].max() >= floor + p.peak_db]
    # the +/-search widening refines an event's edges; a cluster lying wholly outside the nominal span is more
    # likely a word tail / breath than the event, so leave it alone
    tol = 0.05
    cl = [(r0, r1) for r0, r1 in cl
          if t0 + (i0 + r0) * HOP <= e + tol and t0 + (i0 + r1 - 1) * HOP + FR >= t - tol]
    if not cl:
        return [Patch(0, np.zeros(0), dict(base, skipped=f"no energy above floor+{p.peak_db:g} dB in gap"))]
    out = []
    for r0, r1 in cl:
        a_s = t0 + (i0 + r0) * HOP
        b_s = t0 + (i0 + r1 - 1) * HOP + FR
        s0, s1 = max(A0, a_s - p.xfade), min(A1, b_s + p.xfade)
        out.append(fill_room_tone(x, sr, t0, s0, s1, (a_s, b_s), fdb, floor, protected,
                                  list(excluded) + [span], (A0, A1), p, dict(base)))
    return out


def fill_room_tone(x, sr, t0, s0, s1, active, fdb, floor, protected, excluded, gap, p, rep):
    n0, n1 = int(round((s0 - t0) * sr)), int(round((s1 - t0) * sr))
    need, xf = n1 - n0, int(p.xfade * sr)
    busy = (frames_in(len(fdb), t0, protected, pad=0.15) | frames_in(len(fdb), t0, excluded, pad=0.3)
            | frames_in(len(fdb), t0, [(gap[0] - 0.2, gap[1] + 0.2)]))
    ok = (~busy) & (fdb <= floor + p.tone_max_db) & (fdb >= floor - 12)
    cand = [(r0 + 2, r1 - 2) for r0, r1 in runs(ok) if (r1 - r0) * HOP >= p.tone_min_len]
    centre = (s0 + s1) / 2
    cand.sort(key=lambda r: abs(t0 + (r[0] + r[1]) / 2 * HOP - centre))
    sl = lambda r0, r1: x[int(r0 * HOP * sr):int((r1 * HOP + FR) * sr)]
    pieces, got = [], 0
    for r0, r1 in cand:
        pieces.append(sl(r0, r1)); got += len(pieces[-1]) - xf
        if got >= need + 2 * xf:
            break
    if not pieces:
        return Patch(0, np.zeros(0), dict(rep, skipped="no room tone found"))
    tone = eqp_concat(pieces, xf)
    while len(tone) < need:
        tone = eqp_concat([tone, tone], xf)
    tone = tone[:need]
    loc = frames_in(len(fdb), t0, [(s0 - 3, s1 + 3)]) & (~busy) & (fdb <= floor + 6)
    ref = np.concatenate([sl(r0, r1) for r0, r1 in runs(loc)] or [np.zeros(0)])
    if len(ref) < sr * 0.2:
        ref = np.concatenate(pieces)
    tone = eq_match(tone, ref, sr, p.eq_clamp_db)
    # equal-power fades; where the event already runs at the gap edge (no room for a full fade before it),
    # use a short 8 ms fade so the event does not leak through the crossfade
    kmax = max(1, min(xf, need // 4)); short = max(1, min(int(0.008 * sr), kmax))
    kl = short if active[0] - s0 < p.xfade * 0.9 else kmax
    kr = short if s1 - active[1] < p.xfade * 0.9 else kmax
    w_in, w_out = np.ones(need), np.zeros(need)
    tl, tr = np.linspace(0, np.pi / 2, kl), np.linspace(0, np.pi / 2, kr)
    w_in[:kl], w_out[:kl] = np.sin(tl), np.cos(tl)
    w_in[-kr:], w_out[-kr:] = np.cos(tr), np.sin(tr)
    y = x[n0:n1] * w_out + tone * w_in
    ai, bi = int(round((active[0] - t0) * sr)) - n0, int(round((active[1] - t0) * sr)) - n0
    before, after = level_db(x[n0:n1][ai:bi]), level_db(y[ai:bi])
    rep.update(method="room_tone", start=round(s0, 4), end=round(s1, 4),
               event_window=[round(active[0], 4), round(active[1], 4)], tone_pieces=len(pieces),
               before_db=round(before, 2), after_db=round(after, 2), reduction_db=round(before - after, 2))
    return Patch(n0, y, rep)


# ---------------------------------------------------------------- thumps
def detect_thumps(x, sr, p: Params = Params()):
    """Short low-frequency transients: band energy jumps > thump_jump_db over its running median with a sharp
    onset and length <= thump_max_len. Returns [(a_s, b_s, peak_excess_db)] relative to x."""
    nfft, hop = 4096, int(0.010 * sr)
    f, tt, Z = signal.stft(x, sr, nperseg=nfft, noverlap=nfft - hop, boundary="even", padded=True)
    lo, hi = p.thump_band
    lf = 10 * np.log10((np.abs(Z[(f >= lo) & (f < hi)]) ** 2).sum(0) + 1e-20)
    mid = 10 * np.log10((np.abs(Z[(f >= 300) & (f < 3000)]) ** 2).sum(0) + 1e-20)
    med = median_filter(lf, size=int(p.thump_median_s / 0.010) | 1, mode="nearest")
    ex = lf - med
    out = []
    for r0, r1 in runs(ex > p.thump_jump_db):
        if (r1 - r0) * hop / sr > p.thump_max_len:
            continue
        pk = r0 + int(np.argmax(ex[r0:r1]))
        if lf[pk] - lf[max(0, r0 - 5)] < p.thump_rise_db:
            continue
        if lf[pk] - mid[pk] < p.thump_lf_over_mid_db:
            continue
        out.append([max(0.0, tt[r0] - 0.03 - nfft / 4 / sr), min(len(x) / sr, tt[r1 - 1] + 0.08), float(ex[pk])])
    merged = []
    for a, b, pk in sorted(out):
        if merged and a <= merged[-1][1] + 0.05:
            merged[-1][1] = max(merged[-1][1], b); merged[-1][2] = max(merged[-1][2], pk)
        else:
            merged.append([a, b, pk])
    return [tuple(m) for m in merged]


def repair_thumps(x, sr, t0=0.0, avoid=(), p: Params = Params(), protected=()):
    """Detect and attenuate thumps in x (bins <= p.thump_fmax). Returns (y, [Patch]) with patches relative to x."""
    y = x.copy()
    patches = []
    for a, b, pk in detect_thumps(x, sr, p):
        A, B = t0 + a, t0 + b
        if any(A < q1 + 0.05 and B > q0 - 0.05 for q0, q1 in avoid):
            continue
        ia, ib = int(round(a * sr)), int(round(b * sr))
        z = spectral_attenuate(y, sr, ia, ib, fmax=p.thump_fmax, nfft=4096, hop=512, ctx=0.5, mode="median")
        r = int(0.010 * sr)
        s0, s1 = max(0, ia - r), min(len(y), ib + r)
        lfb, lfa = band_db(y[ia:ib], sr, *p.thump_band), band_db(z[ia:ib], sr, *p.thump_band)
        hfb, hfa = band_db(y[ia:ib], sr, 300, sr / 2), band_db(z[ia:ib], sr, 300, sr / 2)
        rep = dict(method=f"spectral_attenuate_lf<={p.thump_fmax:g}Hz", start=round(A - r / sr, 4), end=round(B + r / sr, 4),
                   peak_excess_db=round(pk, 1), lf_before_db=round(lfb, 2), lf_after_db=round(lfa, 2),
                   lf_reduction_db=round(lfb - lfa, 2), above300_change_db=round(hfa - hfb, 3),
                   broadband_reduction_db=round(level_db(y[ia:ib]) - level_db(z[ia:ib]), 2),
                   overlaps_protected=sum(1 for wa, wb in protected if wa < B and wb > A))
        patches.append(Patch(s0, z[s0:s1].copy(), rep))
        y = z
    return y, patches


# ---------------------------------------------------------------- array-level convenience
def repair_array(x, sr, events=(), protected=(), thumps=False, p: Params = Params()):
    """Repair a whole in-memory signal. events: [(t, e)] nominal spans (s). Returns (y, reports)."""
    y = np.asarray(x, dtype=np.float64).copy()
    reports, ev_spans = [], []
    for ev in sorted(events):
        others = [o for o in events if o != ev]
        for pt in repair_event(y, sr, 0.0, ev, protected, others, p):
            reports.append(dict(kind="event", **pt.report))
            if len(pt.new):
                y[pt.start_n:pt.start_n + len(pt.new)] = pt.new
                ev_spans.append((pt.start_n / sr, (pt.start_n + len(pt.new)) / sr))
    if thumps:
        y, tp = repair_thumps(y, sr, 0.0, ev_spans, p, protected)
        reports += [dict(kind="thump", **pt.report) for pt in tp]
    return y, reports


# ---------------------------------------------------------------- file-level, bit-exact streaming
def _read(path, a, b):
    import soundfile as sf
    x, sr = sf.read(str(path), start=max(0, int(a)), stop=int(b), dtype="float64", always_2d=False)
    if x.ndim > 1:
        raise ValueError("mono input only")
    return x, sr


def plan_file(src, events=(), protected=(), thump_ranges=(), p: Params = Params(), log=print):
    """Compute patches for a (possibly very long) mono file without loading it whole.
    events: [(t, e)] nominal spans; protected: [(t, e)]; thump_ranges: [(a, b)] where to look for thumps.
    Returns list of Patch with absolute start_n."""
    import soundfile as sf
    info = sf.info(str(src)); sr, total = info.samplerate, info.frames
    events = sorted(events)
    # one repair per gap: events sharing the same protected-gap are handled together
    groups = []
    for ev in events:
        g = gap_around(protected, *ev)
        if groups and groups[-1][0] == g:
            groups[-1][1].append(ev)
        else:
            groups.append((g, [ev]))
    patches = []
    for g, evs in groups:
        span = (min(v[0] for v in evs), max(v[1] for v in evs))
        c0 = max(0.0, span[0] - p.context); c1 = min(total / sr, span[1] + p.context)
        x, _ = _read(src, c0 * sr, c1 * sr)
        n0 = int(c0 * sr)
        t0 = n0 / sr
        others = [o for o in events if o not in evs]
        for pt in repair_event(x, sr, t0, span, protected, others, p):
            pt.report.update(kind="event", n_events=len(evs), members=[list(v) for v in evs])
            if len(pt.new):
                pt.start_n += n0
            patches.append(pt)
            log(f"event {span[0]:.2f}: {pt.report.get('method', pt.report.get('skipped'))} "
                f"red={pt.report.get('reduction_db')}")
    ev_p = [q for q in patches if len(q.new)]
    pieces = []                       # <=60 s analysis pieces with 2 s padding (bounded memory)
    for a, b in merge_intervals(thump_ranges):
        a, b = max(0.0, a), min(total / sr, b)
        k = a
        while k < b:
            pieces.append((k, min(b, k + 60.0)))
            k += 60.0
    for core_a, core_b in pieces:
        a, b = max(0.0, core_a - 2.0), min(total / sr, core_b + 2.0)
        n0 = int(a * sr)
        x, _ = _read(src, n0, int(b * sr))
        last = [q for q in patches if q.report.get("kind") == "thump"]
        for q in last:                # earlier pieces' thumps already applied
            o0, o1 = q.start_n - n0, q.start_n - n0 + len(q.new)
            if o1 > 0 and o0 < len(x):
                x[max(0, o0):min(len(x), o1)] = q.new[max(0, -o0):len(q.new) - max(0, o1 - len(x))]
        for q in ev_p:   # analyse on event-repaired audio
            o0, o1 = q.start_n - n0, q.start_n - n0 + len(q.new)
            if o1 > 0 and o0 < len(x):
                x[max(0, o0):min(len(x), o1)] = q.new[max(0, -o0):len(q.new) - max(0, o1 - len(x))]
        avoid = [(q.start_n / sr, (q.start_n + len(q.new)) / sr) for q in ev_p]
        avoid += [(q.start_n / sr, (q.start_n + len(q.new)) / sr) for q in last]
        _, tp = repair_thumps(x, sr, n0 / sr, avoid, p, protected)
        for q in tp:
            q.start_n += n0; q.report["kind"] = "thump"
            if core_a <= q.start_n / sr < core_b:
                patches.append(q)
    log(f"{len(ev_p)} event patches, {sum(1 for q in patches if q.report.get('kind') == 'thump')} thumps")
    return patches


def write_patched(src, dst, patches, block=1 << 21):
    """Stream src -> dst (same format, int PCM), replacing only the patched samples. Bit-exact elsewhere."""
    import soundfile as sf
    live = sorted([q for q in patches if len(q.new)], key=lambda q: q.start_n)
    for q0, q1 in zip(live, live[1:]):
        if q0.start_n + len(q0.new) > q1.start_n:
            raise ValueError(f"overlapping patches at sample {q1.start_n}")
    info = sf.info(str(src))
    sub = info.subtype if info.subtype in ("PCM_16", "PCM_24", "PCM_32") else "PCM_24"
    bits = {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32}[sub]
    tmp = f"{dst}.tmp.{os.getpid()}"
    with sf.SoundFile(str(src)) as fi, sf.SoundFile(tmp, "w", info.samplerate, info.channels, subtype=sub,
                                                   format=info.format) as fo:
        pos, qi = 0, 0
        while True:
            d = fi.read(block, dtype="int32", always_2d=False)
            n = len(d)
            if not n:
                break
            while qi < len(live) and live[qi].start_n < pos + n:
                q = live[qi]
                scale = 2 ** (bits - 1)
                v = np.clip(np.round(q.new * scale), -scale, scale - 1).astype(np.int64) << (32 - bits)
                o0 = q.start_n - pos; o1 = o0 + len(v)
                d[max(0, o0):min(n, o1)] = v[max(0, -o0):len(v) - max(0, o1 - n)].astype(np.int32)
                if o1 > n:   # continues into next block
                    live[qi] = Patch(pos + n, q.new[n - o0:], q.report)
                    break
                qi += 1
            fo.write(d)
            pos += n
    os.replace(tmp, dst)


def repair_file(src, dst, events=(), protected=(), thump_ranges=(), p: Params = Params(), log=print):
    """Plan + write. Returns the list of region reports."""
    patches = plan_file(src, events, protected, thump_ranges, p, log)
    write_patched(src, dst, patches)
    return [q.report for q in patches]


# ---------------------------------------------------------------- command line
def _intervals(path):
    """[(a, b)] from JSON: [[a, b], ...] or [{"t"/"start": a, "e"/"end": b}, ...] (or {"words": [...]})."""
    import json
    if not path:
        return []
    data = json.loads(open(path).read())
    if isinstance(data, dict):
        data = data.get("words", data.get("events", data.get("intervals", [])))
    out = []
    for v in data:
        if isinstance(v, (list, tuple)):
            out.append((float(v[0]), float(v[1])))
        else:
            out.append((float(v.get("t", v.get("start"))), float(v.get("e", v.get("end")))))
    return out


def main(argv=None) -> int:
    import argparse
    import json
    import soundfile as sf
    ap = argparse.ArgumentParser(prog="opendubstage specrepair",
                                 description="RX-style spectral repair: room-tone Replace for events in pauses, "
                                             "spectral Attenuate for events over speech, LF thump removal. Mono "
                                             "integer-PCM files; bit-exact outside the reported regions.")
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--events", help="JSON list of [start, end] event spans (s) to repair")
    ap.add_argument("--protected", help="JSON list of [start, end] spans to keep (e.g. word timings)")
    ap.add_argument("--thumps", nargs="?", const="all",
                    help="look for LF thumps: 'all' (default when given) or a JSON list of [start, end] ranges")
    ap.add_argument("--report", help="write the per-region reports as JSON")
    a = ap.parse_args(argv)
    if a.thumps == "all":
        thumps = [(0.0, sf.info(a.src).duration)]
    else:
        thumps = _intervals(a.thumps)
    reps = repair_file(a.src, a.dst, _intervals(a.events), _intervals(a.protected), thumps)
    if a.report:
        with open(a.report, "w") as fh:
            json.dump(reps, fh, indent=1, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
