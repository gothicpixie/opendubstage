"""Public-radio mix QC: the checks a broadcast mixer runs on a finished spoken-word mix, made machine-checkable.

Inputs
------
* a mix file (WAV/FLAC/AIFF, any sample rate, mono or stereo), read in blocks (programme length is not limited by RAM);
* optionally a cue sheet (dict, or a JSON path) describing what is placed where. Without one, only the programme-wide
  checks run (loudness, LRA, true peak, dead air);
* optionally stems (full-length files aligned with the mix) instead of, or as well as, per-cue renders.

Cue-sheet format (all times in seconds on the programme timeline)::

    {"duration": 3600.0,                          # optional; defaults to the mix length
     "cues": [{"id": "n1",                        # optional label, echoed in problems
               "role": "dialogue",                # dialogue | music | ambience | effects
               "class": "narration",              # dialogue voice class (e.g. narration, host, actuality)
               "start": 12.0, "end": 40.5,        # placement of the cue (including any fade tails)
               "fade_in": 0.1, "fade_out": 0.3,   # edge fades in seconds; null/absent = unknown
               "file": "cue.wav",                 # optional: the rendered cue, sample 0 at `start`
               "file_offset": 0.0,                # optional: seconds into `file` that sit at `start`
               "gain_env": "duck",                # optional: name of an envelope applied to this cue in the mix
               "speech_active": [[12.4, 18.0]],   # optional: speech intervals (dialogue); derived from `file` if absent
               "natural_end": false}],            # optional: the material ends by itself (no fade needed)
     "envelopes": {"duck": {"rate": 100, "values": [...] | "path": "duck.npy", "db": false}},  # linear gain per frame
     "stems": {"dialogue": "dx.wav", "background": "bg.wav"},   # optional, aligned with the mix
     "segments": [{"start": 0, "end": 600, "type": "narrated"}],  # optional; default: dialogue cue spans per class
     "scenes": [0.0, 600.0]}                       # optional scene/chapter boundaries (level jumps there are allowed)

Checks (thresholds and their sources are on `Thresholds`)
-----------------------------------------------------------
1. ``lra``        loudness range (EBU Tech 3342) for the programme and per segment type
2. ``jump``       loudness steps across dialogue edits (voice-to-voice), scene changes exempt
3. ``masking``    music+ambience vs dialogue energy in the 1-4 kHz presence band while speech is active
4. ``click``      sample-difference discontinuities at cue edges
5. ``hard_stop``  non-dialogue cues ending on a fade shorter than the minimum (programme end exempt)
6. ``dead_air``   level below the floor for longer than allowed (programme head and tail exempt)
7. ``tonal``      presence-band balance per dialogue voice class vs the mean of the classes
8. ``loudness``   integrated loudness (ITU-R BS.1770-4 gating) and true peak (4x oversampling), as a cross-check

API: ``qc(mix, cues=None, stems=None, thresholds=None) -> (metrics, problems)``; each problem is a string
``"[check] H:MM:SS message"``; ``metrics["events"]`` holds the same findings as structured records.
CLI: ``opendubstage qc MIX [--cues SHEET.json] [--stem dialogue=PATH --stem background=PATH] [--thresholds T.json]
[--json OUT.json] [--md OUT.md]``; exits 1 when a blocking check fails.
Requires numpy, scipy, soundfile.
"""
from __future__ import annotations

import argparse, dataclasses, json, math, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt, sosfilt_zi

FR = 100          # analysis frames per second for peaks / speech activity (10 ms)
HOP = 10          # loudness hop in FR frames (100 ms): BS.1770 gating blocks of 400 ms with 75 % overlap


@dataclasses.dataclass
class Thresholds:
    """Every threshold, with the reason for its default.

    Sources (abbreviated in the field comments):
      [R128]  EBU R 128 (2020), Loudness normalisation and permitted maximum level of audio signals.
      [3341]  EBU Tech 3341, Loudness metering: 'EBU mode' (momentary 400 ms, short-term 3 s).
      [3342]  EBU Tech 3342, Loudness Range: short-term values, -70 LUFS absolute and -20 LU relative gates,
              LRA = 95th - 10th percentile.
      [3343]  EBU Tech 3343, Guidelines for production of programmes in accordance with R 128.
      [1770]  ITU-R BS.1770-4, loudness (K-weighting, gating) and true peak (>= 4x oversampling).
      [A85]   ATSC A/85 (2013), dialogue-anchored loudness, -24 LKFS +/- 2 dB.
      [NFLX]  Netflix Sound Mix Specifications & Best Practices (dialogue-gated -27 LKFS +/- 2; true peak -2 dBTP;
              LRA guidance 4-18 LU for near-field stereo).
      [TD1004] AES TD1004.1.15-10, Recommendation for loudness of audio streaming and network file playback
              (-16..-20 LUFS, true peak <= -1 dBTP; lossy codecs overshoot, so leave headroom).
      [TORC]  Torcoli, Freke-Morin, Paulus, Simon, Shirley, 'Preferred levels for background ducking to produce
              esthetically pleasing audio for TV with clear speech', AES 146th Convention (2019): listeners
              chose background ducking around 10 LU below speech, hearing-impaired listeners more.
      [SHIR]  Shirley & Kendrick (BBC R&D / Salford) and Ofcom/BBC 'clear speech' work: background music and
              effects are the main cause of intelligibility complaints; speech-band (1-4 kHz) masking matters most.
      [JND]   Loudness just-noticeable difference is ~1 LU; broadcast dialogue tolerance is +/- 2 LU [A85][NFLX],
              so a step of more than twice the tolerance between adjacent voices is audible as a level jump.
      [PRSS]  Public Radio Satellite System loudness spec: -24 LUFS +/- 2 LU, -2 dBTP (Current, 2014; Radio World,
              'PRSS tackles audio program consistency').
      [AUPH]  Auphonic, 'Dynamic range processing' (2019): talk programmes ~5 LU LRA, features/drama 5-25 LU.
      [BYERS] NPR Training audio mixing checklist (R. Byers, 2018): "every voice roughly the same volume"; "check
              all your edits to ensure they are natural and not distracting"; balance ambience and music.
      [PRAC]  Engineering practice, not a published standard; tune per programme.
    """
    # 1 LRA ------------------------------------------------------------------------------------------------------
    lra_programme: tuple = (4.0, 18.0)   # [NFLX] near-field range; [3343] LRA is a production descriptor
    lra_segment_max: dict = dataclasses.field(default_factory=lambda: {"*": 15.0})
    #   per segment type ('*' = default); dialogue-only LRA <= 7 LU [NFLX], talk ~5 LU [AUPH]
    # 2 jumps ----------------------------------------------------------------------------------------------------
    jump_lu: float = 5.0                 # [JND][A85][BYERS] > 2x the +/-2 LU dialogue tolerance
    jump_window_s: float = 0.4           # momentary window [3341]; the step is measured within this of the edit
    jump_speech_s: float = 1.5           # speech level each side = loudest momentary within this span [PRAC]
    jump_max_gap_s: float = 1.5          # voice-to-voice edits only: dialogue gap at the edit <= this [PRAC]
    jump_scene_s: float = 2.0            # edits within this of a scene boundary are exempt
    jump_floor_lufs: float = -45.0       # both sides quieter than this: nothing to compare
    # 3 masking --------------------------------------------------------------------------------------------------
    mask_band: tuple = (1000.0, 4000.0)  # [SHIR] presence band carrying consonant cues
    mask_ratio_db: float = -10.0         # [TORC] background >= ~10 LU under speech; here in-band energy ratio.
    #   Stricter: -14 dB in 1.2-4.8 kHz (10 LU broadband + 4 LU in the masking band)
    mask_min_s: float = 3.0              # sustained stretches only
    mask_smooth_s: float = 1.0           # median smoothing of the per-100 ms ratio over speech frames
    # 4 clicks ---------------------------------------------------------------------------------------------------
    click_ratio: float = 8.0             # max |x[n]-x[n-1]| within +/-5 ms vs RMS of the difference around it [PRAC]
    click_win_ms: float = 5.0
    click_ref_ms: float = 50.0
    click_min_dbfs: float = -60.0        # ignore steps below this (inaudible)
    # 5 hard stops -----------------------------------------------------------------------------------------------
    hard_stop_fade_s: float = 1.5        # [PRAC][BYERS] music/ambience tails shorter than this read as cut-offs
    hard_stop_end_s: float = 0.5         # cues ending within this of the programme end are exempt
    hard_stop_tail_db: float = -30.0     # rendered cue's last 100 ms within this of its median level = still sounding
    # 6 dead air -------------------------------------------------------------------------------------------------
    dead_air_dbfs: float = -60.0         # sample peak floor (same as ffmpeg silencedetect noise=-60dB)
    dead_air_s: float = 3.0              # [PRAC] silence detectors trip at a few seconds; strict: -50 dBFS / 2 s
    dead_air_edge_s: float = 5.0         # head/tail exempt
    # 7 tonal balance --------------------------------------------------------------------------------------------
    tonal_band: tuple = (1000.0, 4000.0)
    tonal_dev_db: float = 4.0            # [PRAC] a 4 dB presence difference is clearly audible as a voice change
    # 8 loudness -------------------------------------------------------------------------------------------------
    lufs_range: tuple | None = (-26.0, -22.0)   # [PRSS][A85] -24 +/- 2; set None to skip
    true_peak_max: float = -2.0                 # [PRSS][NFLX]; R128/TD1004 allow -1 but lossy codecs overshoot
    expected_lufs: float | None = None          # cross-check against another meter's reading
    expected_tol: float = 0.5
    # which checks fail the run (the rest are advisory) --------------------------------------------------------
    blocking: tuple = ("loudness", "dead_air", "click", "masking")

    @classmethod
    def from_dict(cls, d):
        t = cls()
        for k, v in (d or {}).items():
            if not hasattr(t, k): raise KeyError(f"unknown threshold {k!r}")
            setattr(t, k, tuple(v) if isinstance(getattr(t, k), tuple) and v is not None else v)
        return t


def hms(t):
    t = max(0.0, float(t)); return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}"


# --- DSP helpers -------------------------------------------------------------------------------------------------
def kweight_sos(sr):
    """BS.1770 K-weighting (pre-filter shelf + RLB high-pass) for any sample rate (libebur128 formulation)."""
    f0, G, Q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    K = math.tan(math.pi * f0 / sr); Vh = 10 ** (G / 20); Vb = Vh ** 0.4996667741545416
    a0 = 1 + K / Q + K * K
    s1 = [(Vh + Vb * K / Q + K * K) / a0, 2 * (K * K - Vh) / a0, (Vh - Vb * K / Q + K * K) / a0,
          1.0, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0]
    f0, Q = 38.13547087602444, 0.5003270373238773
    K = math.tan(math.pi * f0 / sr); a0 = 1 + K / Q + K * K
    s2 = [1.0, -2.0, 1.0, 1.0, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0]
    return np.array([s1, s2])


def frame_mean(x, n):
    """Mean of consecutive length-n frames along axis 0 (drops the remainder)."""
    m = len(x) // n
    return x[:m * n].reshape(m, n, *x.shape[1:]).mean(1)


def lufs_of(p):
    return -0.691 + 10 * np.log10(np.maximum(p, 1e-20))


def gated_integrated(pm):
    """BS.1770 integrated loudness from 400 ms block powers (75 % overlap)."""
    pm = pm[lufs_of(pm) > -70]
    if not len(pm): return -70.0
    rel = lufs_of(pm.mean()) - 10
    pm = pm[lufs_of(pm) > rel]
    return float(lufs_of(pm.mean())) if len(pm) else -70.0


def lra_of(st):
    """EBU Tech 3342 loudness range from short-term loudness values (LUFS)."""
    st = st[st > -70]
    if len(st) < 10: return None
    rel = lufs_of(np.mean(10 ** ((st + 0.691) / 10))) - 20
    st = st[st > rel]
    if len(st) < 10: return None
    return float(np.percentile(st, 95) - np.percentile(st, 10))


def activity(mono, sr, rel_db=-20.0):
    """Speech-active 10 ms frames of a cue: RMS within rel_db of its loud parts (90th percentile)."""
    hop = sr // FR; n = len(mono) // hop
    if n == 0: return np.zeros(0, bool)
    db = 10 * np.log10((mono[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)
    return db > (np.percentile(db, 90) + rel_db)


def runs(mask):
    """[(i0, i1)] index runs where mask is true."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]]).astype(np.int8)
    d = np.diff(m); return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


# --- cue sheet ----------------------------------------------------------------------------------------------------
def load_sheet(cues, stems=None):
    if cues is None: sheet = {"cues": []}
    elif isinstance(cues, (str, Path)):
        p = Path(cues); sheet = json.loads(p.read_text())
        base = p.parent
        for c in sheet.get("cues", []):
            if c.get("file") and not Path(c["file"]).is_absolute(): c["file"] = str(base / c["file"])
        for e in sheet.get("envelopes", {}).values():
            if e.get("path") and not Path(e["path"]).is_absolute(): e["path"] = str(base / e["path"])
    else: sheet = dict(cues)
    sheet.setdefault("cues", [])
    st = dict(sheet.get("stems") or {}); st.update(stems or {})
    for role, path in st.items():          # a stem is a cue spanning the programme, sample 0 at t = 0
        sheet["cues"].append({"id": f"stem:{role}", "role": "dialogue" if role.startswith("dialogue") else "music",
                              "class": role.split(":", 1)[1] if ":" in role else "dialogue",
                              "start": 0.0, "end": sf.info(str(path)).duration, "file": str(path), "_stem": True})
    envs = {}
    for k, e in (sheet.get("envelopes") or {}).items():
        v = np.asarray(e["values"] if "values" in e else np.load(e["path"]), np.float64)
        envs[k] = (float(e.get("rate", FR)), 10 ** (v / 20) if e.get("db") else v)
    sheet["_env"] = envs
    return sheet


# --- the pass over the mix ----------------------------------------------------------------------------------------
def analyse(mix, sheet, th, block_s=60.0, log=None):
    info = sf.info(str(mix)); sr, nch, n = info.samplerate, info.channels, info.frames
    h10 = sr // FR; B = int(block_s) * FR * h10
    kw = kweight_sos(sr)
    band = butter(4, list(th.mask_band), btype="bandpass", fs=sr, output="sos")
    tband = butter(4, list(th.tonal_band), btype="bandpass", fs=sr, output="sos")
    zk = np.repeat(sosfilt_zi(kw)[:, :, None], nch, axis=2) * 0.0

    cues = sheet["cues"]; envs = sheet["_env"]
    placed = []                                 # (stem key, SoundFile, c0 sample on programme, file offset, frames, env)
    for c in cues:
        if not c.get("file"): continue
        f = sf.SoundFile(str(c["file"]))
        if f.samplerate != sr: raise ValueError(f"{c['file']}: {f.samplerate} Hz, mix is {sr} Hz")
        key = f"dlg:{c.get('class', 'dialogue')}" if c.get("role") == "dialogue" else "bg"
        off = int(round(c.get("file_offset", 0.0) * sr))
        placed.append((key, f, int(round(c["start"] * sr)), off, f.frames - off, envs.get(c.get("gain_env"))))
    keys = sorted({p[0] for p in placed})
    zb = {k: sosfilt_zi(band) * 0.0 for k in keys}; zt = {k: sosfilt_zi(tband) * 0.0 for k in keys}

    P10, PK, stemP, stemT, stemF = [], [], {k: [] for k in keys}, {k: [] for k in keys}, {k: [] for k in keys}
    tp = 0.0; t0 = time.time()
    with sf.SoundFile(str(mix)) as fm:
        for b0 in range(0, n, B):
            x = fm.read(min(B, n - b0), dtype="float64", always_2d=True)
            if not len(x): break
            k, zk = sosfilt(kw, x, axis=0, zi=zk)
            P10.append(frame_mean((k ** 2).sum(1), h10))
            m = len(x) // h10
            PK.append(np.abs(x[:m * h10]).reshape(m, h10, nch).max((1, 2)))
            pk = float(np.abs(x).max()) if len(x) else 0.0
            if pk > 10 ** ((th.true_peak_max - 4.0) / 20):
                tp = max(tp, float(np.abs(resample_poly(x, 4, 1, axis=0)).max()))
            tp = max(tp, pk)
            if keys:
                b1 = b0 + len(x); acc = {kk: np.zeros(len(x)) for kk in keys}
                for key, f, c0, off, frames, env in placed:
                    c1 = c0 + frames
                    if c1 <= b0 or c0 >= b1: continue
                    o0, o1 = max(b0, c0), min(b1, c1)
                    f.seek(off + o0 - c0); y = f.read(o1 - o0, dtype="float64", always_2d=True).mean(1)
                    if env is not None:
                        rate, v = env; t = np.arange(o0, o0 + len(y)) / sr * rate
                        y = y * np.interp(t, np.arange(len(v)), v)
                    acc[key][o0 - b0:o0 - b0 + len(y)] += y
                for kk in keys:
                    yb, zb[kk] = sosfilt(band, acc[kk], zi=zb[kk]); stemP[kk].append(frame_mean(yb ** 2, h10))
                    yt, zt[kk] = sosfilt(tband, acc[kk], zi=zt[kk]); stemT[kk].append(frame_mean(yt ** 2, h10))
                    stemF[kk].append(frame_mean(acc[kk] ** 2, h10))
            if (b0 // B) % 20 == 19 and log: log(f"  qc: {(b0 + len(x)) / sr / 60:.0f} min analysed ({time.time() - t0:.0f} s)")
    for p in placed: p[1].close()
    cat = lambda L: np.concatenate(L) if L else np.zeros(0)
    return dict(sr=sr, nch=nch, n=n, dur=n / sr, p10=cat(P10), pk=cat(PK), tp=tp,
                band={k: cat(v) for k, v in stemP.items()}, tband={k: cat(v) for k, v in stemT.items()},
                full={k: cat(v) for k, v in stemF.items()})


def speech_mask(sheet, nf, sr_unused=None):
    """10 ms speech-active frames per dialogue class (from speech_active, else the cue file, else the cue span)."""
    out = {}
    for c in sheet["cues"]:
        if c.get("role") != "dialogue": continue
        m = out.setdefault(c.get("class", "dialogue"), np.zeros(nf, bool))
        iv = c.get("speech_active")
        if iv is None and c.get("file"):
            y, sr = sf.read(str(c["file"]), dtype="float32", always_2d=True)
            off = int(round(c.get("file_offset", 0.0) * sr)); a = activity(y[off:].mean(1), sr)
            i0 = int(round(c["start"] * FR)); seg = a[:max(0, min(len(a), nf - i0))]
            m[i0:i0 + len(seg)] |= seg; continue
        for a, b in (iv if iv is not None else [(c["start"], c["end"])]):
            m[max(0, int(a * FR)):max(0, min(nf, int(math.ceil(b * FR))))] = True
    return out


# --- the checks ---------------------------------------------------------------------------------------------------
def qc(mix, cues=None, stems=None, thresholds=None, log=lambda m: print(m, flush=True), block_s=60.0):
    th = thresholds if isinstance(thresholds, Thresholds) else Thresholds.from_dict(thresholds)
    sheet = load_sheet(cues, stems)
    T0 = time.time(); A = analyse(mix, sheet, th, block_s, log)
    dur = float(sheet.get("duration") or A["dur"])
    ev = []
    def add(check, t, msg, **kw): ev.append({"check": check, "t": float(t), "msg": msg, **kw})

    p10 = A["p10"]; nh = len(p10) // HOP
    p100 = p10[:nh * HOP].reshape(nh, HOP).mean(1)                       # 100 ms powers
    cs = np.concatenate([[0], np.cumsum(p100)])
    pm = (cs[4:] - cs[:-4]) / 4; M = lufs_of(pm)                         # momentary, window ends at (i+4)*0.1 s
    ps = (cs[30:] - cs[:-30]) / 30; S = lufs_of(ps)                      # short-term, window ends at (i+30)*0.1 s
    S_mid = (np.arange(len(S)) + 15) * 0.1

    # 8 loudness / true peak
    I = gated_integrated(pm); TP = 20 * math.log10(max(A["tp"], 1e-12))
    met = {"duration_s": A["dur"], "integrated_lufs": round(I, 2), "true_peak_dbtp": round(TP, 2)}
    if th.lufs_range and not th.lufs_range[0] <= I <= th.lufs_range[1]:
        add("loudness", 0, f"integrated {I:.1f} LUFS outside {th.lufs_range[0]}..{th.lufs_range[1]}")
    if TP > th.true_peak_max: add("loudness", 0, f"true peak {TP:.2f} dBTP > {th.true_peak_max}")
    if th.expected_lufs is not None and abs(I - th.expected_lufs) > th.expected_tol:
        add("loudness", 0, f"integrated {I:.2f} LUFS disagrees with the reference reading {th.expected_lufs}")

    # 1 LRA
    lra = lra_of(S); met["lra_programme"] = None if lra is None else round(lra, 2)
    if lra is not None and not th.lra_programme[0] <= lra <= th.lra_programme[1]:
        add("lra", 0, f"programme LRA {lra:.1f} LU outside {th.lra_programme[0]}..{th.lra_programme[1]}")
    nf = len(A["pk"]); sp = speech_mask(sheet, nf)
    segs = sheet.get("segments")
    if segs is None:
        segs = [{"start": c["start"], "end": c["end"], "type": c.get("class", "dialogue")}
                for c in sheet["cues"] if c.get("role") == "dialogue" and not c.get("_stem")]
    met["lra_segments"] = {}
    for typ in sorted({s["type"] for s in segs}):
        inside = np.zeros(len(S), bool)
        for s in segs:
            if s["type"] == typ: inside |= (S_mid >= s["start"]) & (S_mid < s["end"])
        v = lra_of(S[inside]); met["lra_segments"][typ] = {"lra": None if v is None else round(v, 2),
                                                          "minutes": round(inside.sum() / 600, 1)}
        lim = th.lra_segment_max.get(typ, th.lra_segment_max.get("*"))
        if v is not None and lim is not None and v > lim:
            add("lra", 0, f"segment type {typ!r} LRA {v:.1f} LU > {lim}", type=typ)

    # 6 dead air
    floor = 10 ** (th.dead_air_dbfs / 20); quiet = A["pk"] < floor; nd = 0
    for i0, i1 in runs(quiet):
        a, b = i0 / FR, i1 / FR
        a2, b2 = max(a, th.dead_air_edge_s), min(b, A["dur"] - th.dead_air_edge_s)
        if b2 - a2 > th.dead_air_s:
            add("dead_air", a2, f"{b2 - a2:.1f} s below {th.dead_air_dbfs:.0f} dBFS", dur=b2 - a2); nd += 1

    dl = [c for c in sheet["cues"] if not c.get("_stem")]
    # 2 jumps across voice-to-voice edits
    scenes = sorted(sheet.get("scenes") or [])
    dcues = sorted((c for c in dl if c.get("role") == "dialogue"), key=lambda c: c["start"])
    mt = lambda t: (t - 0.4) / 0.1                                          # momentary index whose window ends at t
    def loud(a, b):
        i0, i1 = int(math.ceil(mt(a + 0.4))), int(math.floor(mt(b)))
        i0, i1 = max(0, i0), min(len(M) - 1, i1)
        return float(M[i0:i1 + 1].max()) if i1 >= i0 else -70.0
    njump = 0; jumps = []
    for prev, nxt in zip(dcues, dcues[1:]):
        gap = nxt["start"] - prev["end"]
        if gap > th.jump_max_gap_s: continue
        t = nxt["start"] if gap >= 0 else (prev["end"] + nxt["start"]) / 2
        if any(abs(t - s) <= th.jump_scene_s for s in scenes): continue
        Lb = loud(prev["end"] - th.jump_speech_s, min(prev["end"], t))
        La = loud(max(nxt["start"], t), nxt["start"] + th.jump_speech_s)
        if max(Lb, La) < th.jump_floor_lufs: continue
        w = th.jump_window_s
        jumps.append(La - Lb)
        if abs(La - Lb) > th.jump_lu:
            add("jump", t, f"{La - Lb:+.1f} LU across the edit {prev.get('class', '')}→{nxt.get('class', '')} "
                           f"({prev.get('id', '?')}→{nxt.get('id', '?')}; speech {Lb:.1f} → {La:.1f} LUFS momentary max)",
                value=La - Lb); njump += 1
    met["jump_edits_checked"] = len(jumps)

    # 3 masking (needs stems / cue renders); 100 ms frames where speech is active
    dkeys = [k for k in A["band"] if k.startswith("dlg:")]
    met["masking_mode"] = "stems" if dkeys and "bg" in A["band"] else "unavailable (no dialogue/background audio)"
    if dkeys and "bg" in A["band"]:
        anysp = np.zeros(nf, bool)
        for v in sp.values(): anysp |= v
        n1 = min(len(A["band"]["bg"]), len(anysp) // HOP * HOP) // HOP * HOP
        act = anysp[:n1].reshape(-1, HOP).mean(1) >= 0.5
        d = sum(frame_mean(A["band"][k][:n1], HOP) for k in dkeys); g = frame_mean(A["band"]["bg"][:n1], HOP)
        act &= d > 1e-10                       # speech per the sheet but no dialogue audio (render missing): skip
        ratio = 10 * np.log10((g + 1e-20) / (d + 1e-20))
        idx = np.flatnonzero(act); r = ratio[idx]; w = max(1, int(th.mask_smooth_s * 10))
        if len(r):
            from scipy.ndimage import median_filter
            rs = median_filter(r, size=w, mode="nearest")
            sm = np.full(len(ratio), np.nan); sm[idx] = rs
            bad = np.zeros(len(ratio), bool); bad[idx] = rs > th.mask_ratio_db
            # bridge speech pauses of up to 1 s inside a bad stretch
            gaps = runs(~bad)
            for a0, a1 in gaps:
                if 0 < a0 and a1 < len(bad) and a1 - a0 <= 10: bad[a0:a1] = True
            rb = r[r > -100]                    # frames with some background at all
            met["masking_frames_with_background"] = round(len(rb) / len(r), 3)
            met["masking_median_db"] = round(float(np.median(rb)), 2) if len(rb) else None
            met["masking_p90_db"] = round(float(np.percentile(rb, 90)), 2) if len(rb) else None
            for i0, i1 in runs(bad):
                if (i1 - i0) / 10 > th.mask_min_s:
                    worst = float(np.nanmax(sm[i0:i1]))
                    add("masking", i0 / 10, f"background {worst:+.1f} dB vs dialogue in "
                        f"{th.mask_band[0]/1000:g}-{th.mask_band[1]/1000:g} kHz for {(i1 - i0) / 10:.1f} s "
                        f"(limit {th.mask_ratio_db:+.0f} dB)", dur=(i1 - i0) / 10, value=worst)

    # 7 tonal balance per voice class (from that class's own audio)
    met["tonal_db"] = {}
    for k in dkeys:
        cls = k[4:]; m = sp.get(cls)
        if m is None: continue
        n1 = min(len(A["tband"][k]), len(m)); sel = m[:n1]
        if sel.sum() < FR * 5: continue
        met["tonal_db"][cls] = round(float(10 * np.log10(A["tband"][k][:n1][sel].sum() / (A["full"][k][:n1][sel].sum() + 1e-20))), 2)
    if len(met["tonal_db"]) >= 2:
        mean = float(np.mean(list(met["tonal_db"].values())))
        for cls, v in met["tonal_db"].items():
            if abs(v - mean) > th.tonal_dev_db:
                add("tonal", 0, f"voice class {cls!r} presence {v - mean:+.1f} dB vs the class mean "
                                f"({th.tonal_band[0]/1000:g}-{th.tonal_band[1]/1000:g} kHz share {v:.1f} dB)", value=v - mean)

    # 5 hard stops
    nhs = 0
    for c in dl:
        if c.get("role") == "dialogue" or c.get("natural_end"): continue
        if c["end"] >= dur - th.hard_stop_end_s: continue
        fo = c.get("fade_out")
        if fo is None or fo >= th.hard_stop_fade_s: continue
        if c.get("file"):              # had the material already ended before its fade began?
            with sf.SoundFile(str(c["file"])) as f:
                fsr = f.samplerate; off = int(round(c.get("file_offset", 0.0) * fsr))
                f.seek(off); y = f.read(int(round((c["end"] - c["start"]) * fsr)), dtype="float32", always_2d=True).mean(1)
            e = len(y) - int(fo * fsr); pre = y[max(0, e - fsr // 10):e]
            if len(y) > fsr and len(pre):
                body = math.sqrt(float(np.median(frame_mean(y.astype(np.float64) ** 2, fsr // 10))) + 1e-20)
                if 20 * math.log10(math.sqrt(float(np.mean(pre.astype(np.float64) ** 2)) + 1e-20) / body) < th.hard_stop_tail_db:
                    continue
        add("hard_stop", c["end"], f"{c.get('role')} cue {c.get('id', '?')} ends with a {fo:.2f} s fade "
                                   f"(< {th.hard_stop_fade_s} s)", value=fo); nhs += 1

    # 4 clicks at cue edges, measured in the mix
    # only HARD edges can click: an edge with a fade of 50 ms or more has no energy at its boundary (it was flagging
    # whatever else began at that instant, e.g. a word onset under a faded-out song)
    edges = sorted({round(t, 3) for c in dl for t, fd in ((c["start"], c.get("fade_in")), (c["end"], c.get("fade_out")))
                    if 0.06 < t < A["dur"] - 0.06 and (fd is None or fd < 0.05)})
    ncl = 0; sr = A["sr"]; hw = int(th.click_win_ms / 1000 * sr); rw = int(th.click_ref_ms / 1000 * sr)
    with sf.SoundFile(str(mix)) as fm:
        last = -1.0
        for t in edges:
            if t - last < 0.01: continue
            last = t; c0 = int(round(t * sr)) - rw - hw - 1; fm.seek(max(0, c0))
            x = fm.read(2 * (rw + hw) + 2, dtype="float64", always_2d=True)
            dd = np.abs(np.diff(x, axis=0)).max(1); mid = len(dd) // 2
            cen = dd[mid - hw:mid + hw + 1]; ref = np.concatenate([dd[:mid - hw], dd[mid + hw + 1:]])
            if not len(cen) or not len(ref): continue
            peak = float(cen.max()); rr = float(np.sqrt(np.mean(ref ** 2)) + 1e-9)
            if peak > 10 ** (th.click_min_dbfs / 20) and peak / rr > th.click_ratio:
                add("click", t, f"step {20 * math.log10(peak):.0f} dBFS = {peak / rr:.0f}x the local difference RMS",
                    value=peak / rr); ncl += 1
    met["edges_checked"] = len(edges)

    ev.sort(key=lambda e: (e["check"] != "loudness", e["t"]))
    met["counts"] = {k: sum(e["check"] == k for e in ev) for k in
                     ("loudness", "lra", "jump", "masking", "click", "hard_stop", "dead_air", "tonal")}
    met["blocking_checks"] = list(th.blocking)
    met["blocking_failures"] = sum(met["counts"][k] for k in th.blocking if k in met["counts"])
    met["events"] = ev; met["runtime_s"] = round(time.time() - T0, 1)
    probs = [f"[{e['check']}] {hms(e['t'])} {e['msg']}" for e in ev]
    return met, probs


def summary_md(met, probs, title="Mix QC"):
    L = [f"# {title}", "", f"- duration {met['duration_s'] / 60:.1f} min, integrated {met['integrated_lufs']} LUFS, "
         f"true peak {met['true_peak_dbtp']} dBTP, LRA {met['lra_programme']} LU", f"- runtime {met['runtime_s']} s",
         "", "| check | problems | blocking |", "|---|---|---|"]
    for k, v in met["counts"].items(): L.append(f"| {k} | {v} | {'yes' if k in met['blocking_checks'] else 'advisory'} |")
    L += ["", "## Segment LRA", ""] + [f"- {k}: {v['lra']} LU over {v['minutes']} min" for k, v in met["lra_segments"].items()]
    if met.get("tonal_db"): L += ["", "## Voice-class presence share (dB, 1-4 kHz of full band)", ""] + \
        [f"- {k}: {v}" for k, v in met["tonal_db"].items()]
    L += ["", f"## Problems ({len(probs)})", ""] + [f"- {p}" for p in probs]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="opendubstage qc", description="Public-radio mix QC")
    ap.add_argument("mix"); ap.add_argument("--cues"); ap.add_argument("--stem", action="append", default=[],
                                                                        help="role=path, e.g. dialogue=dx.wav")
    ap.add_argument("--thresholds"); ap.add_argument("--json"); ap.add_argument("--md")
    a = ap.parse_args(argv)
    th = Thresholds.from_dict(json.loads(Path(a.thresholds).read_text())) if a.thresholds else Thresholds()
    met, probs = qc(a.mix, a.cues, dict(s.split("=", 1) for s in a.stem), th)
    if a.json: Path(a.json).write_text(json.dumps({"metrics": met, "problems": probs}, indent=1, default=float))
    if a.md: Path(a.md).write_text(summary_md(met, probs))
    print(summary_md(met, probs) if not a.md else f"{len(probs)} problems; report {a.md}")
    return 1 if met["blocking_failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
