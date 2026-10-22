"""Synthetic-only checks for statistical room tone; no listening-quality claim."""
import json
import numpy as np
import pytest
import soundfile as sf
from scipy import signal
from opendubstage.restore.roomtone import Params, clean_intervals, generate, learn
from opendubstage.restore import specrepair
from opendubstage.cli import main

SR = 16000


def noise(seconds=12, seed=4):
    return .002 * signal.lfilter([1], [1, -.85], np.random.default_rng(seed).normal(size=int(SR * seconds)))


def test_spectrum_level_and_nonperiodicity():
    x = noise(20)
    y, rep = generate(x, SR, 30, seed=9)
    f, a = signal.welch(x, SR, nperseg=4096)
    _, b = signal.welch(y, SR, nperseg=4096)
    # Resolved bands, not unmeasurable near-DC or Nyquist slivers.
    for centre in [100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000,
                   1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300]:
        band = (f >= centre / 2 ** (1/6)) & (f < centre * 2 ** (1/6))
        assert abs(10 * np.log10(b[band].sum() / a[band].sum())) < 1.5
    assert abs(20 * np.log10(np.sqrt(np.mean(y*y)) / np.sqrt(np.mean(x*x)))) < .2
    # Long-lag autocorrelation excludes expected short-lag colour correlation.
    ac = signal.fftconvolve(y, y[::-1], mode='full')[len(y)-1:]
    assert np.max(np.abs(ac[int(.1*SR):int(5*SR)] / ac[0])) < .05
    assert len(y) == SR * 30
    assert rep['output_rms'] == pytest.approx(rep['source_rms'])
    # No seam at any analysis hop: boundary differences have normal noise scale.
    hop = rep['frame_samples'] // 4
    assert np.mean(np.diff(y)[hop-1::hop] ** 2) < 1.5 * np.mean(np.diff(y)**2)


def test_protected_and_excluded_samples_never_learned():
    x = noise(5); other = x.copy()
    other[SR:2*SR] = .8
    other[3*SR:4*SR] = -.6
    a, _ = generate(x, SR, 2, protected=[(1, 2)], excluded=[(3, 4)])
    b, rep = generate(other, SR, 2, protected=[(1, 2)], excluded=[(3, 4)])
    assert np.array_equal(a, b)
    assert all(not (start < 2.02 and end > .98) for start, end in rep['clean_intervals'])
    assert np.array_equal(x, noise(5))


def test_seed_repeatability_and_fresh_randomness():
    x = noise()
    a, _ = generate(x, SR, 2, seed=3)
    b, _ = generate(x, SR, 2, seed=3)
    c, _ = generate(x, SR, 2, seed=4)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_learns_slow_modulation():
    x = noise(20)
    t = np.arange(len(x)) / SR
    x *= 10 ** (4 * np.sin(2*np.pi*.3*t) / 20)
    model = learn(x, SR)
    assert np.std(model['modulation_db']) > 2
    y, rep = generate(x, SR, 30)
    # Learned modulation survives synthesis, rather than stationary white noise.
    block = int(.25 * SR)
    rms = np.sqrt(np.mean(y[:len(y)//block*block].reshape(-1,block)**2, axis=1))
    assert np.std(20*np.log10(rms)) > 1
    assert rep['modulation_std_db'] > 2


def test_silent_source_stays_silent():
    y, rep = generate(np.zeros(SR), SR, .2)
    assert not y.any()
    assert rep['source_rms'] == 0
    json.dumps(rep, allow_nan=False)


@pytest.mark.parametrize('duration', [0, -1, float('nan'), float('inf'), 1e-10])
def test_invalid_durations(duration):
    with pytest.raises(ValueError):
        generate(noise(1), SR, duration)


@pytest.mark.parametrize('intervals', [[(-1,1)], [(1,.5)], [(0,2)], [(0,float('nan'))]])
def test_invalid_intervals(intervals):
    with pytest.raises(ValueError):
        generate(noise(1), SR, 1, protected=intervals)


def test_insufficient_gaps_and_mono_only():
    with pytest.raises(ValueError, match='insufficient'):
        generate(noise(1), SR, 1, protected=[(.1, .9)])
    with pytest.raises(ValueError, match='mono'):
        generate(np.zeros((SR,2)), SR, 1)
    with pytest.raises(ValueError):
        generate(np.array([np.nan]*SR), SR, 1)
    with pytest.raises(ValueError):
        generate(noise(1), SR, 1, p=Params(guard_s=-1))


def test_merge_protection():
    assert clean_intervals(5, [(1,3),(2,4)], guard_s=0) == [(0.,1), (4,5)]


def test_cli_roundtrip_and_report(tmp_path):
    src, dst, report = [tmp_path / v for v in ('src.wav','dst.wav','report.json')]
    sf.write(src, noise(2), SR, subtype='FLOAT')
    assert main(['roomtone',str(src),str(dst),'--duration','3','--report',str(report)]) == 0
    y, sr = sf.read(dst)
    assert len(y) == SR * 3 and sr == SR
    assert json.loads(report.read_text())['samples'] == len(y)
    with pytest.raises(SystemExit):
        main(['roomtone',str(src),str(src),'--duration','3'])


def _fallback_fixture():
    x = noise(6)
    # Artificial short quiet islands. Tests directly pin candidate selection,
    # independent of noisy event-detector statistics.
    fdb = np.full(len(specrepair.frame_db(x, SR)), -20.)
    for start in [.2,.8,1.4,4.,4.6,5.2]:
        fdb[int(start/.01):int(start/.01)+18] = -50
    return x, fdb


def test_opt_in_fallback_and_untouched_samples():
    x, fdb = _fallback_fixture()
    args = (x, SR, 0., 2.5, 3.5, (2.6,3.4), fdb, -50., [], [(2.5,3.5)], (2.4,3.6))
    old = specrepair.fill_room_tone(*args, specrepair.Params(), {})
    assert old.report['skipped'] == 'no room tone found'
    patch = specrepair.fill_room_tone(*args, specrepair.Params(synthesize_tone=True), {})
    assert patch.report['method'] == 'room_tone_synthesis'
    y = x.copy(); y[patch.start_n:patch.start_n+len(patch.new)] = patch.new
    assert np.array_equal(y[:patch.start_n], x[:patch.start_n])
    assert np.array_equal(y[patch.start_n+len(patch.new):], x[patch.start_n+len(patch.new):])
    assert patch.report['synthesis']['clean_seconds'] >= .5


def test_fallback_respects_context_and_insufficient_evidence():
    x, fdb = _fallback_fixture()
    patch = specrepair.fill_room_tone(x, SR, 0., 2.5, 3.5, (2.6,3.4), fdb, -50., [],
          [(2.5,3.5)], (2.4,3.6), specrepair.Params(synthesize_tone=True, context=.5), {})
    assert patch.report['skipped'] == 'room-tone synthesis unavailable'
    assert not len(patch.new)


def test_synthesized_patch_pcm_bit_exact_outside(tmp_path):
    x, fdb = _fallback_fixture()
    patch = specrepair.fill_room_tone(x, SR, 0., 2.5, 3.5, (2.6,3.4), fdb, -50., [],
          [(2.5,3.5)], (2.4,3.6), specrepair.Params(synthesize_tone=True), {})
    src, dst = tmp_path/'in.wav', tmp_path/'out.wav'
    sf.write(src, x, SR, subtype='PCM_24')
    specrepair.write_patched(src,dst,[patch])
    a, _ = sf.read(src,dtype='int32'); b,_ = sf.read(dst,dtype='int32')
    assert np.array_equal(a[:patch.start_n],b[:patch.start_n])
    end=patch.start_n+len(patch.new)
    assert np.array_equal(a[end:],b[end:])


def test_cli_report_cannot_overwrite_audio(tmp_path):
    src, dst = tmp_path/'in.wav', tmp_path/'out.wav'
    sf.write(src, noise(1), SR)
    before = src.read_bytes()
    with pytest.raises(SystemExit):
        main(['roomtone',str(src),str(dst),'--duration','1','--report',str(src)])
    assert src.read_bytes() == before
    assert not dst.exists()


def test_specrepair_cli_accepts_opt_in_without_changing_unpatched_pcm(tmp_path):
    src, dst = tmp_path/'in.wav', tmp_path/'out.wav'
    sf.write(src, noise(1), SR, subtype='PCM_16')
    assert main(['specrepair',str(src),str(dst),'--synthesize-tone','--tone-seed','7']) == 0
    a,_ = sf.read(src,dtype='int32'); b,_ = sf.read(dst,dtype='int32')
    assert np.array_equal(a,b)


@pytest.mark.parametrize('data', [{"wordz": [[0,1]]}, {"words": None}, {"words": [], "events": []},
                                  "oops", [[0,1,2]], [{"start": 0}], [[False,1]], [[0,"1"]]])
def test_malformed_timing_schema_fails_closed(tmp_path, data):
    src, dst, timings = tmp_path/'in.wav', tmp_path/'out.wav', tmp_path/'words.json'
    sf.write(src, noise(1), SR)
    timings.write_text(json.dumps(data))
    with pytest.raises(SystemExit):
        main(['roomtone',str(src),str(dst),'--duration','1','--protected',str(timings)])
    assert not dst.exists()
    with pytest.raises(SystemExit):
        main(['specrepair',str(src),str(dst),'--synthesize-tone','--protected',str(timings)])
    assert not dst.exists()


@pytest.mark.parametrize('tool', ['roomtone', 'specrepair'])
@pytest.mark.parametrize('output', ['audio', 'report'])
def test_outputs_cannot_overwrite_timing_inputs(tmp_path, tool, output):
    src, dst, timing = tmp_path/'in.wav', tmp_path/'out.wav', tmp_path/'words.json'
    sf.write(src, noise(1), SR)
    timing.write_text('[]')
    args=[tool,str(src),str(timing if output=='audio' else dst),'--protected',str(timing)]
    args += ['--duration','1'] if tool=='roomtone' else ['--synthesize-tone']
    if output=='report':
        args += ['--report',str(timing)]
    with pytest.raises(SystemExit):
        main(args)
    assert timing.read_text() == '[]'
    assert not dst.exists()


def test_fallback_report_intervals_are_absolute_for_windowed_input():
    x, fdb = _fallback_fixture()
    patch = specrepair.fill_room_tone(x, SR, 10., 12.5, 13.5, (12.6,13.4), fdb, -50., [],
          [(12.5,13.5)], (12.4,13.6), specrepair.Params(synthesize_tone=True), {})
    assert patch.report['method'] == 'room_tone_synthesis'
    assert patch.report['synthesis']['source_offset_seconds'] == 10.
    assert all(10 <= a < b <= 16 for a,b in patch.report['synthesis']['clean_intervals'])
