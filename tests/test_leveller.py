"""Synthetic invariants for phrase-safe levelling, no recordings or models."""
import json

import numpy as np
import pytest
import soundfile as sf

from opendubstage.cli import main
from opendubstage.mix.leveller import level, phrase_loudness

SR = 16000


def tone(seconds=4, amplitude=.03):
    t = np.arange(int(seconds * SR)) / SR
    return amplitude * np.sin(2 * np.pi * 440 * t)


def test_converges_and_never_moves_inside_words():
    x = tone()
    x[2 * SR:3 * SR] *= 4
    timings = {'words': [{'start': .5, 'end': 1}, {'start': 1.1, 'end': 1.5},
                         {'start': 2, 'end': 3}]}
    y, rep = level(x, SR, timings, min_gain=-24, max_gain=24)
    for e in rep['events']:
        assert abs(e['after_loudness'] + 24) < 1
        a, b = e['start_sample'], e['end_sample']
        np.testing.assert_allclose(y[a:b], x[a:b] * 10 ** (e['gain_db'] / 20), rtol=1e-14)
    for a, b in rep['untouched_ranges']:
        assert np.array_equal(y[a:b], x[a:b])
    assert np.array_equal(y[:int(.3 * SR)], x[:int(.3 * SR)])


def test_noop_bit_exact_input_not_mutated():
    x = tone()
    original = x.copy()
    target = phrase_loudness(x[SR:2 * SR], SR)
    y, rep = level(x, SR, {'words': [{'start': 1, 'end': 2}]}, target=target)
    assert np.array_equal(x, original) and np.array_equal(y, original)
    assert rep['changed_ranges'] == []
    assert rep['untouched_ranges'] == [[0, len(x)]]


def test_silence_empty_timings_and_empty_audio():
    for x, words in [(np.zeros(SR), [{'start': 0, 'end': 1}]), (tone(), []), (np.zeros(0), [])]:
        y, rep = level(x, SR, {'words': words})
        assert np.array_equal(x, y)
        json.dumps(rep, allow_nan=False)


def test_stereo_linked_and_limits():
    x = np.stack([tone(), tone() * .5], axis=1)
    y, rep = level(x, SR, {'words': [{'start': 1, 'end': 2}]}, target=-10, max_gain=2)
    assert rep['events'][0]['gain_db'] == 2
    np.testing.assert_allclose(y[:, 0], y[:, 1] * 2)
    assert rep['events'][0]['gain_limited']


def test_peak_limit_checks_ramp_support():
    x = tone(amplitude=.01)
    x[int(.9 * SR)] = .88  # in the fade, not in the phrase
    y, rep = level(x, SR, {'words': [{'start': 1, 'end': 2}]}, max_gain=30)
    assert rep['events'][0]['headroom_limited']
    assert np.max(np.abs(y)) <= 10 ** (-1 / 20) + 1e-12


def test_breath_keep_splits_close_phrases_and_is_exact():
    x = tone()
    timing = {'words': [{'start': .5, 'end': 1}, {'start': 1.2, 'end': 2}],
              'breaths': [{'start': 1.05, 'end': 1.15}]}
    y, rep = level(x, SR, timing, ramp=.01)
    assert len(rep['events']) == 3
    assert np.array_equal(y[int(1.05 * SR):int(1.15 * SR)], x[int(1.05 * SR):int(1.15 * SR)])
    z, rep = level(x, SR, timing, breath_policy='attenuate', ramp=.01)
    a, b = int(1.05 * SR), int(1.15 * SR)
    np.testing.assert_allclose(z[a:b], x[a:b] * 10 ** (-6 / 20))


@pytest.mark.parametrize('words,breaths', [
    ([{'start': -1, 'end': 1}], []), ([{'start': 1, 'end': 5}], []),
    ([{'start': 1, 'end': 2}, {'start': 1.5, 'end': 3}], []),
    ([{'start': float('nan'), 'end': 2}], []),
    ([{'start': 1, 'end': 2}], [{'start': 1.9, 'end': 2.2}]),
    ([{'start': 1, 'end': 1}], []),
])
def test_invalid_intervals_rejected(words, breaths):
    with pytest.raises(ValueError):
        level(tone(), SR, {'words': words, 'breaths': breaths})


@pytest.mark.parametrize('options', [{'ramp': 0}, {'max_gain': -1}, {'target': float('inf')},
                                     {'breath_db': 2}, {'breath_policy': 'delete'}, {'tolerance': -1}])
def test_invalid_options(options):
    with pytest.raises(ValueError):
        level(tone(), SR, {'words': []}, **options)


def test_touching_words_share_gain_and_fractional_edges_protected():
    x = tone()
    y, rep = level(x, SR, {'words': [{'start': .5 + .2 / SR, 'end': 1},
                                   {'start': 1, 'end': 2}]})
    assert len(rep['events']) == 1
    a, b = rep['events'][0]['start_sample'], rep['events'][0]['end_sample']
    np.testing.assert_allclose(y[a:b], x[a:b] * 10 ** (rep['events'][0]['gain_db'] / 20))


def test_cli_roundtrip_preserves_unmodified_pcm_and_report(tmp_path):
    src, dst, timings, report = [tmp_path / s for s in ('in.wav', 'out.wav', 'words.json', 'report.json')]
    sf.write(src, tone(), SR, subtype='PCM_24')
    timings.write_text(json.dumps({'words': [{'start': 1, 'end': 2}]}))
    assert main(['leveller', str(src), str(dst), '--timings', str(timings), '--report', str(report)]) == 0
    x, sr = sf.read(src, dtype='int32')
    y, _ = sf.read(dst, dtype='int32')
    rep = json.loads(report.read_text())
    assert sr == SR and sf.info(dst).subtype == 'PCM_24'
    for a, b in rep['untouched_ranges']:
        assert np.array_equal(y[a:b], x[a:b])
    assert rep['schema'] == 'opendubstage.leveller/0.1'
    with pytest.raises(SystemExit):
        main(['leveller', str(src), str(src), '--timings', str(timings), '--report', str(report)])


def test_touching_breath_boundaries_skip_rides_without_pause():
    x = np.full(3 * SR, .03)
    timings = {'words': [{'start': .5, 'end': 1}, {'start': 1.1, 'end': 2}],
               'breaths': [{'start': 1, 'end': 1.1}]}
    for policy in ('keep', 'attenuate'):
        y, report = level(x, SR, timings, breath_policy=policy)
        assert np.array_equal(x, y)
        for event in report['events']:
            if event['requested_gain_db'] != 0:
                assert event['insufficient_pause']
                assert event['reason'] == 'insufficient_pause'
                assert event['gain_db'] == 0


def test_recording_edges_do_not_require_external_ramps():
    x = tone(1)
    y, report = level(x, SR, {'words': [{'start': 0, 'end': 1}]})
    event = report['events'][0]
    assert event['gain_db'] != 0 and not event['insufficient_pause']
    np.testing.assert_allclose(y, x * 10 ** (event['gain_db'] / 20))


def test_short_nonzero_pause_does_not_substitute_one_sample_ramp():
    x = tone(3)
    timings = {'words': [{'start': .5, 'end': 1}, {'start': 1.1 + 4 / SR, 'end': 2}],
               'breaths': [{'start': 1 + 2 / SR, 'end': 1.1 + 2 / SR}]}
    y, report = level(x, SR, timings, breath_policy='attenuate')
    assert np.array_equal(x, y)
    assert all(e['insufficient_pause'] for e in report['events'])
    assert all(e['gain_db'] == 0 for e in report['events'])
