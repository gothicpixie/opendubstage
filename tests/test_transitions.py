"""Synthetic tests for opendubstage.mix.transitions. Run: python -m pytest tests/test_transitions.py"""
import copy
import sys
import unittest
from pathlib import Path

from opendubstage.mix import transitions as T

DUR = {"a.wav": 600.0, "b.wav": 600.0, "songA.mp3": 200.0, "songB.mp3": 200.0, "short.mp3": 50.0,
       "room1.flac": 1200.0, "room2.flac": 1200.0, "cam.mts": 400.0, "cam0.mts": 400.0, "vo.wav": 100.0}
CH = [{"t0": 0.0}, {"t0": 100.0}, {"t0": 200.0}]


def sheet(**kw):
    s = {"total": 300.0, "dx": [], "live": [], "mx": [], "amb": []}
    s.update(kw)
    return s


def mx(src, at, dur, ss=0.0, mode="under", fi=0.0, fo=3.0):
    return {"src": src, "ss": ss, "dur": dur, "at": at, "mode": mode, "fade_in": fi, "fade_out": fo}


def amb(src, at, dur, ss=100.0, fade=4.0):
    return {"src": src, "ss": ss, "dur": dur, "at": at, "fade": fade}


def live(src, at, dur, ss=50.0, fade=0.4, **kw):
    return {"src": src, "ss": ss, "dur": dur, "at": at, "fade": fade, **kw}


def run(s, **kw):
    log = []
    out = T.transitions(s, CH, src_dur=DUR, log=log, **kw)
    return out, log


class TestTransitions(unittest.TestCase):
    def test_input_not_mutated_and_total_kept(self):
        s = sheet(amb=[amb("room1.flac", 0, 100), amb("room2.flac", 100, 100)])
        before = copy.deepcopy(s)
        out, _ = run(s)
        self.assertEqual(s, before)
        self.assertEqual(out["total"], 300.0)

    def test_ambience_prelap_and_tail(self):
        s = sheet(amb=[amb("room1.flac", 0, 100), amb("room2.flac", 100, 100)])
        out, _ = run(s)
        a, b = out["amb"]
        self.assertAlmostEqual(T.span("amb", a)[1], 103.0, places=2)           # L-cut tail 3 s
        self.assertAlmostEqual(T.span("amb", b)[0], 98.0, places=2)            # J-cut 2 s
        self.assertAlmostEqual(b["ss"], 98.0, places=2)                        # sample still lands at the same film time
        self.assertGreaterEqual(b["fade_in"], 2.0)
        self.assertEqual(b["fade_out"], 4.0)                                   # other end keeps its fade
        self.assertEqual(T.bridge_gaps(out, CH, DUR), [])                      # room2 tails across 200 too

    def test_live_uses_pre_post_and_keeps_sync(self):
        s = sheet(live=[live("cam.mts", 100, 100, ss=50, fade=0.6)])
        out, _ = run(s)
        c = out["live"][0]
        self.assertEqual((c["at"], c["ss"], c["dur"]), (100, 50, 100))      # sync point untouched
        self.assertGreaterEqual(c["pre"], 2.0)
        self.assertGreaterEqual(c["post"], 2.0)
        self.assertGreaterEqual(c["fade_in"], 2.0)
        self.assertGreaterEqual(c["fade_out"], 2.0)

    def test_live_prelap_limited_by_source_start(self):
        s = sheet(live=[live("cam0.mts", 100, 50, ss=0.0)])
        out, _ = run(s)
        self.assertEqual(T.span("live", out["live"][0])[0], 100.0)             # no audio before the source start
        self.assertGreaterEqual(out["live"][0]["fade_in"], 2.0)

    def test_live_tail_stops_before_next_scene_dialogue(self):
        s = sheet(dx=[{"src": "vo.wav", "ss": 0, "dur": 10, "at": 101.0}], live=[live("cam.mts", 50, 50)])
        out, _ = run(s)
        self.assertLessEqual(T.span("live", out["live"][0])[1], 101.0 - 0.25 + 1e-6)
        self.assertEqual(out["dx"], s["dx"])

    def test_music_tail_into_next_scene(self):
        s = sheet(mx=[mx("songA.mp3", 0, 100, fo=3.0)], amb=[amb("room2.flac", 100, 100)])
        out, _ = run(s)
        m = out["mx"][0]
        self.assertAlmostEqual(T.span("mx", m)[1], 103.0, places=2)
        self.assertGreaterEqual(m["fade_out"], 4.5)

    def test_music_crossfade_equal_window(self):
        s = sheet(mx=[mx("songA.mp3", 0, 100, ss=10), mx("songB.mp3", 100, 80, ss=30)])
        out, _ = run(s)
        a, b = out["mx"]
        b0, a1 = T.span("mx", b)[0], T.span("mx", a)[1]
        self.assertAlmostEqual(a1 - b0, 3.0, places=2)                         # 3 s overlap
        self.assertAlmostEqual(b0, 98.5, places=2)                             # centred: mid-song entry pre-laps 1.5 s
        self.assertGreaterEqual(b["fade_in"], 3.0)
        self.assertLessEqual(a["fade_out"], 6.0); self.assertGreaterEqual(a["fade_out"], 3.0)
        self.assertAlmostEqual(b["ss"] + (100 - b["at"]), 30.0, places=2)      # downbeat still at the cut

    def test_up_songs_get_gap_not_crossfade(self):
        s = sheet(mx=[mx("songA.mp3", 0, 100, mode="up"), mx("songB.mp3", 100, 80, mode="up")])
        out, _ = run(s)
        a, b = out["mx"]
        self.assertLessEqual(T.span("mx", a)[1], 100 - 0.5 + 1e-6)
        self.assertGreaterEqual(T.span("mx", a)[1], 100 - 1.0 - 1e-6)
        self.assertEqual(b["at"], 100)                                          # up song keeps its sync
        self.assertGreaterEqual(a["fade_out"], 2.0); self.assertGreaterEqual(b["fade_in"], 1.5)

    def test_same_recording_split_is_spliced(self):
        s = sheet(mx=[mx("songA.mp3", 50, 51, ss=0), mx("songA.mp3", 100, 60, ss=50)])
        out, log = run(s)
        a, b = out["mx"]
        self.assertTrue(any("splice" in x for x in log))
        self.assertLessEqual(T.span("mx", a)[1] - T.span("mx", b)[0], 0.1)
        self.assertEqual(T.check_transitions(out, CH, src_dur=DUR), [])

    def test_never_past_source(self):
        s = sheet(mx=[mx("short.mp3", 60, 40, ss=9)], live=[live("cam.mts", 100, 100, ss=299)])  # 1 s left / 1 s left
        out, _ = run(s)
        m, l = out["mx"][0], out["live"][0]
        self.assertLessEqual(m["ss"] + m["dur"], 50.0)
        self.assertLessEqual(l["ss"] + l["dur"] + l.get("post", 0), 400.0)
        self.assertGreaterEqual(m["fade_out"], 1.5)

    def test_overlong_cue_is_clamped_so_its_fade_lands_on_audio(self):
        s = sheet(mx=[mx("short.mp3", 10, 80)])                                 # asks for 80 s of a 50 s file
        out, _ = run(s)
        self.assertLessEqual(out["mx"][0]["dur"], 50.0)

    def test_dialogue_untouched(self):
        dx = [{"src": "vo.wav", "ss": 0, "dur": 20, "at": 85.0}, {"src": "vo.wav", "ss": 20, "dur": 20, "at": 105.0}]
        s = sheet(dx=copy.deepcopy(dx), amb=[amb("room1.flac", 0, 100), amb("room2.flac", 100, 100)],
                  mx=[mx("songA.mp3", 0, 100)])
        out, _ = run(s)
        self.assertEqual(out["dx"], dx)

    def test_programme_edges_respected(self):
        s = sheet(amb=[amb("room1.flac", 0, 100, ss=0.5), amb("room2.flac", 100, 200)])
        out, _ = run(s)
        for k, c in T._all(out):
            a, b = T.span(k, c)
            self.assertGreaterEqual(a, 0.0); self.assertLessEqual(b, 300.0 + 1e-6)

    def test_chained_clips_left_alone(self):
        c1, c2 = live("cam.mts", 10, 20, ss=100), live("cam.mts", 29.65, 20, ss=130)
        out, _ = run(sheet(live=[c1, c2]))
        self.assertEqual(out["live"][0].get("post", 0), 0)                     # successor continues: no tail added
        self.assertGreaterEqual(out["live"][1]["fade_out"], 1.5)               # last clip gets a real fade
        self.assertGreaterEqual(out["live"][1].get("post", 0), 0.99)

    def test_check_flags_hard_edges_and_pass_fixes_them(self):
        s = sheet(mx=[mx("songA.mp3", 0, 100, fo=0.2), mx("songB.mp3", 200, 50, ss=40)],
                  live=[live("cam.mts", 100, 100, fade=0.3)])
        probs = T.check_transitions(s, CH, src_dur=DUR)
        self.assertGreaterEqual(len(probs), 3)
        out, _ = run(s)
        self.assertEqual(T.check_transitions(out, CH, src_dur=DUR), [])
        self.assertEqual(T.bridge_gaps(out, CH, DUR), [])

    def test_idempotent_enough(self):
        s = sheet(mx=[mx("songA.mp3", 0, 100, ss=10), mx("songB.mp3", 100, 80, ss=30)],
                  amb=[amb("room1.flac", 0, 100), amb("room2.flac", 100, 100)])
        once, _ = run(s)
        twice = T.transitions(once, CH, src_dur=DUR)
        self.assertEqual(T.check_transitions(twice, CH, src_dur=DUR), [])
        for k, c in T._all(twice):                                              # no runaway growth
            self.assertLessEqual(T.span(k, c)[1] - T.span(k, c)[0], 220.0)

    def test_settings_are_parameters(self):
        s = sheet(amb=[amb("room1.flac", 0, 100), amb("room2.flac", 100, 100)])
        out, _ = run(s, settings=T.Settings(prelap=3.0, tail=1.5))
        self.assertAlmostEqual(T.span("amb", out["amb"][1])[0], 97.0, places=2)
        self.assertAlmostEqual(T.span("amb", out["amb"][0])[1], 101.5, places=2)


if __name__ == "__main__":
    unittest.main()
