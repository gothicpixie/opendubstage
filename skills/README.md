# OpenDubStage skills

Claude Code skills for documentary and film sound post-production, distilled from a research
pass over public-radio practice (NPR Training, Transom, This American Life), feature-documentary
sound interviews (CAS, MPSE, Emmy, Cinema Eye, BAFTA Factual, Third Coast winners), the
controlled ducking study (Torcoli et al., AES 2019), delivery specs (PRSS, ATSC A/85, EBU R128,
Netflix) and a feature-length mix built and measured with the OpenDubStage tools. Every rule
has a number or a test; every number has a source or says "measured in practice".

Each skill is a folder with a `SKILL.md` (YAML frontmatter with `name` and a `description` that
says when to load it, then a concise body of rules, numbers, procedures and checks) and, where
useful, `references/*.md` for detail loaded on demand.

## Index

| Skill | Use it when | OpenDubStage module / CLI |
|---|---|---|
| [`documentary-sound-design`](documentary-sound-design/SKILL.md) | planning or auditing the sound of a documentary, narrated stills film, oral history or long-form podcast; the order of operations (audio first) and the eighteen principles; the map to the other skills | all |
| [`dialogue-editing-and-restoration`](dialogue-editing-and-restoration/SKILL.md) | cutting narration with measured air, breaths and laughs kept, room tone, spectral repair of coughs/thumps/clicks, de-rumble and chain order, sentence levelling, what neural restoration may and may not do | `opendubstage.restore` · `opendubstage specrepair` |
| [`voice-matching`](voice-matching/SKILL.md) | LTAS EQ match, bridging a TTS or cloned line into a recording, riding and placing an off-mic second speaker, phone matching, when voices should stay different | `opendubstage.match` · `opendubstage ltasmatch` |
| [`music-under-narration`](music-under-narration/SKILL.md) | spotting, entries and exits, posts and buttons, duck depth and the 1–4 kHz carve, the music budget, instrumentals instead of sung vocals, song-relative levels, key lines, the ending ladder | `opendubstage.mix.carve` · `opendubstage carve` |
| [`ambience-and-sound-for-stills`](ambience-and-sound-for-stills/SKILL.md) | beds and gap fill (never dead air), the layer hierarchy, spot-effect density and timing, no-sound classes, anti-cartoon rules, the witness test | `opendubstage.foley` · `opendubstage foley` (analyze / plan / check / generate / render) |
| [`scene-transitions`](scene-transitions/SKILL.md) | J-cuts, L-cuts, music seams, live edges, act breaths, key-line ladders, designed and memory-POV silences, the ending ladder | `opendubstage.mix.transitions` · `opendubstage transitions` |
| [`mix-qc-and-delivery`](mix-qc-and-delivery/SKILL.md) | the QC gate before any picture render: dialogue anchor, true peak at 4× oversampling, LRA, dead air, clicks, masking, jumps, the transcription check of every cut, half-volume and multi-system listens; delivery specs | `opendubstage.qc` · `opendubstage qc` |
| [`ai-review-of-audio`](ai-review-of-audio/SKILL.md) | a multimodal model as a second pair of ears: briefs, schema, chunking, blind A/B with position-bias controls, hidden-vocal checks, cross-checking against the QC gate | (your own review script; schema in the skill) |

Shared citation list: [`documentary-sound-design/references/sources.md`](documentary-sound-design/references/sources.md).

## Install

The skills are plain folders; Claude Code loads any folder containing a `SKILL.md` from
`~/.claude/skills/` (user-wide) or `.claude/skills/` (per project). Pick one:

**A. Symlink from a clone (recommended; updates with `git pull`)**

```sh
git clone https://github.com/gothicpixie/opendubstage.git
cd opendubstage
for d in skills/*/; do
  name=$(basename "$d")
  ln -sfn "$(pwd)/$d" "$HOME/.claude/skills/$name"
done
```

**B. Copy into a project**

```sh
mkdir -p .claude/skills
cp -R /path/to/opendubstage/skills/* .claude/skills/
```

**C. Plugin marketplace** (when the repository ships a `.claude-plugin/marketplace.json`)

```
claude plugin marketplace add gothicpixie/opendubstage
/plugin            # browse and enable the skills you want
```

Enabling only the skills you need keeps each session's skill list small. Start a new Claude
Code session after installing; the available-skills list should show the eight names above.

## Conventions

- Frontmatter: `name` (matches the folder) and `description` (states precisely when to use the
  skill and when not to, because the description is what triggers loading).
- Bodies are rules, numbers, procedures and checks; long detail lives in `references/`.
- Levels are stated relative to dialogue at −24 LUFS; convert for other anchors.
- Where a number here and a constant in the OpenDubStage code disagree, the code wins and the
  skill needs a line changed. Open an issue or a pull request with the measurement.

## Verifying this folder

```sh
python3 - <<'EOF'
import yaml, pathlib, re, sys
ok = True
for p in sorted(pathlib.Path("skills").glob("*/SKILL.md")):
    fm = p.read_text().split("---")[1]
    d = yaml.safe_load(fm)
    assert d["name"] == p.parent.name and d["description"].strip(), p
    for m in re.finditer(r"\]\(([^)#]+)\)", p.read_text()):
        t = m.group(1)
        if not t.startswith("http") and not (p.parent / t).exists(): print("broken link", p, t); ok = False
print("ok" if ok else "FAIL"); sys.exit(0 if ok else 1)
EOF
```
