# OpenDubStage art

All artwork here is original and offered under CC0 1.0 Universal. You may copy, modify, and use it without attribution. The SVGs were drawn for this brief; the two stage scenes were generated with OpenAI's built-in image tool and composited locally with Pillow. No existing logos or photographs were used.

| File | How it was made | Licence |
| --- | --- | --- |
| `logo.svg` | Hand-drawn SVG stage arch with five console faders | CC0 1.0 |
| `logo-512.png` | Pillow raster rendering of the same mark at 512×512 | CC0 1.0 |
| `logo-128.png` | Pillow raster rendering of the same mark at 128×128 | CC0 1.0 |
| `wordmark.svg` | Hand-drawn SVG mark and live text; light lettering on dark backgrounds, with a `prefers-color-scheme` light-mode switch | CC0 1.0 |
| `wordmark-dark.svg` | Hand-drawn SVG mark and dark lettering for light backgrounds | CC0 1.0 |
| `social-preview.png` | OpenAI built-in image-generated stage scene, cropped and branded with Pillow; 1280×640 | CC0 1.0 |
| `readme-banner.png` | OpenAI built-in image-generated panoramic stage scene, cropped and branded with Pillow; 1600×400 | CC0 1.0 |
| `icons/restore.svg` | Hand-drawn SVG waveform and repair sparkle | CC0 1.0 |
| `icons/match.svg` | Hand-drawn SVG paired voice contours | CC0 1.0 |
| `icons/mix.svg` | Hand-drawn SVG three-fader mixer | CC0 1.0 |
| `icons/qc.svg` | Hand-drawn SVG meter and check | CC0 1.0 |
| `icons/foley.svg` | Hand-drawn SVG still image and sound lines | CC0 1.0 |
| `source/social-stage.png` | Unbranded source scene from OpenAI's built-in image tool | CC0 1.0 |
| `source/banner-stage.png` | Unbranded source scene from OpenAI's built-in image tool | CC0 1.0 |
| `build_assets.py` | Local Pillow assembly script for the PNG deliverables | CC0 1.0 |

The palette is near-black `#10191B`, amber `#D7A25D` and `#F1BE75`, and teal `#426F70`. The mark uses a proscenium arch around five fader caps; the fader heights make a quiet audio contour.

## Image-generation prompts

These are the final background prompts. The lettering and mark were added afterward, so the generated images contain no intended text.

**Social scene:** Original cinematic illustration of an empty film re-recording dub stage, seen from slightly above and behind a wide analog mixing console, with a large dark projection screen, subtle acoustic wall panels, amber meter lights, and muted teal reflections. Wide 2:1 scene; calm, dark left third for branding; console concentrated in the lower right; premium editorial digital illustration with subtle grain. No people, text, letters, symbols, logos, watermarks, or readable labels.

**Banner scene:** Original panoramic editorial illustration of an empty film dub stage and mixing console, with long receding fader rows flowing toward a dark theatrical arch and screen. Amber meter lights and a restrained teal acoustic line. Ultra-wide 4:1 crop-safe composition; quiet, dark middle-left for branding; console and architecture toward the right. Near-black charcoal, restrained amber and teal, subtle analog grain. No people, text, lettering, logos, symbols, watermarks, or readable labels.

To rebuild the PNGs with the supplied source scenes, run `python3 out/build_assets.py` from the project root. This uses the macOS Avenir Next font for the raster wordmark and requires Pillow.
