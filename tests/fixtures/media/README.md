# Media fixtures

Placeholder recordings whose **names** are the fixture. None of them is a real
video file, and that is deliberate.

## Why there are no real videos here

Stage 09 needs to test three things about where a timestamp comes from:

1. parsing a timestamp out of a filename — needs only the name;
2. falling back to a weaker source when a stronger one is unavailable, and
   reporting the reduced reliability — needs only the absence of the stronger
   one;
3. reading a container's creation time — needs a demuxer.

The demuxer arrives in **stage 10**, which introduces the video-source
abstraction and the dependency that can open an MP4. Committing a real MP4 now
would let stage 09 test a decoder it does not own, against a library it does not
depend on, and the test would belong to stage 10 anyway.

So the container path is tested through its seam: `select_source` is given the
creation time a stage-10 reader will supply, or given `None` to stand for
metadata a copy stripped. When stage 10 lands, the real probe plugs in behind
that seam and these tests keep their meaning.

## What each file is for

| file | what it exercises |
|---|---|
| `cam_01_20260810_142211.mp4` | the compact vendor naming, `YYYYMMDD_HHMMSS` |
| `cam_03-2026-08-10T14-22-11.mkv` | the ISO-ish naming the same vendor writes after a firmware update |
| `recording_2026_08_10_14_22_11.avi` | a third separator convention |
| `front_door_clip.mp4` | a name carrying no timestamp: must fail loudly, not guess |
| `cam_07_20261025_013000.mp4` | a local time inside the DST fall-back hour: must refuse to pick a reading |

All five parse to the same instant — 2026-08-10T14:22:11Z — except the last two,
which are the failure cases.
