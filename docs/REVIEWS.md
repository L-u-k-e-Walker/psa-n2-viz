# Reviews

Every mock and draft, what changed, and the user's verdict. `render.py --final vNN`
runs only if the row for vNN says **approved: final vNN** (the user's words).

| Version | Tier | Date | Config | Files | Changes since last version | Render time | Verdict |
|---|---|---|---|---|---|---|---|
| v01 | mock | 2026-11-20 | config_3962d860 | previews/v01/contact_v01_mock_A.png, _B.png, _C.png | First mock: three layout options. A: plant on the left two thirds with profile strips beside each bed, dashboard column on the right. B: beds with one shared profile block between them (bed 1 solid, bed 2 dashed), dashboard on the right. C: wide plant on top with profile strips, time panels in a strip along the bottom. Physics is preliminary: the 60-cell sizing state at 1.16 m CMS (product O2 ~3,000 ppm); the 180-cell final is still being computed | 19.9 s (3 layouts x 8 stills, incl. the frame cache) | **layout B** (user, 2026-11-21) |
