# Preserved student pilot — paused

These are the existing scripts from the completed paired student experiment. They are retained for continuity and are not part of the current MedGemma CLI. No new student training was performed during repository assembly.

They require a separate environment with PyTorch 2.8.0 and Ultralytics 8.4.150, plus the data and teacher run paths passed explicitly on the command line. Check the upstream Ultralytics license before redistribution. Model files and per-study outputs are kept in private `state/runs/student-pilot26/`.

The image encoder was frozen. Only the study aggregation head was trained. Mean held-out AUC was 0.615959 for labels-only and 0.607098 with distillation; no benefit was observed in this small comparison.
