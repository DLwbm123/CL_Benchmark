# Continual medical segmentation adapters and ongoing five-seed study

This branch adds research adapters to the existing public benchmark. It is an ongoing campaign: S3R seed1 and TED seed1 have completed review (2/70 runs); both are negative results, not evidence that all methods/settings/seeds finished. See results/20261026/S3R_SEED1_RESULT_ZH.md and results/20261026/TED_SEED1_RESULT_ZH.md. No five-seed mean or standard deviation is available yet.

Paper-derived 2D adapters and Organ-CL extensions are explicitly distinct from native author implementations. See README_natural_ciss.md, README_medical_methods.md and THIRD_PARTY_NOTICES.md. Numerical safeguards and optimizer-alignment changes are disclosed individually in results/20261026. The additions do not grant rights beyond applicable upstream licenses.

Training uses 150 epochs per task, batch8 and seeds1..5. Current results preserve interrupted/failed attempt costs and original per-seed24-hour clocks; failures are not counted as independent new experiments. Patient data, images, patient identities, model weights, CLIP weights/embeddings, server addresses and credentials are excluded. Supply data and dependencies separately; this is a source supplement to the existing benchmark, not a standalone data distribution.

The test/validation distinction, background-inclusive metrics, legacy ASD handling, task-masked versus whole-heart metrics and unavailable normalized RMA must be considered when interpreting results. A background-only predictor can appear near0.5 macro Dice; the completed S3R seed1 should not be described as successful continual segmentation.

Publication-package validation covered154 synthetic tests. The initial package had140 passes and14 integration failures caused by omitted matching logger/metric modules; after including these existing modules, all41 affected integration/Organ tests passed. A private server-scheduler test is excluded with its server-control implementation. These packaging failures did not affect live training.
