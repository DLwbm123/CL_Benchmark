# Continual medical segmentation adapters and ongoing five-seed study

**Correction: S3R/TED probability-BCE saturation and TED optimizer configuration were audited and corrected. Previous three completed runs are preserved as invalid attempts, not evidence against the original methods. See results/20261026/DOMAIN_COLLAPSE_AUDIT_ZH.md. The v3 implementation passed162 CPU tests and a bounded8-update train-only qualification; full corrected runs are pending under original clocks.**

This branch adds research adapters to the existing public benchmark. It is an ongoing campaign: Eleven reviewed runs are retained under their explicitly disclosed adapter protocols, including Class/Organ-CLIP-CIS seeds1/2; three formerly completed S3R/TED runs have been invalidated by the numerical audit and requeued under v3 with their original deadlines. Historical reviewed combinations (plus Organ-LDKA seed1) include: S3R, TED, Class-CLIP-CIS, Organ-CLIP-CIS, LGKD, NeST, Class-EWF and Organ-EWF. The old S3R/TED scores reflect invalid numerical implementations/configurations; Organ-CLIP-CIS shows substantial forgetting, not evidence that all methods/settings/seeds finished. See results/20261026/CIT_ORGAN_SEED1_RESULT_ZH.md, CIT_SEED1_RESULT_ZH.md, S3R_SEED2_RESULT_ZH.md, EWF_SEED1_RESULT_ZH.md, SEED1_FOUR_METHODS_RESULT_ZH.md and the separate S3R/TED negative-result reports. No five-seed mean or standard deviation is available yet.

Paper-derived 2D adapters and Organ-CL extensions are explicitly distinct from native author implementations. See README_natural_ciss.md, README_medical_methods.md and THIRD_PARTY_NOTICES.md. Numerical safeguards and optimizer-alignment changes are disclosed individually in results/20261026. The additions do not grant rights beyond applicable upstream licenses.

Training uses 150 epochs per task, batch8 and seeds1..5. Current results preserve interrupted/failed attempt costs and original per-seed24-hour clocks; failures are not counted as independent new experiments. Patient data, images, patient identities, model weights, CLIP weights/embeddings, server addresses and credentials are excluded. Supply data and dependencies separately; this is a source supplement to the existing benchmark, not a standalone data distribution.

The test/validation distinction, background-inclusive metrics, legacy ASD handling, task-masked versus whole-heart metrics and unavailable normalized RMA must be considered when interpreting results. A background-only predictor can appear near0.5 macro Dice; the completed S3R seed1 should not be described as successful continual segmentation.

Publication-package validation covered154 synthetic tests. The initial package had140 passes and14 integration failures caused by omitted matching logger/metric modules; after including these existing modules, all41 affected integration/Organ tests passed. A private server-scheduler test is excluded with its server-control implementation. These packaging failures did not affect live training.

ReBac now uses a disclosed incremental-only gradient cap1 under the same original run clocks; see results/20261026/REBAC_GRADIENT_GUARD_V2_ZH.md. Its seed1 failure and seed2 protocol-switch interruption remain archived separately. This revision passed32 relevant synthetic tests including the new guard persistence test; no full long-run stability claim is made.

Organ-LDKA seed1 completed with macro Dice0.26605 and all66750 incremental steps clipped. This is weak performance from the disclosed paper-derived Organ adapter, not an author-native result; see results/20261026/LDKA_ORGAN_SEED1_RESULT_ZH.md.

Class-CLIP-CIS seed2 is recorded in results/20261026/CLIP_CIS_SEED2_RESULT_ZH.md; task-conditioned and whole-heart numbers remain distinct, with only2/5seeds available.

Organ-CLIP-CIS seed2 is recorded in results/20261026/CLIP_CIS_ORGAN_SEED2_RESULT_ZH.md; both available seeds show forgetting, with no five-seed claim.
