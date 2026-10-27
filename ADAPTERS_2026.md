# Continual medical segmentation adapters and ongoing five-seed study

**Correction: S3R/TED probability-BCE saturation and TED optimizer configuration were audited and corrected. Previous three completed runs are preserved as invalid attempts, not evidence against the original methods. See results/20261026/DOMAIN_COLLAPSE_AUDIT_ZH.md. The v3 implementation passed162 CPU tests and a bounded8-update train-only qualification; S3R seed1 has now completed under v3 with its original clock; other corrected runs remain ongoing.**

This branch adds research adapters to the existing public benchmark. It is an ongoing campaign: Thirty-one reviewed runs are retained under their explicitly disclosed adapter protocols, including Class/Organ-CLIP-CIS seeds1/2; three formerly completed S3R/TED runs have been invalidated by the numerical audit and requeued under v3 with their original deadlines. Historical reviewed combinations (plus Organ-LDKA seed1) include: S3R, TED, Class-CLIP-CIS, Organ-CLIP-CIS, LGKD, NeST, Class-EWF and Organ-EWF. The old S3R/TED scores reflect invalid numerical implementations/configurations; Organ-CLIP-CIS shows substantial forgetting, not evidence that all methods/settings/seeds finished. See results/20261026/CIT_ORGAN_SEED1_RESULT_ZH.md, CIT_SEED1_RESULT_ZH.md, S3R_SEED2_RESULT_ZH.md, EWF_SEED1_RESULT_ZH.md, SEED1_FOUR_METHODS_RESULT_ZH.md and the separate S3R/TED negative-result reports. No five-seed mean or standard deviation is available yet.

Paper-derived 2D adapters and Organ-CL extensions are explicitly distinct from native author implementations. See README_natural_ciss.md, README_medical_methods.md and THIRD_PARTY_NOTICES.md. Numerical safeguards and optimizer-alignment changes are disclosed individually in results/20261026. The additions do not grant rights beyond applicable upstream licenses.

Training uses 150 epochs per task, batch8 and seeds1..5. Current results preserve interrupted/failed attempt costs and original per-seed24-hour clocks; failures are not counted as independent new experiments. Patient data, images, patient identities, model weights, CLIP weights/embeddings, server addresses and credentials are excluded. Supply data and dependencies separately; this is a source supplement to the existing benchmark, not a standalone data distribution.

The test/validation distinction, background-inclusive metrics, legacy ASD handling, task-masked versus whole-heart metrics and unavailable normalized RMA must be considered when interpreting results. A background-only predictor can appear near0.5 macro Dice; the completed S3R seed1 should not be described as successful continual segmentation.

Publication-package validation covered154 synthetic tests. The initial package had140 passes and14 integration failures caused by omitted matching logger/metric modules; after including these existing modules, all41 affected integration/Organ tests passed. A private server-scheduler test is excluded with its server-control implementation. These packaging failures did not affect live training.

ReBac now uses a disclosed incremental-only gradient cap1 under the same original run clocks; see results/20261026/REBAC_GRADIENT_GUARD_V2_ZH.md. Its seed1 failure and seed2 protocol-switch interruption remain archived separately. This revision passed32 relevant synthetic tests including the new guard persistence test; no full long-run stability claim is made.

Organ-LDKA seed1 completed with macro Dice0.26605 and all66750 incremental steps clipped. This is weak performance from the disclosed paper-derived Organ adapter, not an author-native result; see results/20261026/LDKA_ORGAN_SEED1_RESULT_ZH.md.

Class-CLIP-CIS seed2 is recorded in results/20261026/CLIP_CIS_SEED2_RESULT_ZH.md; task-conditioned and whole-heart numbers remain distinct, with only2/5seeds available.

Organ-CLIP-CIS seed2 is recorded in results/20261026/CLIP_CIS_ORGAN_SEED2_RESULT_ZH.md; both available seeds show forgetting, with no five-seed claim.

Reporting follows the manuscript metrics and benchmarking purpose; see results/20261026/BENCHMARK_REPORTING_POLICY_ZH.md. Legacy CSV backward_transfer already calculates BWTR; this terminology correction changes no measured values or training. Poor scores alone do not authorize tuning. Legacy storage efficiency is not silently renamed DRR.

Class-CIT seed2 completed under the frozen source-optimizer-v2 adapter: task-conditioned A-Dice0.8066008770, BWTR−0.0178707669; whole-heart slice Dice0.8040923066. Only2/5 seeds are complete; see results/20261026/CIT_SEED2_RESULT_ZH.md.

Class-LDKA seed1 and ReBac seeds1/2 are reviewed with their disclosed gradient guards. Weak scores are retained without tuning; detailed BWTR, task-conditioned/whole-heart metrics, guard counts and original costs are in results/20261026/LDKA_REBAC_RESULTS_ZH.md. No combination has all five seeds.

Corrected S3R seed1 v3 completed all six tasks: A-Dice0.6572465192, BWTR−0.1867492093, E-FWT0.2824687704, finite checkpoints and nonzero importance. Old probability-BCE results stay invalid and separate; see results/20261026/S3R_SEED1_V3_RESULT_ZH.md. Only1/5 valid S3R seeds is complete.

Organ-EWF seed2 and PCDD seed1 are reviewed: see results/20261026/EWF_ORGAN_PCDD_RESULTS_ZH.md for separate metrics, original elapsed costs and PCDD guard statistics. Results are retained without score-driven tuning; no five-seed completion claim.

TED seed1 v3 and Class-LDKA seed2 completed under frozen protocols; see results/20261026/TED_V3_LDKA2_RESULTS_ZH.md. Weak scores and guard statistics are retained; old invalid TED attempts remain excluded. No five-seed complete combination yet.

Class-EWF seed2 is reviewed under the frozen protocol: A-Dice0.5454974024, BWTR0.0010767432, whole-heart slice Dice0.5907598901. See results/20261026/EWF_SEED2_RESULT_ZH.md; only2/5 seeds are complete.

Organ-CIT seed2 and corrected S3R seed2 v3 are reviewed; each combination has2/5 valid seeds. Metrics and original costs are in results/20261026/CIT_ORGAN_S3R2_RESULTS_ZH.md; old probability-BCE S3R results remain invalid and separate.

LGKD seed2 failed with a non-finite loss at the start of task3. It is retained separately, not counted complete or automatically retried; bounded train-only audit and source-recipe differences are documented in results/20261026/LGKD_SEED2_FAILURE_ZH.md.

Corrected S3R seed3 and TED seed2 are reviewed under v3: see results/20261026/S3R3_TED2_RESULTS_ZH.md. S3R has3/5 and TED2/5 valid seeds; LGKD seed2 remains a separate failure.

Organ-LDKA seed2 completed with weak near-static results and all66750 incremental steps clipped; see results/20261026/LDKA_ORGAN_SEED2_RESULT_ZH.md. Scores and guard statistics are retained without tuning; only2/5 seeds are complete.

Class-CLIP-CIS seed3 and NeST seed2 are reviewed: see results/20261026/CLIP3_NEST2_RESULTS_ZH.md for separate task-conditioned/whole-heart metrics. No five-seed complete claim.

Organ-CLIP-CIS seed3 is reviewed with substantial forgetting retained: see results/20261026/CLIP_CIS_ORGAN_SEED3_RESULT_ZH.md. Only3/5 seeds complete; no tuning or selective exclusion.

ReBac seed3 completed and LGKD seed3 failed at task3 startup; both outcomes are retained in results/20261026/REBAC3_LGKD3_REPORT_ZH.md. LGKD seeds2/3 failures are separate from completed results, without automatic retries.

Class-CIT seed3 is reviewed under frozen source-optimizer-v2; metrics and limitations are in results/20261026/CIT_SEED3_RESULT_ZH.md. Only3/5 seeds complete.
