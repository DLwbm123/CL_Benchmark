# TED and CauAug numerical audit / training extension

Self-authored scripts; add the sibling medical_shared_backbone directory to PYTHONPATH for its data loader, fixed backbone and pinned upstream-definition loaders. Author repositories are not distributed. Refer to THIRD_PARTY_NOTICES for source revisions and permissions.

Set WORK_ROOT, REFERENCE_ROOT, DATA_ROOT (directory directly containing the six HDF5 files), ARM_SEED, ARM_METHOD, CUDA_VISIBLE_DEVICES. Diagnose additionally needs OLD_ROOT pointing at the private original shared-backbone results. The fixed-checkpoint diagnostic reads only training samples, does not update parameters, and restores BN buffers; it is not a validation comparison.

For TED, run smoke.py first. Use ted_stable_runner.py with the matching protocol copied as protocol.json and STABILITY_VARIANT=stable_sup or stable_all. stable_sup changes supervised two-class DiceBCE to stable Dice+logitsCE; stable_all also computes image/SKA KD with log_softmax while retaining reduction axes and original AKT/GUGF. Both retain teacher train-BN and restore evaluation state. Moderate-logit value/gradient equivalence and extreme-logit gradient recovery are checked in numerics.py.

CauAug150 uses the unchanged sibling medical_shared_backbone/runner.py and cauaug150_protocol.json, ARM_METHOD=cauaug. No numerical-loss change is applied without evidence; only the planned epoch count is extended.

Provision independent hard timeout and per-run/window storage guards before execution. After complete paired seeds, run validate.py with METHOD=ted or cauaug. The private supervisor state.json, logs/results and rolling checkpoints are required. Keep all weights, features, archives and per-domain values private. These are development validation experiments; no final-test claims.

Completed long-run and numeric-diagnostic update: [report](../../results/TED_CAUAUG_LONGRUN_UPDATE_ZH.md). `ted_sup150_protocol.json` freezes the sufficient supervised-only TED repair at 150 epochs; `cauaug_sup150_protocol.json` and `cauaug_stable_runner.py` freeze the evidence-driven CauAug supervised-only repair. `cauaug_smoke.py` exercises the six-stage synthetic pipeline; all source components remain external.
