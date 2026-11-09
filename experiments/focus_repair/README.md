# FOCUS adaptation diagnosis

These are self-authored diagnostic scripts using the sibling medical_shared_backbone modules. Add that directory to PYTHONPATH. Use separately acquired pinned FOCUS source as documented in THIRD_PARTY_NOTICES; this directory distributes no author source.

For diagnose.py, set WORK_ROOT, OLD_ROOT (private completed shared-backbone experiment), REFERENCE_ROOT, DATA_ROOT (six HDF5 files directly inside), ARM_SEED and CUDA_VISIBLE_DEVICES. It loads the old final checkpoint, reproduces original running-BN evaluation, compares running BN batch1 versus only BN using current-slice statistics with CCD disabled, and restores all state. This is not training or checkpoint selection.

For train_probe.py, provision independent timeout and cumulative storage guards before execution. Set WORK_ROOT containing probe_protocol.json copied as protocol.json, plus ARM_VARIANT, ARM_SITE, ARM_SEED and the preceding data/source/GPU variables. Four variants, sites1/6 and seeds41/43 constitute 16 independent first-task arms. Complete all arms before aggregate interpretation. no_ccd/no_gin are diagnostic-only interventions; full_adam preserves FOCUS components with a declared optimizer adaptation. LR is one prescribed value, not a grid search. Diagnostic models are not final continual-learning methods.

Run probe_smoke.py before the training group. The smoke validates CCD bypass identity and matched RNG consumption, and finite forward/backward/optimizer updates. Infrastructure-specific launchers are not distributed. Private checkpoint and per-domain metrics must not be uploaded.
