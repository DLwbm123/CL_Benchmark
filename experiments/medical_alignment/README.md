# Matched medical-method alignment audit

The goal is reliable adaptation on the existing benchmark. Author-identical data/protocol is not required; the author U-Net is a diagnostic reference, not a replacement for the shared-backbone result.

The two runners preserve the existing stable supervised Dice+logitsCE, source distillation components, optimizer, validation policy and method-specific inputs. `ARM_BACKBONE=author|shared` is the controlled factor. After construction, `900000+ARM_SEED` resets data/augmentation randomness so architecture initialization does not consume a different prefix of that stream. The stream differs from historical runs; compare the current matched controls, not old numbers as if paired.

`gradient_audit.py` uses `autograd.grad` with retained graphs to measure supervised/distillation parameter-gradient norms and cosine on fixed training steps. It does not update parameters or `.grad`. `smoke_alignment.py` checks this noninterference and all four six-stage synthetic pipelines. Each run saves selected stage models and a triangular development-validation matrix privately.

## Dependencies and invocation

Use the existing pinned, separately obtained TED/CauAug author checkouts. Do not redistribute them. Put these self-written files together in a private run directory with `base_network.py`, `source_components.py`, `ted_components.py`, `shared.py`, and `shared_backbone.py` from `../medical_shared_backbone/`, and `numerics.py` from `../ted_cauaug_repair/`. The existing environment provides PyTorch, NumPy and h5py. Copy the appropriate frozen JSON as `protocol.json`.

Set `WORK_ROOT`, `DATA_ROOT`, `REFERENCE_ROOT`, `ARM_METHOD` (`ted` or `cauaug`), `ARM_BACKBONE`, `ARM_SEED`, `STABILITY_VARIANT=stable_sup`, and `DISTILL_WEIGHT=.001`. Run the corresponding entry through the established private resource/time guard; server-control details are intentionally not published. Reference root contains `TED_DCMIS/` and `CauAug_DCMIS/`. Validation uses `WORK_ROOT` and `METHOD`; the group must have both completed seeds and its private state/log/checkpoint artifacts.

The next-step gate is frozen in `prospective_gate.json`. Train-gradient calibration is only a candidate and requires a fresh paired full validation; partial results cannot trigger tuning. Test data is never accessed. Patient-derived stage models, replay/domain archives and per-domain metrics stay private.
