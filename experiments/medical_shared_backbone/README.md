# Matched ResUNet32 development validation

The recorded protocol uses six domains, seeds 41/43 and 50 epochs/domain. The backbone has 2,915,808 parameters. This directory is an explicit adaptation, not author-native reproduction. FOCUS completed; other method results were pending when this report was written.

Use a CUDA PyTorch environment with numpy/h5py and the dependencies of the separately acquired pinned upstream sources listed in THIRD_PARTY_NOTICES.md. Set WORK_ROOT to a private writable output directory containing protocol.json, DATA_ROOT to the existing six HDF5 files, and REFERENCE_ROOT to a directory containing FOCUS, CauAug_DCMIS and TED_DCMIS. No source repositories, data or checkpoints are distributed here. The TED loader in ted_components.py reads the pinned repository files in their original relative paths.

Run smoke.py first. Set ARM_METHOD and ARM_SEED for a protocol-listed pair; use runner.py for FOCUS/CauAug and ted_runner.py for TED. Provision independent wall-time and aggregate storage guards before training. The infrastructure-specific supervisor is intentionally not distributed. Run validate.py with METHOD only after both seeds of that method finish; it expects the private supervisor's state.json with arms[key].returncode. It does not certify unfinished methods.

Model/optimizer/teacher/replay/archive checkpoints and per-domain metrics must remain private. Validation uses the first metadata block for selection and the remaining blocks for development evaluation. Blocks are not verified patients. No test split is accessed. Standard benchmark 150-epoch/batch8 results are not directly interchangeable with this 50-epoch/batch4 experiment.
