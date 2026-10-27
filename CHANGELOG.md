# Changelog

All notable changes to BrainO. Versions follow [semantic versioning](https://semver.org); before
1.0, minor versions may change interfaces.

## 0.1.0

The first release: the EEG engine as a Python library and command line.

**Import**
- Readers for EDF/BDF, BrainVision, FIF, EEGLAB, Neuroscan CNT, EGI MFF, GDF, Curry, Nihon
  Kohden and Persyst recordings, all normalised to the same channel names, types, positions and
  events.
- De-identification (headers, dates shifted per subject, event text) and conversion to BIDS,
  checked with the official validator. `braino import`, `braino inspect`, `braino qc`.

**Analysis tools**
- `eeg.preprocess`: filtering, notch, resampling, re-referencing.
- `eeg.ica_clean` (decision): extended-infomax ICA (Picard) with ICLabel or EOG labels.
- `eeg.detect_bad_channels` (decision): PyPREP or BrainO's quality screen, on an
  average-referenced copy with blinks left out.
- `eeg.epoch` with autoreject and stimulus-delay correction; `eeg.erp`, `eeg.spectral`, `eeg.tfr`.
- `stats.mass_univariate` (t-tests, repeated-measures ANOVA; FDR, Bonferroni, max-statistic)
  and `stats.cluster_permutation`.
- Decision tools only propose; a person approves, edits or rejects (`braino proposals`).
  Decisions made on an older version of an earlier step's output are refused.

**Pipelines, provenance and reproducibility**
- Pipeline specs in YAML/JSON, checked against the tools, run with cached results, re-run from
  any step, waiting at decisions (`braino pipeline`).
- Every tool call recorded with arguments, results, file hashes and library versions.
- Script export (`braino runs export`, `braino pipeline export`): a standalone script that
  re-runs the analysis, makes the same decisions and checks every result.

**Validation**
- ERP CORE P3 and N170 data with pinned hashes, the published per-participant scores, and
  `braino validate erpcore` to run the built-in pipelines and compare. Both reproduce the
  published effects on all 40 participants (participant scores r = 0.95 and 0.97).

**Documentation**: <https://kmnuddin.github.io/BrainO_v2/>, with example notebooks in
`examples/`.
