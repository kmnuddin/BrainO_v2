# Validation

BrainO's analysis pipeline is checked against **ERP CORE** (Kappenman et al., 2021,
*NeuroImage* 225:117465; CC-BY-4.0): 40 young adults recorded with a 30-channel BioSemi system
in six tasks designed to elicit standard ERP components. The authors published each
participant's score, so BrainO can be compared participant by participant, not only on the
group mean.

```bash
braino validate erpcore P3          # all 40 participants: 2.2 GB download
braino validate erpcore N170        # 3.4 GB
braino validate erpcore P3 --subjects 001,002,003
```

The command downloads the data (each file checked against a pinned SHA-256 hash), runs the
built-in pipeline (`braino pipeline show erpcore-p3`) with its decisions approved as proposed,
excludes participants with more than 25% of trials rejected (ERP CORE's rule), and compares.

## What counts as reproduced

1. **Group level:** the 95% confidence interval of BrainO's mean includes the published mean,
   and the effect is significant in the published direction (p < .001).
2. **Participant level:** BrainO's scores correlate with the published ones at r ≥ 0.9 over the
   participants both kept.

Which participants each side excluded is reported but is not a pass/fail check: the authors also
removed some artifacts by hand.

## Results

All 40 participants, default settings of the built-in pipelines (2026-10-25):

| Component | Measure | Published | BrainO | Participant level | Result |
|---|---|---|---|---|---|
| P3 | rare − frequent letters, Pz, 300–600 ms | +6.29 µV (n = 34) | +7.04 µV [95% CI +5.36, +8.71] (n = 34) | r = 0.949 over 30, mean abs. difference 0.77 µV | **reproduced** |
| N170 | faces − cars, PO8, 110–150 ms | −3.37 µV (n = 37) | −3.68 µV [95% CI −4.61, −2.75] (n = 40) | r = 0.967 over 37, mean abs. difference 0.63 µV | **reproduced** |

Exclusions (more than 25% of epochs rejected): for the P3, BrainO excluded 003, 006, 009,
021, 026 and 037, the authors 006, 009, 010, 030, 035 and 040; for the N170, BrainO excluded
nobody, the authors 001, 005 and 016.

## What BrainO does differently

The filter (0.1–30 Hz), sampling rate (256 Hz), reference (P9/P10 for the P3; the average of all
EEG and EOG channels for the N170), the 26 ms monitor delay the authors measured with a
photosensor, epochs (−200 to 800 ms, baseline −200 to 0 ms), electrode and window follow the
authors. Artifact handling is BrainO's own, with no hand editing: ICA with ICLabel instead of the
authors' ICA with components chosen by eye; PyPREP bad-channel detection; autoreject instead of
the authors' moving-window peak-to-peak and step-function rejection; and trials are not limited
to correct responses.

Getting there taught three things:

- **Run ICA before marking bad channels.** PyPREP marks the channels near the eyes bad for eye
  activity, ICA then cannot remove the blinks, and what remains can cancel the effect
  (participant 007's P3 fell from 6.8 to 0.2 µV). This is now BrainO's recommended order.
- **PyPREP's correlation criterion is too strict for 30 channels** at PREP's 1% of the time; 5%
  is BrainO's default.
- **Timing matters for early components.** Without the 26 ms monitor correction the N170 came
  out at −1.73 µV with r = 0.64; with it, −3.68 µV and r = 0.97. `eeg.epoch` takes the delay
  as `event_delay_s`.
