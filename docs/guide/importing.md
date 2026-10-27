# Importing recordings

`braino import` turns recordings in any supported format into a BIDS dataset. The pipeline is
the same for every format: **detect → read → normalise → de-identify → write BIDS**.

```bash
# One recording. The key file links subject labels to real IDs: keep it private, outside the
# dataset (BrainO refuses to put it inside).
braino import raw/patient_a.edf --subject-id "patient A" --task rest \
  --dataset study_bids --key private/key.json

# Many, from a CSV with columns source,subject_id,task[,session,run,line_freq]
braino import --mapping recordings.csv --dataset study_bids --key private/key.json
```

## Formats

EDF/EDF+, BDF, BrainVision, FIF, EEGLAB, Neuroscan CNT, EGI MFF, GDF, Curry, Nihon Kohden and
Persyst, all read through MNE-Python. XDF (Lab Streaming Layer) is planned.

## Normalisation

Every format names channels and stores events differently, so import makes them alike and
reports every change:

- standard electrode names get their standard spelling (`FP1` → `Fp1`, `EEG Fz-REF` → `Fz`);
- channels typed EEG whose names say otherwise become EOG, ECG, EMG or miscellaneous;
- electrode positions come from the file, or from a standard layout when the names identify
  one (and the BIDS files then say they are approximate);
- events in trigger channels become annotations.

## De-identification

Names, birth dates and recording dates are removed from file headers; dates are shifted per
subject into 1902-1924, keeping age, time of day and intervals; identifiers, dates, e-mail
addresses and phone numbers are removed from event text, and any other free text is listed for
you to check. Data are always written new, never copied, because an original header still holds
the personal data. The provenance record keeps only hashes of the original files, not their
names.

## Checking the result

```bash
braino inspect --dataset study_bids    # subjects, tasks, channels and events per recording
braino qc --dataset study_bids         # flat and noisy channels, line noise, artifacts
```

The BIDS files pass the official validator (tested in CI with only "recommended metadata"
warnings).

!!! tip
    Datasets that are already in BIDS (like ERP CORE) need no import: point the analysis tools
    at them directly.
