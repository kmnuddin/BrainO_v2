# Analysing EEG

Analyses are made of **tools**. A tool is one step (filter a recording, average epochs, run a
cluster test) with a typed set of parameters; see the [tool reference](../reference/tools.md).
The same tools run from Python, from `braino tools run`, from [pipelines](pipelines.md) and,
later, from the AI agent.

## Steps and their files

Tools that work on one recording take the recording's `subject` (and `task`, `session`, `run`
when needed). Each step writes new files to `<dataset>/derivatives/braino/sub-<label>/eeg/`,
named after the recording, and the next step reads them. Nothing is changed in place.

| Step | Tool | Writes |
|---|---|---|
| Filter, notch, resample, re-reference | `eeg.preprocess` | `*_desc-preproc_eeg.fif` (+ `.json`) |
| Remove eye/heart components (decision) | `eeg.ica_clean` | `*_desc-preproc_ica.fif`, then `*_desc-clean_eeg.fif` |
| Mark bad channels (decision) | `eeg.detect_bad_channels` | `*_desc-preproc_channels.tsv` |
| Epoch, reject or repair artifacts | `eeg.epoch` | `*_desc-<label>_epo.fif` |
| Averages, difference waves, amplitudes | `eeg.erp` | `*_desc-<label>_ave.fif`, `*_erp.tsv` |
| Spectra and band power | `eeg.spectral` | `*_psd.npz`, `*_bands.tsv` |
| Time-frequency power (Morlet) | `eeg.tfr` | `*_tfr.npz` |
| Mass-univariate group tests | `stats.mass_univariate` | `group/task-<task>/*_stats.tsv` |
| Cluster permutation tests | `stats.cluster_permutation` | `group/task-<task>/*_clusters.tsv` |

Tools return a short summary (what a person or the agent needs to read) and write the full
result to the files.

## Decisions

`eeg.ica_clean` and `eeg.detect_bad_channels` change which data the results rest on, so they
only **propose**. A person approves (optionally editing the proposal, e.g. a shorter list of
channels) or rejects it:

```bash
braino proposals list --pending --dataset study_bids
braino proposals show <id> --dataset study_bids
braino proposals approve <id> --dataset study_bids --edit '{"channels": ["T8"]}' --note "T8 loose"
braino proposals reject <id> --dataset study_bids --note "keep all channels"
```

The proposal file keeps what was proposed, what was applied, who decided, when and why. The AI
agent will be able to make proposals, never to approve them.

If an earlier step runs again (say, preprocessing with another filter), decisions made on its
old output are refused until they are made again: BrainO checks the hashes of the files each
decision was made on.

## Recommended order

**Preprocess → ICA → bad channels → epoch.** ICA before bad-channel detection works best: eye
activity makes the channels near the eyes look bad, but ICA needs them to remove that activity;
on ERP CORE, marking them bad first left enough blink activity to cancel most of the P3 in some
participants. A channel that is really broken shows up in ICA as a "channel noise" component.
Either order is allowed; bad-channel detection reads the ICA-cleaned recording if there is one.

## What each step does

**Preprocessing** applies, in this order: notch filter (optional, at the line frequency and its
harmonics), band-pass (zero-phase FIR by default, or Butterworth IIR), resampling and
re-referencing (to the average of the good EEG channels, the average of the EEG and EOG
channels, or chosen channels such as linked mastoids). Electrode positions are added from a standard layout if the file has none.

**ICA** is extended infomax (solved with Picard), fitted on an average-referenced copy high-passed
at 1 Hz, as ICLabel expects. ICLabel labels each component (brain, eye, heart, muscle, line
noise, channel noise, other); eye-blink and heart components with a probability of at least 0.8
are proposed for removal. Without ICLabel, components correlating with the EOG channels are.

**Bad channels** are found with PyPREP (deviation, correlation, high-frequency noise, RANSAC) or
BrainO's own quality screen, on an average-referenced copy with blinks left out. Channels used
as the reference cannot be marked bad. PyPREP's correlation criterion counts a channel bad when
it correlates poorly with every other channel in more than 5% of 1-s windows (PREP uses 1%,
which marks most channels of 32-channel caps bad).

**Epoching** cuts epochs around events grouped into named conditions (optionally moved by a
measured stimulus delay, such as a monitor's), corrects the baseline,
interpolates bad channels and handles artifacts with autoreject (per-channel learned
thresholds, repairing or dropping epochs), a global learned threshold, a fixed peak-to-peak
threshold, or not at all.

**ERPs** are condition averages and difference waves, measured as mean amplitude (the most
robust ERP measure) or peak amplitude and latency in a window, over chosen channels.

**Spectra** use Welch's method or multitapers; band power is reported in µV² and relative to the
total. **Time-frequency** power uses Morlet wavelets with cycles growing with frequency,
baseline-corrected in dB by default.

**Group statistics**: one-sample t-tests (a difference wave against 0), paired t-tests (two
conditions) or repeated-measures ANOVA (more), corrected by false discovery rate, Bonferroni
or permutation of the maximum statistic; and cluster-based permutation tests over neighbouring
channels and time or frequency points.
