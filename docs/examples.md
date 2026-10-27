# Examples

Jupyter notebooks in the [`examples/`](https://github.com/kmnuddin/BrainO_v2/tree/main/examples)
folder of the repository, with their outputs. Each downloads the public data it needs.

| Notebook | What it shows |
|---|---|
| [`01_erp_one_participant.ipynb`](https://github.com/kmnuddin/BrainO_v2/blob/main/examples/01_erp_one_participant.ipynb) | One ERP CORE participant step by step: preprocessing, ICA and bad-channel decisions, epochs, the P3, spectra and time-frequency power |
| [`02_pipeline_group_and_export.ipynb`](https://github.com/kmnuddin/BrainO_v2/blob/main/examples/02_pipeline_group_and_export.ipynb) | The built-in pipeline over several participants, group statistics, comparison with the published scores, and an exported script that reproduces it all |

To run them:

```bash
pip install "braino[eeg]" jupyter
jupyter lab examples/
```
