# disgem
Distractor Generation for Multiple Choice Question


## Installation

```shell
conda env create -f environment.yml
```

## Generate Distractors

Download datasets by the following command. This script will download CLOTH and DGen datasets.

```shell
bash scripts/download_data.sh
```

To generate distractors for CLOTH test-high dataset, run the following command. You can alter `top-k` and `dispersion` parameters.

```shell
python -m generate data/CLOTH/test/high --data-format cloth --top-k 3 --dispersion 0 --output-path cloth_test_outputs.json
```

To see the arguments for generation see `python -m generate --help`.
