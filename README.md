# disgem
Distractor Generation for Multiple Choice Question


## Installation

```shell
conda env create -f environment.yml
```

To generate distractors prepare a JSON file and a SQuAD style data.

```shell
python -m generate <path/to/file>
```

To see the arguments for generation see `python -m generate --help`.
