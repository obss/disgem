# DisGeM: Distractor Generation for Multiple Choice Questions with Span Masking

<a href="https://arxiv.org/abs/2409.18263"><img src="https://img.shields.io/badge/arXiv-2409.18263-b31b1b.svg" alt="Arxiv"></a>
<a href="https://paperswithcode.com/paper/disgem-distractor-generation-for-multiple"><img src="https://img.shields.io/badge/DisGeM-temp?style=square&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB2ZXJzaW9uPSIxLjEiIGlkPSJMYXllcl8xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB4PSIwcHgiIHk9IjBweCIgdmlld0JveD0iMCAwIDUxMiA1MTIiIHN0eWxlPSJlbmFibGUtYmFja2dyb3VuZDpuZXcgMCAwIDUxMiA1MTI7IiB4bWw6c3BhY2U9InByZXNlcnZlIj4gPHN0eWxlIHR5cGU9InRleHQvY3NzIj4gLnN0MHtmaWxsOiMyMUYwRjM7fSA8L3N0eWxlPiA8cGF0aCBjbGFzcz0ic3QwIiBkPSJNODgsMTI4aDQ4djI1Nkg4OFYxMjh6IE0yMzIsMTI4aDQ4djI1NmgtNDhWMTI4eiBNMTYwLDE0NGg0OHYyMjRoLTQ4VjE0NHogTTMwNCwxNDRoNDh2MjI0aC00OFYxNDR6IE0zNzYsMTI4IGg0OHYyNTZoLTQ4VjEyOHoiLz4gPHBhdGggY2xhc3M9InN0MCIgZD0iTTEwNCwxMDRWNTZIMTZ2NDAwaDg4di00OEg2NFYxMDRIMTA0eiBNNDA4LDU2djQ4aDQwdjMwNGgtNDB2NDhoODhWNTZINDA4eiIvPjwvc3ZnPg%3D%3D&label=paperswithcode&labelColor=%23555&color=%2321b3b6&link=https%3A%2F%2Fpaperswithcode.com%2Fpaper%2Fdisgem-distractor-generation-for-multiple" alt="DisGeM"></a>


A Distractor Generation framework utilizing Pre-trained Language Models (PLMs) that are pre-trained with Masked Language Modeling (MLM) objective.

[Paper](https://arxiv.org/abs/2409.18263) 

### Abstract

> Recent advancements in Natural Language Processing (NLP) have impacted numerous sub-fields such as natural language generation, natural language inference, question answering, and more. However, in the field of question generation, the creation of distractors for multiple-choice questions (MCQ) remains a challenging task. In this work, we present a simple, generic framework for distractor generation using readily available Large Language Models (LLMs). Unlike previous methods, our framework relies solely on pre-trained language models and does not require additional training on specific datasets. Building upon previous research, we introduce a two-stage framework consisting of candidate generation and candidate selection. Our proposed distractor generation framework outperforms previous methods without the need for training or fine-tuning. Human evaluations confirm that our approach produces more effective and engaging distractors. The related codebase is publicly available at https://github.com/obss/disgem.

## Installation

Clone the repository.

```bash
git clone https://github.com/obss/disgem.git
cd disgem
```

In the project root, create a virtual environment (preferably using conda) as follows:

```shell
conda env create -f environment.yml
```

## Datasets

Download datasets by the following command. This script will download CLOTH and DGen datasets.

```shell
bash scripts/download_data.sh
```

## Generate Distractors

To see the arguments for generation see `python -m generate --help`.

The following provides an example to generate distractors for CLOTH test-high dataset. You can alter `top-k` and `dispersion` parameters as needed.

```shell
python -m generate data/CLOTH/test/high --data-format cloth --top-k 3 --dispersion 0 --output-path cloth_test_outputs.json
```


## Contributing

Format and check the code style of the codebase as follows.

To check the codestyle,

```bash
python -m scripts.run_code_style check
```

To format the codebase,

```bash
python -m scripts.run_code_style format
```