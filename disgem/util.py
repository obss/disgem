import json
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
from transformers.utils import ModelOutput


def replace_str(s: str, new: str, start_index: int, end_index: int) -> str:
    if start_index not in range(len(s)) or end_index not in range(len(s) + 1):
        raise ValueError("index outside given string")
    return s[:start_index] + new + s[end_index:]


def geometric_mean(values: List) -> Union[int, float]:
    return np.prod(values) ** (1 / len(values))


def harmonic_mean(values: List) -> Union[int, float]:
    return len(values) / np.sum(1 / np.array(values))


def read_json(filepath: str):
    with open(filepath, "r") as fd_in:
        return json.load(fd_in)


@dataclass
class DistractorGenerationOutput(ModelOutput):
    """
    Data class for distractor generation pipeline outputs.

    Args:
        distractors (`list(dict)`):
            Generated distractors as list of dictionary.
        discarded_distractors (`list(dict))`:
            Distractors that are eliminated within the postprocess due to entailment.
    """

    distractors: List[Dict] = None
    discarded_distractors: List[Dict] = None
