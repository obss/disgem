import json
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from transformers.utils import ModelOutput


def replace_str(s: str, new: str, start_index: int, end_index: int) -> str:
    if start_index not in range(len(s)) or end_index not in range(len(s)):
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


@dataclass
class InstanceStruct:
    context: str
    answers: List[Dict[str, Union[str, int]]]
    gt_distractors: List[List[str]] = None


class DataLoader:
    def __init__(self, filepath: str):
        self.dataset = self.read(Path(filepath))

    @abstractmethod
    def read(self, filepath):
        pass

    def __len__(self):
        return sum([len(inst.answers) for inst in self.dataset])

    def __iter__(self) -> InstanceStruct:
        for instance in self.dataset:
            yield instance


class SquadLoader(DataLoader):
    def read(self, filepath):
        instances = []
        raw_data = read_json(filepath)

        for article in raw_data:
            for paragraph in article["paragraphs"]:
                answers = []
                for qa in paragraph["qas"]:
                    answer = qa["answers"][0]
                    answer["start"] = answer.pop("answer_start")
                    answer["end"] = answer["start"] + len(answer["text"])
                    answers.append(answer)
                instances.append(InstanceStruct(context=paragraph["context"], answers=answers))
        return instances


class ClothLoader(DataLoader):
    def get_option(self, choices: List[str], choice: str):
        if choice not in list("ABCD"):
            raise ValueError(f"`choice` must be one of A,B,C,D; got {choice}.")
        if choice == "A":
            return choices[0], 0
        if choice == "B":
            return choices[1], 1
        if choice == "C":
            return choices[2], 2
        return choices[3], 3  # Option D

    def read(self, filepath):
        assert filepath.is_dir(), "`filepath` for CLOTH dataset needs to be a directory."

        instances = []
        for p in filepath.glob("*.json"):
            data = read_json(p)
            ctx = data["article"]
            answers = []
            distractors = []
            for choices, answer in zip(data["options"], data["answers"]):
                start = ctx.find(" _ ")
                ans, opt = self.get_option(choices, answer)
                ctx = ctx.replace(" _ ", ans, 1)  # only replace 1 occurance
                answer = {"text": ans, "start": start}
                answer["end"] = answer["start"] + len(answer["text"])
                answers.append(answer)
                distractors.append(choices[:opt] + choices[opt+1:])  # remove the answer from choices
            instances.append(InstanceStruct(context=ctx, answers=answers, gt_distractors=distractors))
        return instances
