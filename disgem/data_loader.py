from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Union

from disgem.util import read_json


@dataclass
class InstanceCollection:
    context: str
    answers: List[Dict[str, Union[str, int]]]
    distractors_collection: List[List[str]] = None


@dataclass
class Instance:
    context: str
    answer: Dict[str, Union[str, int]]
    distractors: List[str] = None


class DataLoader:
    def __init__(self, filepath: str):
        self.dataset = self.read(Path(filepath))

    @abstractmethod
    def read(self, filepath):
        pass

    def __len__(self):
        return sum([len(inst.answers) for inst in self.dataset])

    def __iter__(self) -> Instance:
        for instance in self.dataset:
            for i, ans in enumerate(instance.answers):
                if instance.distractors_collection is not None:
                    yield Instance(
                            context=instance.context,
                            answer=ans,
                            distractors=instance.distractors_collection[i]
                    )
                else:
                    yield Instance(
                            context=instance.context,
                            answer=ans
                    )


class SquadLoader(DataLoader):
    """
    A Data loader designed to load instances from SQuAD style datasets in
    a form compatible for distractor generation.
    See the home page for SQuAD: https://rajpurkar.github.io/SQuAD-explorer/
    """

    def read(self, filepath):
        instances = []
        raw_data = read_json(filepath)

        for article in raw_data["data"]:
            for paragraph in article["paragraphs"]:
                answers = []
                for qa in paragraph["qas"]:
                    answer = qa["answers"][0]
                    answer["start"] = answer.pop("answer_start")
                    answer["end"] = answer["start"] + len(answer["text"])
                    answers.append(answer)
                instances.append(InstanceCollection(context=paragraph["context"], answers=answers))
        return instances


class ClothLoader(DataLoader):
    """
    A Data loader designed to load instances from CLOTH style datasets in
    a form compatible for distractor generation.
    See the home page for CLOTH: https://www.cs.cmu.edu/~glai1/data/cloth/
    """
    _dataset_mask_str = " _ "
    _pipeline_mask_str = "<mask>"

    def __iter__(self) -> Instance:
        """
        The iter method is overridden due to the fact that there are multiple gaps in a single context.
        Unlike this method CDGP uses isolated contexts/sentences to generate distractors from
        passages with a single token only, we incorporate the whole context available.
        """
        for instance in self.dataset:
            for i, ans in enumerate(instance.answers):
                yield Instance(
                        context=self.replace_nth(instance.context, self._pipeline_mask_str, ans["text"], i+1),
                        answer=ans,
                        distractors=instance.distractors_collection[i]
                )

    @staticmethod
    def replace_nth(text: str, substr: str, replace: str, nth: int):
        """
        Replace nth occurance of a substring.
        Taken from https://stackoverflow.com/a/66338058
        """
        arr = text.split(substr)
        part1 = substr.join(arr[:nth])
        part2 = substr.join(arr[nth:])

        return part1 + replace + part2

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
        files = sorted(filepath.glob('*.json'))
        for p in files:
            data = read_json(p)
            ctx = data["article"]
            answers = []
            distractors = []
            for choices, answer in zip(data["options"], data["answers"]):
                start = ctx.find(self._dataset_mask_str)
                ans, opt = self.get_option(choices, answer)
                ctx = ctx.replace(self._dataset_mask_str, self._pipeline_mask_str, 1)  # only replace 1 occurance
                answer = {"text": ans, "start": start}
                answer["end"] = answer["start"] + len(answer["text"])
                answers.append(answer)
                distractors.append(choices[:opt] + choices[opt+1:])  # remove the answer from choices
            assert len(answers) == len(distractors), "The length of the `answers` and `distractors` must be equal."
            instances.append(InstanceCollection(context=ctx, answers=answers, distractors_collection=distractors))
        return instances


class CdgpClothLoader(DataLoader):
    """
    A Data loader designed to load instances from modified CLOTH style datasets
    in a form compatible for distractor generation. We refer to this style as CDGP
    as it is used and published in a related work.
    See the home page for CDGP style CLOTH: https://huggingface.co/datasets/AndyChiang/cloth
    """
    _dataset_mask_str = " [MASK] "

    def read(self, filepath):
        instances = []
        data = read_json(filepath)
        for instance in data:
            ctx = instance["sentence"]
            ans = instance["answer"]
            start = ctx.find(self._dataset_mask_str)
            ctx = ctx.replace(self._dataset_mask_str, ans, 1)
            answers = [{"text": ans, "start": start, "end": start + len(ans)}]
            instances.append(InstanceCollection(context=ctx, answers=answers, distractors_collection=[instance["distractors"]]))
        return instances


class DGenLoader(CdgpClothLoader):
    """
    A Data loader designed to load instances from DGen dataset
    in a form compatible for distractor generation.
    See the home page for DGEN Dataset: AndyChiang/dgen
    """
    _dataset_mask_str = "**blank**"
