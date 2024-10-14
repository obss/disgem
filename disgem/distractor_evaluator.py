from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from disgem.util import replace_str


class BaseDistractorEvaluator(ABC):
    """
    Base distractor evaluator class.

    Args:
        model_name_or_path (str): Model name or path.
        **kwargs: Additional keyword arguments for uses in child classes.

    Notes:
        All child classes must define `AUTOMODEL_FACTORY` attribute to
        load the desired model in appropriate architecture. The value
        has to be an `Auto Class`_ of `transformers`.

        .. _Auto Class: https://huggingface.co/docs/transformers/model_doc/auto
    """

    AUTOMODEL_FACTORY: "AutoModel"  # noqa: F821

    def __init__(self, model_name_or_path: str, **kwargs):
        self._model = None
        self._tokenizer = None
        self._load_model_and_tokenizer(model_name_or_path)

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    def _load_model_and_tokenizer(self, model_name_or_path: str):
        """
        Load model and tokenizer from given path using `AUTOMODEL_FACTORY`
        attribute.
        """
        self._model = self.AUTOMODEL_FACTORY.from_pretrained(model_name_or_path)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    @abstractmethod
    def __call__(self, inputs: Dict[str, Any], *args, **kwargs):
        pass


class NLIBasedDistractorEvaluator(BaseDistractorEvaluator):
    """
    NLI based distractor evaluation, meant to be designed to provide classification
    outputs for textual entailment, e.g. ('contradiction', 'neutral', 'entailment').
    Label ids or names can vary among different models, there is no standard
    fine-tuning scheme on the class ids/names.

    Args:
        model_name_or_path (str): Model name or path.
    """

    AUTOMODEL_FACTORY = AutoModelForSequenceClassification

    def __init__(
        self,
        model_name_or_path: str = "geckos/bart-fined-tuned-on-entailment-classification",
    ):
        super(NLIBasedDistractorEvaluator, self).__init__(model_name_or_path)

    def preprocess(
        self,
        sentence: str,
        answer: Dict,
        distractor: Union[str, Dict] = None,
        reverse: bool = False,
        **kwargs,
    ) -> str:
        if distractor is None:
            return sentence
        if isinstance(distractor, dict):
            distractor = distractor["token_str"]
        context_with_distractor = replace_str(sentence, distractor, answer["start"], answer["end"])
        if reverse:
            return context_with_distractor + " " + sentence
        return sentence + " " + context_with_distractor

    def preprocess_distractors(
        self,
        sentence: str,
        answer: Dict,
        distractor1: Union[str, Dict],
        distractor2: Union[str, Dict],
        **kwargs,
    ) -> str:
        if isinstance(distractor1, dict):
            distractor1 = distractor1["token_str"]
        if isinstance(distractor2, dict):
            distractor2 = distractor2["token_str"]

        context_with_d1 = replace_str(sentence, distractor1, answer["start"], answer["end"])
        context_with_d2 = replace_str(sentence, distractor2, answer["start"], answer["end"])

        return context_with_d1 + " " + context_with_d2

    def get_model_output(self, input_text: str):
        model_inputs = self.tokenizer(input_text, return_tensors="pt")
        with torch.no_grad():
            model_outputs = self.model(**model_inputs)
        output_cat_id = model_outputs.logits.softmax(-1).argmax().item()
        return self.model.config.id2label[output_cat_id]

    def __call__(
        self,
        inputs: Dict[str, Any],
        distractor_ids: Tuple[int, int] = None,
        **kwargs,
    ) -> Union[List[str], str]:
        if distractor_ids is None:
            results = []
            for distractor in inputs["distractors"]:
                input_text = self.preprocess(**inputs, distractor=distractor)
                nli_output = self.get_model_output(input_text)
                input_text_rev = self.preprocess(**inputs, distractor=distractor, reverse=True)
                nli_output_rev = self.get_model_output(input_text_rev)
                results.append(nli_output + "-" + nli_output_rev)
            return results
        else:
            distractor1 = inputs["distractors"][distractor_ids[0]]
            distractor2 = inputs["distractors"][distractor_ids[1]]
            input_text = self.preprocess_distractors(**inputs, distractor1=distractor1, distractor2=distractor2)
            nli_output = self.get_model_output(input_text)

            input_text_rev = self.preprocess_distractors(**inputs, distractor1=distractor2, distractor2=distractor1)
            nli_output_rev = self.get_model_output(input_text_rev)
            return f"{nli_output}-{nli_output_rev}"
