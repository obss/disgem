from typing import Dict, List, Union

from transformers import AutoModelForMaskedLM, AutoTokenizer

from disgem.pipeline import DistractorGenerationPipeline
from disgem.util import DistractorGenerationOutput


class MaskedLMBasedDistractorGenerator:
    """
    Generates distractors from given sentence and answer pair by masking.

    Args:
        pretrained_model_name_or_path: Pretrained model name or path.
        **kwargs: Additional keyword arguments to be passed on pipeline constructor.

    Examples:
        >>> distractor_generator = MaskedLMBasedDistractorGenerator(pretrained_model_name_or_path="bert-large-cased")
        >>> context = "My sentence which has the answer is right here."
        >>> answer = {"text": "right here", "start": 36, "end": 46}
        >>> print(distractor_generator(context=context, answer=answer, top_k=5))
        [
            {'text': 'this sentence', 'score': 0.007555645013882173},
            {'text': ': Yes', 'score': 0.0072860184115599025},
            {'text': 'no more', 'score': 0.007278744063432152},
            {'text': 'complete nonsense', 'score': 0.006642898213074899},
            {'text': '..', 'score': 0.004911088230391353}
        ]
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "bert-large-cased",
        **kwargs,
    ):
        model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self._pipeline = DistractorGenerationPipeline(model, tokenizer, **kwargs)

    def __call__(
        self,
        context: str,
        answer: Dict[str, Union[int, str]],
        minify_output: bool = True,
        only_strings: bool = True,
        **kwargs,
    ) -> Union[List[Dict[str, Union[str, float]]], DistractorGenerationOutput]:
        instance = {"context": context, "answer": answer}
        outputs = self._pipeline(instance, **kwargs)
        if minify_output:
            if not only_strings:
                return [
                    {
                        "text": distractor["token_str"],
                        "score": distractor["ranking_score"],
                    }
                    for distractor in outputs.distractors
                ]
            else:
                return [distractor["token_str"] for distractor in outputs.distractors]
        return outputs
