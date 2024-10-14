from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import spacy
from transformers import FillMaskPipeline, ModelCard, PreTrainedTokenizer, add_end_docstrings
from transformers.feature_extraction_utils import PreTrainedFeatureExtractor
from transformers.pipelines.base import PIPELINE_INIT_ARGS, ArgumentHandler, GenericTensor, PipelineException
from transformers.utils import logging

from disgem.distractor_evaluator import NLIBasedDistractorEvaluator
from disgem.util import DistractorGenerationOutput, geometric_mean, harmonic_mean, replace_str

logger = logging.get_logger(__name__)


@add_end_docstrings(
    PIPELINE_INIT_ARGS,
    r"""
        single_mask (`bool`, *optional*): (default=False)
            If True regardless of the number of tokens of the answer, the masked_input will always have a single mask 
            token. Otherwise, the token size will be the same as the number of tokens of the answer. It is the same as 
            passing `n_mask=1` & `dispersion=0`.
        n_mask (`int`, *optional*): (default=None)
            Determines the masked token count for the model input. If `None`, the mask token count is determined by the 
            token count of the answer span.
        dispersion (`int`, *optional*): (default=1)
            Sets the interval for random sample of token sizes. If greater than 0, then the interval is set as 
            `(n_mask-dispersion, n_mask+dispersion)` inclusive and token sizes are randomly chosen from this interval. 
            Since this is currently experimental, the sample size is set to min(3, interval_size) as hardcoded, it may 
            be subject to change in the future versions. 
            Note that if `dispersion=0` it results in faster computation as min(3, interval_size) will always yield 1 
            decreasing the number of generation computations. Higher dispersion values results in higher diversity in 
            the resulting generations although dispersion values that are too high generally decreases the quality of 
            the outputs.
        use_harmonic_mean (`bool`, *optional*): (default=True)
            Whether to use harmonic mean or not for ranking generations with different mask count. Usually, harmonic 
            mean tends to be fair among different token counts results in higher diversity.
        strategy (`str`, *optional*): (default="l2r")
            Determining the unmasking strategy in case of multiple masks. Options are:
             ("l2r", "r2l", "ctl").
                - "l2r": Generation is done in a left-to-right fashion where the mask 
                tokens are predicted one-by-one from left-to-right determining the prediction of next mask tokens. 
                Input has multiple masks in each iteration (masks are generated in a consecutive order). That is, 
                if we have five masks, the unmasking order is 1,2,3,4,5.
                - "r2l": Generation is done in a right-to-left fashion, that this implementation proceeds 
                in a reverse order of `l2r` (from right-to-left, i.e first generate the 
                lastest mask token). That is, if we have five masks, the unmasking order is 5,4,3,2,1.
                - "cocktail_shaker": This generation strategy is a mix of `l2r` and `r2l` decoding strategies 
                The name comes from the cocktail shaker sort. It procedurally generates the first mask token and 
                the last mask token respectively between steps. That is, if we have five masks, the unmasking order is 
                1,5,2,4,3.
    """,
)
class DistractorGenerationPipeline(FillMaskPipeline):
    """
    Distractor generation pipeline utilizing mask-filling/unmasking using any `ModelWithLMHead`.

    The models that this pipeline can use are models that have been trained with a masked language modeling objective,
    which includes the bi-directional models in the library. See the up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=fill-mask).

    This pipeline extends HF "fill-mask" pipeline which uses joint probabilities and recursive generations in case of
    multiple masks. However, note that this implementation is probably suitable to contiguous multiple masks. This
    pipeline provides joint probabilities in contrast to the original `FillMaskPipeline`. See the notes for the
    `FillMaskPipeline` output and discussion.

    Note:
        `FillMaskPipeline` works for single mask inputs and experimentally supports multiple masks. However, in
        the latter scenario disjoint probabilities are returned that may be limiting desired applications.
        For further read, see https://github.com/huggingface/transformers/pull/10222
    """

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],  # noqa: F821
        tokenizer: Optional[PreTrainedTokenizer] = None,
        feature_extractor: Optional[PreTrainedFeatureExtractor] = None,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        task: str = "",
        args_parser: ArgumentHandler = None,
        device: int = -1,
        binary_output: bool = False,
        **kwargs,
    ):
        self._decoding: str = "l2r"
        self._use_harmonic_mean: bool = True
        self._search_multiplier: int = 4
        super(DistractorGenerationPipeline, self).__init__(
            model,
            tokenizer,
            feature_extractor,
            modelcard,
            framework,
            task,
            args_parser,
            device,
            binary_output,
            **kwargs,
        )
        self.evaluator = NLIBasedDistractorEvaluator()
        self.spacy = spacy.load("en_core_web_sm")

    def _mask_answer(self, context: str, answer: Dict, n_mask: int):
        mask_str = " ".join([self.tokenizer.mask_token] * n_mask)
        return replace_str(context, mask_str, start_index=answer["start"], end_index=answer["end"])

    def _sanitize_parameters(
        self,
        top_k: int = 3,
        targets=None,
        decoding: str = None,
        single_mask: bool = None,
        dispersion: int = None,
        n_mask: int = None,
        use_harmonic_mean: bool = None,
        seed: int = 42,
    ):
        preprocess_params = {}
        forward_params = {}
        postprocess_params = {}

        if decoding is not None:
            self._decoding = decoding

        if use_harmonic_mean is not None:
            self._use_harmonic_mean = use_harmonic_mean

        if single_mask is not None:
            preprocess_params["single_mask"] = single_mask

        elif single_mask is None and n_mask is not None:
            # If 'single_mask' is set as True, n_mask is ignored
            if n_mask < 1:
                raise ValueError("'n_mask' must be at least 1.")
            preprocess_params["n_mask"] = n_mask

        if dispersion is not None:
            if dispersion < 0:
                raise ValueError("'dispersion' cannot be negative.")
            preprocess_params["dispersion"] = dispersion

        # Put seed in forward_params which will be later popped by run_single()
        forward_params["seed"] = seed

        if targets is not None:
            target_ids = self.get_target_ids(targets, top_k)
            postprocess_params["target_ids"] = target_ids

        if top_k is not None:
            if top_k < 1:
                raise ValueError("'top_k' must be at least 1.")
            postprocess_params["top_k"] = top_k

        if single_mask or (dispersion == 0 and n_mask == 1):
            # Raising the search multiplier to not shrink search space too low.
            self._search_multiplier = 8

        if self.tokenizer.mask_token_id is None:
            raise PipelineException(
                "fill-mask",
                self.model.base_model_prefix,
                "The tokenizer does not define a `mask_token`.",
            )

        return preprocess_params, forward_params, postprocess_params

    def get_masked_index(self, input_ids: GenericTensor, as_tuple=False) -> Union[List[Tuple[int, int]], np.ndarray]:
        masked_index = super().get_masked_index(input_ids)
        if as_tuple:
            # noinspection PyTypeChecker
            return [tuple(t) for t in masked_index.tolist()]
        return masked_index

    def _get_lr_dispersion(self, n_tokens: int, dispersion: int) -> Tuple[int, int]:
        l_dispersion = max(n_tokens - dispersion, 1)
        r_dispersion = n_tokens + dispersion
        return l_dispersion, r_dispersion

    def preprocess(
        self,
        inputs,
        return_tensors=None,
        single_mask=False,
        dispersion: int = 1,
        n_mask: int = None,
    ) -> GenericTensor:
        if return_tensors is None:
            return_tensors = self.framework or "pt"
        if single_mask:
            dispersion = 0
            n_mask = 1
        if n_mask is not None:
            n_tokens = n_mask
        else:
            n_tokens = len(self.tokenizer(inputs["answer"]["text"])["input_ids"][1:-1])
        l_disp, r_disp = self._get_lr_dispersion(n_tokens, dispersion)
        masked_inputs = [
            self._mask_answer(**inputs, n_mask=n_mask)
            for n_mask in np.random.choice(
                np.arange(l_disp, r_disp + 1),
                min(3, r_disp - l_disp + 1),
                replace=False,
            )
        ]
        model_inputs = [self.tokenizer(masked_input, return_tensors=return_tensors) for masked_input in masked_inputs]
        for model_input in model_inputs:
            self.ensure_exactly_one_mask_token(model_input)
        return model_inputs

    def postprocess_all_outputs(self, all_outputs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        return sorted(all_outputs, key=lambda d: d["ranking_score"], reverse=True)

    @staticmethod
    def find_span_start_end(span_start: int, span_end: int, sentence_start: int):
        span_start = span_start - sentence_start
        span_end = span_end - sentence_start
        return span_start, span_end

    def split_sentences(self, context: str):
        return self.spacy(context).sents

    @staticmethod
    def _fix_cocktail_shaker_list(tokens) -> None:
        """Fixes the order of input list `tokens` for `cocktail_shaker` decoding inplace."""
        mid_idx = len(tokens) // 2
        temp_tokens = tokens[:mid_idx]
        del tokens[:mid_idx]
        tokens.extend(temp_tokens)

    def _preprocess_input_for_distractor_evaluation(
        self,
        inputs: Dict[str, Any],
        outputs: List[Dict[str, Any]],
    ):
        # Prepare sentence & update answer start & end accordingly.
        sentences = self.split_sentences(inputs["context"])
        updated_answer = dict(text=inputs["answer"]["text"])
        for candid in sentences:
            if inputs["answer"]["start"] >= candid.start_char and inputs["answer"]["end"] <= candid.end_char:
                start, end = self.find_span_start_end(
                    span_start=inputs["answer"]["start"],
                    span_end=inputs["answer"]["end"],
                    sentence_start=candid.start_char,
                )
                updated_answer["start"] = start
                updated_answer["end"] = end
                # update outputs
                break
        return dict(
            answer=updated_answer,
            distractors=outputs,
            sentence=candid.text,
        )

    def _evaluate_answer_distractors(
        self, inputs: Dict[str, Any], outputs: List[Dict[str, Any]], top_k: int = 3
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        processed_input = self._preprocess_input_for_distractor_evaluation(inputs, outputs)
        # Compare distractors with the answer
        answer_distractor_evaluation_results = self.evaluator(processed_input)
        check = lambda x: "contradiction" in x[0] or "neutral" in x[0]
        mask = np.apply_along_axis(check, arr=np.array(answer_distractor_evaluation_results).reshape(-1, 1), axis=-1)
        outputs_array = np.array(outputs, dtype=dict)
        filtered_distractors = outputs_array[mask]

        # Compare distractors within
        processed_input["distractors"] = filtered_distractors.tolist()
        discarded_distractors = []
        increment = True
        kept_index = 1
        while kept_index < top_k < len(processed_input["distractors"]):
            processed_input["distractors"][kept_index]["nli_output"] = []
            for j in range(kept_index):
                nli_out = self.evaluator(processed_input, distractor_ids=(j, kept_index))
                processed_input["distractors"][kept_index]["nli_output"].append(
                    {
                        "result": nli_out,
                        "with": processed_input["distractors"][j]["token_str"],
                    }
                )
                if nli_out == "entailment-entailment":
                    increment = False
                    discarded_distractors.append(processed_input["distractors"].pop(kept_index))
                    break
                increment = True

            if increment:
                # Since we remove elements inplace, only increment when it's stated.
                kept_index += 1

        return processed_input["distractors"][:top_k], discarded_distractors

    def _finalize_generation_outputs(
        self, outputs: List[Dict[str, Any]], cocktail_shaker: bool
    ) -> List[Dict[str, Any]]:
        for i, output in enumerate(outputs):
            if cocktail_shaker:
                self._fix_cocktail_shaker_list(output["token_list"])
                self._fix_cocktail_shaker_list(output["token_str"])
            outputs[i]["token_str_list"] = output["token_str"]
            outputs[i]["token_str"] = self.tokenizer.decode(output["token_list"]).strip()
            outputs[i]["score"] = np.prod(output["score_list"])
            if self._use_harmonic_mean:
                outputs[i]["ranking_score"] = harmonic_mean(output["score_list"])
            else:
                outputs[i]["ranking_score"] = geometric_mean(output["score_list"])

        return sorted(outputs, key=lambda d: d["score"], reverse=True)

    def _generate_distractors(
        self,
        model_inputs,
        forward_params,
        postprocess_params,
        prev_outputs=None,
        reverse: bool = False,
        cocktail_shaker: bool = False,
    ):
        idx = -1 if reverse else 0
        try:
            masked_indices = self.get_masked_index(model_inputs[0]["input_ids"], as_tuple=True)
            current_mask_index = masked_indices[idx]
        except (KeyError, IndexError):  # End of the generation
            return self._finalize_generation_outputs(
                prev_outputs,
                cocktail_shaker=cocktail_shaker,
            )

        is_start = False
        is_last = False
        if prev_outputs is None:
            is_start = True
        if len(masked_indices) == 1:
            is_last = True

        model_outputs = [self.forward(model_input, **forward_params) for model_input in model_inputs]
        postprocess_params_ = deepcopy(postprocess_params)
        if is_start:
            postprocess_params_["top_k"] *= self._search_multiplier
        else:
            postprocess_params_["top_k"] = 1
        outputs = [self.postprocess(model_output, **postprocess_params_) for model_output in model_outputs]
        if is_start and is_last:
            outputs = [outputs]
        new_model_inputs = []
        prev_outputs_ = []
        for i, output in enumerate(outputs):
            if is_start:
                output_at_idx = [output[idx]] if isinstance(output[idx], dict) else output[idx]
                for out in output_at_idx:
                    model_inputs_ = deepcopy(model_inputs[0])
                    model_inputs_["input_ids"][current_mask_index] = out["token"]
                    new_model_inputs.append(model_inputs_)
                    prev_outputs_.append(out)
            else:
                model_inputs_ = deepcopy(model_inputs[i])
                if is_last:
                    model_inputs_ = output[0]
                    prev_outputs_.append(output[0])
                else:
                    model_inputs_["input_ids"][current_mask_index] = output[idx][0]["token"]
                    prev_outputs_.append(output[idx][0])
                new_model_inputs.append(model_inputs_)

        if prev_outputs is None:
            prev_outputs = deepcopy(prev_outputs_)
            for i, _ in enumerate(prev_outputs_):
                prev_outputs[i]["score_list"] = [prev_outputs_[i]["score"]]
                prev_outputs[i]["token_list"] = [prev_outputs_[i]["token"]]
                prev_outputs[i]["token_str"] = [prev_outputs_[i]["token_str"]]
        else:
            for i, prev_output in enumerate(prev_outputs_):
                insert_idx = 0 if reverse else len(prev_outputs[i]["token_list"])
                prev_outputs[i]["score_list"].insert(insert_idx, prev_output["score"])
                prev_outputs[i]["token_list"].insert(insert_idx, prev_output["token"])
                prev_outputs[i]["token_str"].insert(insert_idx, prev_output["token_str"])
                prev_outputs[i]["sequence"] = prev_output["sequence"]

        if cocktail_shaker:
            reverse = not reverse
        return self._generate_distractors(
            new_model_inputs,
            forward_params,
            postprocess_params,
            prev_outputs=prev_outputs,
            reverse=reverse,
            cocktail_shaker=cocktail_shaker,
        )

    def generate_distractors(
        self,
        model_inputs,
        forward_params,
        postprocess_params,
        reverse: bool = False,
        cocktail_shaker: bool = False,
    ):
        """
        Generating distractors in a l2r fashion. Generated tokens in each
        step determines the candidates in the next step. The generation is repeatedly
        performed until there is no mask tokens left.
        """
        all_outputs = []
        for model_input in model_inputs:
            outputs = self._generate_distractors(
                [model_input],
                forward_params,
                postprocess_params,
                reverse=reverse,
                cocktail_shaker=cocktail_shaker,
            )
            all_outputs.extend(outputs)
        all_outputs = self.postprocess_all_outputs(all_outputs, **postprocess_params)
        return all_outputs

    def generate(self, model_inputs, forward_params, postprocess_params):
        if self._decoding == "l2r":
            return self.generate_distractors(model_inputs, forward_params, postprocess_params)
        elif self._decoding == "r2l":
            return self.generate_distractors(model_inputs, forward_params, postprocess_params, reverse=True)
        elif self._decoding == "ctl":
            return self.generate_distractors(
                model_inputs,
                forward_params,
                postprocess_params,
                cocktail_shaker=True,
            )
        else:
            raise ValueError(
                f"Unknown unmasking strategy '{self._decoding}'. Supported types are " f"('l2r', 'r2l', 'ctl')"
            )

    def run_single(self, inputs, preprocess_params, forward_params, postprocess_params) -> DistractorGenerationOutput:
        model_inputs = self.preprocess(inputs, **preprocess_params)
        all_outputs = self.generate(model_inputs, forward_params, postprocess_params)
        kept, discarded = self._evaluate_answer_distractors(
            inputs=inputs, outputs=all_outputs, top_k=postprocess_params["top_k"]
        )
        return DistractorGenerationOutput(distractors=kept, discarded_distractors=discarded)
