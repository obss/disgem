import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer, FillMaskPipeline, set_seed

from disgem import MaskedLMBasedDistractorGenerator
from disgem.data_loader import CdgpClothLoader, ClothLoader, DGenLoader, SquadLoader
from disgem.util import harmonic_mean, read_json


def create_args():
    parser = argparse.ArgumentParser(prog="DisGeM", description="Distractor Generator for MCQ")
    parser.add_argument("filepath", type=str, help="Path to SQuAD style data.")
    parser.add_argument(
        "--data-format",
        type=str,
        default="squad",
        choices=["cloth", "cdgp-cloth", "squad", "dgen"],
        help="Data format whether SQuAD style or CLOTH style dataset. Default 'squad'.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="roberta-large",
        help="Masked LM for distractor generation phase. Models are loaded from huggingface hub. Default 'roberta-large'.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of distractors. By default 3.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size, batched inference might be even slower, "
        "see https://huggingface.co/docs/transformers/main_classes/pipelines#pipeline-batching. By default 1.",
    )
    parser.add_argument(
        "--output-path", type=str, default=None, help="File path to dump outputs. By default no output file is created."
    )
    parser.add_argument("--output-format", type=str, default="cdgp", choices=["cdgp", "all"])
    parser.add_argument(
        "--question-limit", type=int, default=100, help="Question limit to stop generation at. Default 100."
    )
    parser.add_argument(
        "--dispersion",
        type=int,
        default=1,
        help="Dispersion parameter to determine interval for sampling num mask tokens. By default 1.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Device for generation phase. Set -1 for cpu, numbers 0,1,2,... refer to that gpu device. By default -1.",
    )
    parser.add_argument(
        "--no-minify-output", action="store_true", help="If given, no minification is placed on outputs."
    )
    parser.add_argument(
        "--decoding",
        type=str,
        default="l2r",
        choices=["l2r", "r2l", "ctl"],
        help="Generation strategy for the generation phase.By default 'snowball'.",
    )
    parser.add_argument(
        "--n-mask",
        type=int,
        default=None,
        help="Number of mask tokens to be replaced with answer text. Default `none`.",
    )
    parser.add_argument(
        "--use-geometric-mean",
        action="store_true",
        help="If given, uses geometric mean to determine final ranking, otherwise uses harmonic mean.",
    )
    parser.add_argument(
        "--single-mask",
        action="store_true",
        help="If given, only applies a single mask to replace the answer. It is the same as setting `dispersion=0` and `n_mask=1`.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for RNG. Default 42.")
    parser.add_argument(
        "--prepend-question",
        type=str,
        default="none",
        choices=["none", "begin", "mid"],
        help="If not `none`, prepends `question` to the context to guide the distractor generation with the question. "
        "Default option is `none`.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="If given, starts evaluation process rather than generation. You must supply result json file for evaluation.",
    )
    return parser.parse_args()


def main(args):
    if args.prepend_question != "none":
        warnings.warn("`--prepend-question` is only available for squad format.")
    if args.batch_size > 1:
        warnings.warn("Currently, batched inference is not supported.")
        args.batch_size = 1
    if args.data_format == "cloth":
        data_loader = ClothLoader(args.filepath)
    elif args.data_format == "cdgp-cloth":
        data_loader = CdgpClothLoader(args.filepath)
    elif args.data_format == "dgen":
        data_loader = DGenLoader(args.filepath)
    elif args.data_format == "squad":
        data_loader = SquadLoader(args.filepath, prepend_question=args.prepend_question)
    else:
        raise ValueError(f"Unknown data format {args.data_format}.")

    distractor_generator = MaskedLMBasedDistractorGenerator(
        pretrained_model_name_or_path=args.model,
        dispersion=args.dispersion,
        n_mask=args.n_mask,
        device=args.device,
        decoding=args.decoding,
        single_mask=args.single_mask,
    )

    squad_answers = []
    outputs = []
    count = 0
    pbar = tqdm(data_loader)
    for instance in pbar:
        pbar.set_postfix({"count": count})
        if count == args.question_limit:
            break

        dgen_tokenizer = distractor_generator._pipeline.tokenizer
        if len(dgen_tokenizer.encode(instance.context)) > dgen_tokenizer.model_max_length:
            # Skip if tokenized context does not fit into model max input length
            continue

        if args.data_format == "squad":
            if args.prepend_question:
                pass
            elif instance.answer in squad_answers:
                # squad contains different questions for some answer spans. Since our
                # framework does not depend on question, we skip these questions as
                # it would yield the same distractors.
                continue
            else:
                squad_answers.append(instance.answer)

        generations = distractor_generator(
            context=instance.context,
            answer=instance.answer,
            minify_output=not args.no_minify_output,
            top_k=args.top_k,
            use_harmonic_mean=not args.use_geometric_mean,
            batch_size=args.batch_size,
        )
        if args.data_format == "squad":  # no gt distractors/evaluation, put context as well
            outputs.append(
                {
                    "context": instance.context,
                    "question": instance.question,
                    "answer": instance.answer,
                    "generations": generations,
                }
            )
        else:
            if args.output_format == "cdgp":
                outputs.append({"generations": generations, "distractors": instance.distractors})
            else:
                # For better readability, put blank in the output
                ctx = (
                    instance.context[: instance.answer["start"]] + " ____ " + instance.context[instance.answer["end"] :]
                )
                outputs.append(
                    {
                        "context": ctx,
                        "answer": instance.answer["text"],
                        "generations": generations,
                        "distractors": instance.distractors,
                    }
                )
        count += 1

    if args.output_path is not None:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path.as_posix(), "w") as fd_out:
            json.dump(outputs, fd_out)


def evaluate(args):
    """

    Args:
            args:

    Returns:

    """

    # metrics
    def precision(preds, targets, k: int = 1):
        matches = [int(generation in targets) for generation in preds]
        return sum(matches[:k]) / k

    def recall(preds, targets, k: int = 1):
        matches = [int(generation in targets) for generation in preds]
        return sum(matches[:k]) / len(targets)

    def f1(preds, targets, k: int = 1):
        p = precision(preds, targets, k)
        r = recall(preds, targets, k)
        return harmonic_mean([p, r])

    def ndcg_at_k(preds, targets, k: int = 1):
        def dcg_at_k(r, k):
            r = np.asfarray(r)[:k]
            if r.size:
                return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
            return 0.0

        r = [int(generation in targets) for generation in preds]
        idcg = dcg_at_k(sorted(r, reverse=True), k)
        if not idcg:
            return 0.0
        return dcg_at_k(r, k) / idcg

    def mmr_at_k(preds, targets, k: int = 1):
        matches = [int(generation in targets) for generation in preds]
        k = len(matches) if k > len(matches) else k
        for i in range(k):
            if matches[i] == 1:
                return 1 / (i + 1)
        return 0.0

    outputs = read_json(args.filepath)
    avg_eval = {
        "P@1": 0.0,
        "P@3": 0.0,
        "P@5": 0.0,
        "P@10": 0.0,
        "R@1": 0.0,
        "R@3": 0.0,
        "R@5": 0.0,
        "R@10": 0.0,
        "F1@1": 0.0,
        "F1@3": 0.0,
        "F1@5": 0.0,
        "F1@10": 0.0,
        "MRR@1": 0.0,
        "MRR@3": 0.0,
        "MRR@5": 0.0,
        "MRR@10": 0.0,
        "NDCG@1": 0.0,
        "NDCG@3": 0.0,
        "NDCG@5": 0.0,
        "NDCG@10": 0.0,
    }
    for output in outputs:
        distractors = [d.lower() for d in output["distractors"]]
        generations = [d.lower() for d in output["generations"]]

        for key in avg_eval.keys():
            metric, k = key.split("@")
            if metric == "P":
                metric_fn = precision
            elif metric == "R":
                metric_fn = recall
            elif metric == "F1":
                metric_fn = f1
            elif metric == "NDCG":
                metric_fn = ndcg_at_k
            elif metric == "MRR":
                metric_fn = mmr_at_k
            else:
                continue
            avg_eval[key] += metric_fn(preds=generations, targets=distractors, k=int(k))

    # calculate average
    for key in avg_eval.keys():
        avg_eval[key] /= len(outputs)
        avg_eval[key] = str(round(100 * avg_eval[key], 4)) + "%"

    print(json.dumps(avg_eval, indent=2))
    if args.output_path is not None:
        with open(args.output_path, "w") as fd_out:
            json.dump(avg_eval, fd_out, indent=2)


if __name__ == "__main__":
    args = create_args()
    set_seed(args.seed)
    if args.evaluate:
        evaluate(args)
    else:
        main(args)
