import argparse
import json
import warnings

import numpy as np
from tqdm import tqdm
from transformers import FillMaskPipeline, AutoModelForMaskedLM, AutoTokenizer

from disgem import MaskedLMBasedDistractorGenerator
from disgem.data_loader import ClothLoader, CdgpClothLoader, SquadLoader, DGenLoader
from disgem.util import read_json, harmonic_mean


def create_args():
	parser = argparse.ArgumentParser(
		prog="DisGeM",
		description="Distractor Generator for MCQ"
		)
	parser.add_argument("filepath", type=str, help="Path to SQuAD style data.")
	parser.add_argument("--data-format", type=str, default="squad",
	                    help="Data format whether SQuAD style or CLOTH style dataset. Default 'squad'. Available formats [cloth, cdgp-cloth, squad, dgen]")
	parser.add_argument("--model", type=str, default="roberta-large", help="Masked LM for distractor generation phase. Models are loaded from huggingface hub. Default 'roberta-large'.")
	parser.add_argument("--top-k", type=int, default=3, help="Number of distractors. By default 3.")
	parser.add_argument("--batch-size", type=int, default=1, help="Batch size, batched inference might be even slower, "
	                                                              "see https://huggingface.co/docs/transformers/main_classes/pipelines#pipeline-batching. By default 1.")
	parser.add_argument("--output-path", type=str, default=None,
	                    help="File path to dump outputs. By default no output file is created.")
	parser.add_argument("--question-limit", type=int, default=100, help="Question limit to stop generation at. Default 100.")
	parser.add_argument("--dispersion", type=int, default=1, help="Dispersion parameter to determine interval for sampling num mask tokens. By default 1.")
	parser.add_argument("--device", type=int, default=-1, help="Device for generation phase. Set -1 for cpu, numbers 0,1,2,... refer to that gpu device. By default -1.")
	parser.add_argument("--no-minify-output", action="store_true", help="If given, no minification is placed on outputs.")
	parser.add_argument("--strategy", type=str, default="snowball", help="Generation strategy for the generation phase.By default 'snowball'.")
	parser.add_argument("--n-mask", type=int, default=None, help="Number of mask tokens to be replaced with answer text. Default `none`.")
	parser.add_argument("--use-geometric-mean", action="store_true", help="If given, uses geometric mean to determine final ranking, otherwise uses harmonic mean.")
	parser.add_argument("--single-mask", action="store_true", help="If given, only applies a single mask to replace the answer. It is the same as setting `dispersion=0` and `n_mask=1`.")
	parser.add_argument("--seed", type=int, default=42, help="Seed for RNG. Default 42.")
	parser.add_argument("--evaluate", action="store_true", help="If given, starts evaluation process rather than generation. You must supply result json file for evaluation.")
	return parser.parse_args()


def main(args):
	if args.batch_size > 1:
		warnings.warn("Currently, batched inference is not supported.")
		args.batch_size = 1
	if args.data_format == "cloth":
		data_loader = ClothLoader(args.filepath)
		model = AutoModelForMaskedLM.from_pretrained("roberta-large")
		tokenizer = AutoTokenizer.from_pretrained("roberta-large")
		cloth_fill_pipe = FillMaskPipeline(model, tokenizer)
	elif args.data_format == "cdgp-cloth":
		data_loader = CdgpClothLoader(args.filepath)
	elif args.data_format == "dgen":
		data_loader = DGenLoader(args.filepath)
	else:
		data_loader = SquadLoader(args.filepath)

	distractor_generator = MaskedLMBasedDistractorGenerator(
		pretrained_model_name_or_path=args.model, 
		dispersion=args.dispersion,
		n_mask=args.n_mask,
		device=args.device,
		strategy=args.strategy,
		single_mask=args.single_mask
	)

	outputs = []
	count = 0
	for instance in tqdm(data_loader):
		if count == args.question_limit:
			break

		ctx = instance.context
		if args.data_format == "cloth":
			if len(tokenizer.encode(ctx)) > tokenizer.model_max_length:
				# Skip if tokenized context does not fit into model max input length
				continue
			pipe_out = cloth_fill_pipe(ctx, top_k=1)
			for out in pipe_out:
				substr = "<mask>"
				mask_idx = ctx.find(substr)
				filled_str = out[0]["token_str"]
				ctx = ctx.replace(substr, filled_str, 1)
				if -1 < mask_idx < instance.answer["start"]:
					# For replaced mask tokens we need to fix the answer start and end
					# positions to not break the generation process. -1 means not found.
					char_displacement = len(filled_str) - len(substr)
					instance.answer["start"] += char_displacement
					instance.answer["end"] += char_displacement
		generations = distractor_generator(
				context=ctx,
				answer=instance.answer,
				minify_output=not args.no_minify_output,
				top_k=args.top_k,
				use_harmonic_mean=not args.use_geometric_mean,
				batch_size=args.batch_size
		)
		outputs.append(
				{
					"generations": generations,
					"distractors": instance.distractors
				}
		)
		count += 1

	if args.output_path is not None:
		with open(args.output_path, "w") as fd_out:
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
			return 0.
		r = [int(generation in targets) for generation in preds]
		idcg = dcg_at_k(sorted(r, reverse=True), k)
		if not idcg:
			return 0.
		return dcg_at_k(r, k) / idcg

	outputs = read_json(args.filepath)
	avg_eval = {
		"P@1"   : 0.0, "P@3": 0.0, "P@5": 0.0, "P@10"  : 0.0,
		"R@1": 0.0, "R@3": 0.0, "R@5": 0.0, "R@10": 0.0,
		"F1@1": 0.0,  "F1@3": 0.0,   "F1@5"  : 0.0, "F1@10": 0.0,
		"NDCG@1": 0.0, "NDCG@3": 0.0, "NDCG@5": 0.0, "NDCG@10": 0.0}
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
	if args.evaluate:
		evaluate(args)
	else:
		main(args)
