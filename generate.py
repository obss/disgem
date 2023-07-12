import argparse
import json
import warnings

from tqdm import tqdm
from transformers import FillMaskPipeline, AutoModelForMaskedLM, AutoTokenizer

from disgem import MaskedLMBasedDistractorGenerator
from disgem.data_loader import ClothLoader, CdgpLoader, SquadLoader


def create_args():
	parser = argparse.ArgumentParser(
		prog="DisGeM",
		description="Distractor Generator for MCQ"
		)
	parser.add_argument("filepath", type=str, help="Path to SQuAD style data.")
	parser.add_argument("--data-format", type=str, default="squad",
	                    help="Data format whether SQuAD style or CLOTH style dataset. Default 'squad'.")
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
	parser.add_argument("--use-geometric-mean", action="store_true", help="If given, uses geometric mean to determine final ranking. Default is harmonic mean.")
	parser.add_argument("--seed", type=int, default=42, help="Seed for RNG. Default 42.")
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
	elif args.data_format == "cdgp":
		data_loader = CdgpLoader(args.filepath)
	else:
		data_loader = SquadLoader(args.filepath)

	distractor_generator = MaskedLMBasedDistractorGenerator(
		pretrained_model_name_or_path=args.model, 
		dispersion=args.dispersion,
		n_mask=args.n_mask,
		device=args.device,
		strategy=args.strategy,
	)

	outputs = []
	for i, instance in enumerate(tqdm(data_loader)):
		ctx = instance.context
		if args.data_format == "cloth":
			pipe_out = cloth_fill_pipe(ctx, top_k=1)
			for out in pipe_out:
				ctx = ctx.replace("<mask>", out[0]["token_str"], 1)
		generations = distractor_generator(
				context=ctx,
				answer=instance.answer,
				minify_output=not args.no_minify_output,
				top_k=args.top_k,
				use_harmonic_mean=not args.use_geometric_mean,
				batch_size=args.batch_size,
		)
		outputs.append(
				{
					"generations": generations,
					"distractors": instance.distractors
				}
		)
		if i == args.question_limit:
			break

	if args.output_path is not None:
		with open(args.output_path, "w") as fd_out:
			json.dump(outputs, fd_out)


if __name__ == "__main__":
	args = create_args()
	main(args)
	