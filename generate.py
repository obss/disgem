import argparse
import json
from pathlib import Path

from tqdm import tqdm

from disgem import MaskedLMBasedDistractorGenerator
from disgem.util import ClothLoader, SquadLoader


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
	parser.add_argument("--dispersion", type=int, default=1, help="Dispersion parameter to determine interval for sampling num mask tokens. By default 1.")
	parser.add_argument("--device", type=int, default=-1, help="Device for generation phase. Set -1 for cpu, numbers 0,1,2,... refer to that gpu device. By default -1.")
	parser.add_argument("--no-minify-output", action="store_true", help="If given, no minification is placed on outputs.")
	parser.add_argument("--strategy", type=str, default="snowball", help="Generation strategy for the generation phase.By default 'snowball'.")
	parser.add_argument("--n-mask", type=int, default=None, help="Number of mask tokens to be replaced with answer text. Default `none`.")
	parser.add_argument("--use-geometric-mean", action="store_true", help="If given, uses geometric mean to determine final ranking. Default is harmonic mean.")
	parser.add_argument("--seed", type=int, default=42, help="Seed for RNG. Default 42.")
	return parser.parse_args()


def read_squad(fp: Path):
	"""
	Reads squad style data given filepath 'fp'.
	"""
	with open(fp, 'r') as fd_in:
		data = json.load(fd_in)
	return data["data"]


def main(args):
	if args.data_format == "cloth":
		data_loader = ClothLoader(args.filepath)
	else:
		data_loader = SquadLoader(args.filepath)

	distractor_generator = MaskedLMBasedDistractorGenerator(
		pretrained_model_name_or_path=args.model, 
		dispersion=args.dispersion,
		n_mask=args.n_mask,
		device=args.device,
		strategy=args.strategy,
	)

	for instance in tqdm(data_loader):
		for answer in instance.answers:
			tqdm.write(str(distractor_generator(
					context=instance.context,
					answer=answer,
					top_k=args.top_k,
					use_harmonic_mean=not args.use_geometric_mean)
			))


if __name__ == "__main__":
	args = create_args()
	main(args)
	