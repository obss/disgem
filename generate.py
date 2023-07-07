import argparse
import json
from pathlib import Path

from tqdm import tqdm

from disgem import MaskedLMBasedDistractorGenerator


def create_args():
	parser = argparse.ArgumentParser(
		prog="DisGeM",
		description="Distractor Generator for MCQ"
		)
	parser.add_argument("filepath", type=str, help="Path to SQuAD style data.")
	parser.add_argument("--model", type=str, default="roberta-large", help="Masked LM for distractor generation phase. Models are loaded from huggingface hub.")
	parser.add_argument("--top-k", type=int, default=3, help="Number of distractors.")
	parser.add_argument("--dispersion", type=int, default=1, help="Dispersion parameter to determine interval for sampling num mask tokens.")
	parser.add_argument("--device", type=int, default=-1, help="Device for generation phase. Set -1 for cpu, numbers 0,1,2,... refer to that gpu device.")
	parser.add_argument("--minify-output", type=bool, default=True, help="Whether to minify outputs without residues.")
	parser.add_argument("--strategy", type=str, default="snowball", help="Generation strategy for the generation phase.")
	parser.add_argument("--n-mask", type=int, default=None, help="Number of mask tokens to be replaced with answer text.")
	parser.add_argument("--seed", type=int, default=42, help="Seed for RNG.")
	return parser.parse_args()

def read_squad(fp: Path):
	"""
	Reads squad style data given filepath 'fp'.
	"""
	with open(fp, 'r') as fd_in:
		data = json.load(fd_in)
	return data["data"]


def main(args):
	file = Path(args.filepath)
	data = read_squad(file)

	distractor_generator = MaskedLMBasedDistractorGenerator(
		pretrained_model_name_or_path=args.model, 
		dispersion=args.dispersion,
		n_mask=args.n_mask,
		device=args.device,
		strategy=args.strategy,
	)

	for article in data:
		for paragraph in tqdm(article["paragraphs"]):
			ctx = paragraph["context"]
			for qa in paragraph["qas"]:
				answer = qa["answers"][0]
				answer["start"] = answer.pop("answer_start")
				answer["end"] = answer["start"] + len(answer["text"])
				print(distractor_generator(context=ctx, answer=answer, top_k=args.top_k, minify_output=args.minify_output))
    

if __name__ == "__main__":
	args = create_args()
	main(args)
	