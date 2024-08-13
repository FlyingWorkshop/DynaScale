import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, help='Specify where to save results')
parser.add_argument('--iteration', type=int, help='Specify which iteration to run')

args = parser.parse_args()
args.output_dir, args.iteration
print(args.iteration)
os.makedirs(args.output_dir, exist_ok=True)
with open(os.path.join(args.output_dir, "output.txt"), "a") as f:
    f.write(str(args.iteration))