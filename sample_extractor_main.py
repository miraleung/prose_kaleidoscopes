# Extracts n random dataset samples.
#
# Usage example:
#
#  python sample_extractor_main.py --in src_data/small_amazon.json --num_samples 500 --label_field "overall" --out /tmp/test.json
#

import argparse

import pandas as pd
import numpy as np

# Own files
import dataset_io

__author__ = "Mira Leung"

def get_weights(input_filepath, df, label_field):
  labels = getattr(df, label_field).to_numpy().tolist()
  if "amazon" in input_filepath:
    # Weights of the original dataset. Label order:R 1, 2, 3,4 , 5.
    weights = [0.04313356, 0.0465181 , 0.10492072, 0.2555559 , 0.54987173]
    freq_lst = list(map(lambda x : weights[x - 1], labels))
    return freq_lst
  if "imdb" in input_filepath:
    weights = [0.5, 0.5]
    freq_lst = list(map(lambda _ : 0.5, labels))
    return freq_lst
  if "sst2" in input_filepath:
    # Weights of the original dataset. Label order: 0 (negative), 1 (positive)
    weights = [0.44217435, 0.55782565]
    freq_lst = list(map(lambda x : weights[x], labels))
    return freq_lst

  raise ValueError("Unsupported dataset {0}".format(input_filepath))


def main():
  arg_parser = argparse.ArgumentParser("Sample extractor argument parser")

  # Required.
  arg_parser.add_argument("--input", required=True, type=str,
      help="Path to input dataset file")
  arg_parser.add_argument("--out", required=True, type=str,
      help="Path to output dataset file")
  arg_parser.add_argument("--num_samples", required=True, type=int,
      help="Number of samples, times the number of labels, to extract")
  arg_parser.add_argument("--label_field", required=True, type=str,
      help="Name of the label field")

  args = arg_parser.parse_args()

  # Validation.
  if dataset_io._getext(args.input) != dataset_io._getext(args.out):
    raise ValueError("Input file type is not equal to output type")
  if args.num_samples <= 0:
    raise ValueError("Num_samples must be a positive integer")

  # If the input and output files are the same, verify whether the user
  # intended to overwrite the original file.
  should_continue = args.input != args.out
  while not should_continue:
    should_continue = input(
        "Are you sure you want to overwrite the original dataset? [y/n]")
    if should_continue.lower() == "n":
      exit(0)
    should_continue = should_continue.lower() == "y"

  # Parse.
  df = dataset_io.to_df(args.input)
  num_samples = min(args.num_samples, len(df.to_numpy()))
  freqs_lst = get_weights(args.input, df, args.label_field)
  num_labels = len(set(getattr(df, args.label_field)))

  new_df = df.sample(num_samples * num_labels, weights=freqs_lst)

  # Equal selection of all labels.
  # new_df = df.sample(frac = 1.0).groupby(args.label_field).head(num_samples)

  if args.label_field is not None:
    print(getattr(new_df, args.label_field).value_counts())

  # Write.
  # We assume the input dataset has already marked the original rows.
  dataset_io.to_file(new_df, args.out, index=False)

if __name__ == "__main__":
  main()

