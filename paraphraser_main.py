# Augments a dataset via paraphrasing.
#
# Usage example:
#
#  python sample_extractor_main.py --in src_data/small_amazon.json --num_samples 500 --label_field "overall" --out /tmp/test.json
#

import argparse
import os

import pandas as pd
import numpy as np

# Own files
import dataset_io
import main_utils

from paraphraser import Paraphraser

__author__ = "Mira Leung"

def para(paraphraser, df, field_name, destpath, leven_distp=0.1):
  return paraphraser.augment_df(df, field_name, destpath, leven_distp=leven_distp)

def main():
  arg_parser = argparse.ArgumentParser("Paraphraser argument parser")

  # Required.
  arg_parser.add_argument("--input", required=True, type=str,
      help="Path to input dataset file")
  arg_parser.add_argument("--out", required=True, type=str,
      help="Path to output dataset file")
  arg_parser.add_argument("--field", required=True, type=str,
      help="Name of the text field to augment")

  # Optional.
  arg_parser.add_argument("--leven", required=False, type=float,
      help="Minimum Leven distance threshold as a percentage of word length")


  args = arg_parser.parse_args()

  # Default values.
  leven_distp = 0.1 if args.leven is None else args.leven

  # Validation.
  if dataset_io._getext(args.input) != dataset_io._getext(args.out):
    raise ValueError("Input file type is not equal to output type")

  outdir = "/".join(args.out.split("/")[:-1])
  if not os.path.isdir(outdir):
    raise ValueError("Output directory {0} does not exist".format(outdir))

  # If the input and output files are the same, verify whether the user
  # intended to overwrite the original file.
  should_continue = main_utils.should_continue(args.input, args.out)
  if not should_continue:
    exit(0)

  # Setup.
  paraphraser = Paraphraser()

  # Parse.
  df = dataset_io.to_df(args.input)
  open(args.out, 'w').close()
  para(paraphraser, df, args.field, args.out, leven_distp=leven_distp)

if __name__ == "__main__":
  main()

