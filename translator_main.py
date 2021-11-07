# Extracts n dataset samples per label
#
# Usage example:
#
#  python translator_main.py --in src_data/small_amazon.json --num_samples 500 --field "overall" --out /tmp/test.json
#

import argparse

import pandas as pd
import numpy as np

# Own files
import dataset_io
import main_utils

from translator import Translator

def main():
  arg_parser = argparse.ArgumentParser("Translator argument parser")

  # Required.
  arg_parser.add_argument("--input", required=True, type=str,
      help="Path to input dataset file")
  arg_parser.add_argument("--out", required=True, type=str,
      help="Path to output dataset file")
  arg_parser.add_argument("--field", required=True, type=str,
      help="Name of the text field to augment")

  # Optional.
  arg_parser.add_argument("--use_actual", required=False,
      action='store_true',
      help="Use the actual Google Cloud Translate API")

  args = arg_parser.parse_args()

  # Validation.
  if dataset_io._getext(args.input) != dataset_io._getext(args.out):
    raise ValueError("Input file type is not equal to output type")

  # If the input and output files are the same, verify whether the user
  # intended to overwrite the original file.
  should_continue = main_utils.should_continue(args.input, args.out)
  if not should_continue:
    exit(0)

  # Setup.
  use_actual = False if args.use_actual is None else args.use_actual
  print("Using Google Translate API directly? " + str(use_actual))
  translator = Translator(use_actual)

  # Parse.
  df = dataset_io.to_df(args.input)
  new_df = translator.augment_df(df, args.field)

  # Write.
  # We assume the input dataset has already marked the original rows.
  dataset_io.to_file(new_df, args.out, index=False)

if __name__ == "__main__":
  main()

