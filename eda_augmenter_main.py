# Augments an NLP dataset with EDA.
#
# Usage example:
#
#  python augmenter_main.py --in src_data/small_imdb.csv --field "review" --out /tmp/test.csv
#

import argparse
import re
import sys

import pandas as pd
import numpy as np

# Own files
from eda_augmenter import EdaAugmenter
import eda  # From EDA paper
import dataset_io
import main_utils

__author__ = "Mira Leung"

def eda_augment(augmenter, df, field_name, destpath):
  return augmenter.eda(df, field_name, destpath)

def main():
  arg_parser = argparse.ArgumentParser("EDA argument parser")

  # Required.
  arg_parser.add_argument("--input", required=True, type=str,
      help="Path to input dataset file")
  arg_parser.add_argument("--out", required=True, type=str,
      help="Path to output dataset file")
  arg_parser.add_argument("--field", required=True, type=str,
      help="Name of the text field to augment")

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
  EDA_AUGMENTER = EdaAugmenter()

  # Parse.
  df = dataset_io.to_df(args.input)

  new_df = None
  eda_augment(EDA_AUGMENTER, df, args.field, args.out)

if __name__ == "__main__":
  main()

