# Normalizes an NLP dataset.
#
# Usage example:
#
#  python normalizer_main.py --in src_data/small_imdb.csv --field "review" --out /tmp/test.csv
#

import argparse
import pandas as pd
import sys

# Own files.
import dataset_io
import main_utils

from normalizer import Normalizer

__author__ = "Mira Leung"

def _norm_update(normalizer, df, field_name):
  """
  Normalizes the text in field_name. Modifies df.
  """
  values = getattr(df, field_name)
  num_values = len(values)
  # Pre-allocate for speed.
  new = num_values * [None]
  for i in range(num_values):
    if i % 10 == 0:
      sys.stdout.write("Normalizing {0} / {1} \r".format(i, num_values))
      sys.stdout.flush()
    new[i] = normalizer.normalize(values[i])
  setattr(df, field_name, pd.Series(new))


def main():
  arg_parser = argparse.ArgumentParser("File normalizer argument parser")

  # Required.
  arg_parser.add_argument("--input", required=True, type=str,
      help="Path to input dataset file")
  arg_parser.add_argument("--out", required=True, type=str,
      help="Path to output dataset file")
  arg_parser.add_argument("--field", required=True, type=str,
      help="Name of the text field to normalize")

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
  normalizer = Normalizer()

  # Parse.
  df = dataset_io.to_df(args.input)

  # Handle index columns.
  write_index = False
  if "orig_index" not in df.columns:
    write_index = True
    if df.index.name is None:
      df.index.name = "orig_index"
    # For JSON files, do this to include the new index column.
    if args.input.endswith("json"):
      df.reset_index(inplace=True)

  # Normalize.
  print("Normalizing {0}".format(args.input))
  _norm_update(normalizer, df, args.field)

  # Write.
  dataset_io.to_file(df, args.out, index=write_index)


if __name__ == "__main__":
  main()

