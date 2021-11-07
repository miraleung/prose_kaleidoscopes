# Performs edit-distance filtering on a dataset.
#
# Usage example:
#
#  python farthest_main.py --in src_data/small_amazon.json  --field "overall" --out /tmp/test.json --thres_p = 0.4
#

import argparse
import sys

import pandas as pd
import numpy as np

# Own files
import dataset_io
import utils
import main_utils

__author__ = "Anonymized Author of Prose Kaleidoscopes"

def get_farthest(df, orig_idx, field_name, thres_p):
  grp_df = df.loc[df['orig_index'] ==  orig_idx]
  grp_vals = getattr(grp_df, field_name).to_numpy().tolist()
  farthest = len(grp_vals) * [None]
  farthest[0] = grp_vals[0]  # Original.
  distances = utils.ldists(farthest[0], grp_vals)

  for i in range(1, len(grp_vals)):
    lastlen = len(farthest[i - 1])
    curr_idx = np.argmax(distances)
    min_dist = int(thres_p * lastlen)
    if distances[curr_idx] < min_dist:
      break
    farthest[i] = grp_vals[curr_idx]
    distances = np.minimum(distances, utils.ldists(farthest[i], grp_vals))

  return list(filter(None, farthest))

def to_farthest_df(df, field_name, thres_p):
  text_col_index = df.columns.tolist().index(field_name)
  df_np = df.to_numpy()
  # Add one for the original row, plus some extra room because EDA may
  # return more data than we think.
  new_data = np.ndarray((df_np.shape[0] * 6, df_np.shape[1]),
                         dtype=df_np.dtype)

  curr_index = 0

  # Assumes the file has orig_index, which comes from normalizing the file
  # when orig_index wasn't present (aka the original version).
  orig_idxs = getattr(df, "orig_index").to_numpy().tolist()
  orig_idxs_unique = list(set(orig_idxs))
  group_df = df.groupby("orig_index").head([1])
  first_entries_np = getattr(group_df, "orig_index").to_numpy()
  num_groups = len(first_entries_np)
  for i in range(num_groups):
    orig_idx = first_entries_np[i]
    farthest = get_farthest(df, orig_idx, field_name, thres_p)
    if i % 10 == 0:
      sys.stdout.write(
          "Farthest for {0} / {1}:\n\tOriginal: {2}\n\tNew: {3}\n".format(
            i, len(orig_idxs_unique), farthest[0], farthest[1:]))
    for f in farthest:
      new_data[curr_index] = group_df.to_numpy()[i]
      new_data[curr_index][text_col_index] = f
      curr_index += 1

  # Prune any leftover empty rows.
  # Condition: orig_index is not None (although any other column works too).
  new_data = new_data[np.array([r[0] is not None for r in new_data])]
  return pd.DataFrame(data=new_data, columns=df.columns)


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
  DEFAULT_THRES_P = 0.35
  arg_parser.add_argument("--thres_p", required=False, type=float,
      nargs='?', const=DEFAULT_THRES_P,
      help="Percentage threshold for distance")

  args = arg_parser.parse_args()

  # Validation.
  if dataset_io._getext(args.input) != dataset_io._getext(args.out):
    raise ValueError("Input file type is not equal to output type")

  # If the input and output files are the same, verify whether the user
  # intended to overwrite the original file.
  should_continue = main_utils.should_continue(args.input, args.out)
  if not should_continue:
    exit(0)

  # Parse.
  df = dataset_io.to_df(args.input)
  thres_p = DEFAULT_THRES_P if not args.thres_p else args.thres_p
  print("Using thres_p = {0}".format(thres_p))
  new_df = to_farthest_df(df, args.field, thres_p)

  # Write.
  # We assume the input dataset has already marked the original rows.
  dataset_io.to_file(new_df, args.out, index=False)

if __name__ == "__main__":
  main()

