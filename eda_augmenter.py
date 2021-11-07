import argparse
import os
import re
import sys

import pandas as pd
import numpy as np

# Own files
import dataset_io
import eda  # From EDA paper

__author__ = "Anonymized Author of Prose Kaleidoscopes"

class EdaAugmenter:
  REGEX_WORD = re.compile("[a-zA-Z]+")

  def eda(self, df, field_name, destpath):
    """
    Returns a new DataFrame with augmented data.
    """
    def eda_params(num_rows):
      """
      Optimal params as per the EDA paper: https://arxiv.org/abs/1901.11196
      """
      # Default values for num_rows <= 500.
      alpha = 0.05
      n_aug = 16
      if num_rows in range(500, 2001):
        n_aug = 8
      elif num_rows > 2000:
        alpha = 0.1
        n_aug = 4
      return alpha, n_aug

    num_orig_rows = len(df.values)
    alpha, n_aug = eda_params(num_orig_rows)
    text_col_index = df.columns.tolist().index(field_name)
    df_np = df.to_numpy()
    num_cols = df_np.shape[1]

    # Number of data points to write to a file in one batch.
    batch_size = 20

    # Add one for the original row, plus some extra room because EDA may
    # return more data than we think.
    new_data = np.ndarray((batch_size * (n_aug + 10), num_cols), dtype=df_np.dtype)
    curr_index = 0
    for i in range(num_orig_rows):
      curr_text = df_np[i][text_col_index]
      should_break = False
      # Skip EDA errors.
      if self.REGEX_WORD.search(curr_text) is None:
        should_break = True
      else:
        new_data[curr_index] = df_np[i]   # Original data.
        curr_index += 1
        augmented_sents = []
        try:
          augmented_sents = eda.eda(curr_text,
                                    alpha_sr=alpha,
                                    alpha_ri=alpha,
                                    alpha_rs=alpha,
                                    p_rd=alpha,
                                    num_aug=n_aug)
        except Exception as e:
          sys.stdout.write(
              "Could not augment L{0}, skipping '{1}'\n".format(i, curr_text))
          sys.stdout.flush()
          should_break = True

      if not should_break:
        for aug_sent in set(augmented_sents):
          new_data[curr_index] = df_np[i]
          new_data[curr_index][text_col_index] = aug_sent
          # Accumulate indices instead of using (i * n_aug + j) becausse there may
          # be fewer than n_aug sentences after EDA.
          curr_index += 1

      if i % batch_size == 0:
        sys.stdout.write("EDA-augmenting {0} / {1}\r".format(i, num_orig_rows))
        sys.stdout.flush()

        # Prune any leftover empty rows.
        new_data = new_data[np.array([r[0] is not None for r in new_data])]
        if not os.path.exists(destpath):
          open(destpath, 'a').close()
        dataset_io.append_to_file(
            pd.DataFrame(data=new_data, columns=df.columns), destpath)

        # Reset the accumulator.
        new_data = np.ndarray((batch_size * (n_aug + 10), num_cols), dtype=df_np.dtype)
        curr_index = 0

    # Prune any leftover empty rows.
    # Condition: orig_index is not None (although any other column works too).
    new_data = new_data[np.array([r[0] is not None for r in new_data])]
    dataset_io.append_to_file(
        pd.DataFrame(data=new_data, columns=df.columns), destpath)
    print("\nWritten to {0}".format(destpath))

