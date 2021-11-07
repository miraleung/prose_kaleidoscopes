import os
import pandas as pd

__author__ = "Anonymized Author of Prose Kaleidoscopes"

def _getext(filepath):
  return os.path.splitext(filepath)[1][1:].lower()  # Remove the first dot.

def is_csv(filepath):
  return _getext(filepath) == "csv"

def is_tsv(filepath):
  return _getext(filepath) == "tsv"

def is_json(filepath):
  return _getext(filepath) == "json"


def to_df(filepath):
  """
  Returns a pands.DataFrame with the parsed file.
  sep applies only to CSV files.
  """
  if is_csv(filepath):
    return pd.read_csv(filepath)
  if is_tsv(filepath):
    return pd.read_csv(filepath, sep="\t")
  if is_json(filepath):
    return pd.read_json(filepath, lines=True)

  raise ValueError("{0} files are not yet supported".format(_getext(filepath)))


def _to_file(df, destpath, index=False, mode='w', header=True):
  if is_csv(destpath):
    return df.to_csv(destpath, mode=mode, index=index, header=header)
  if is_tsv(destpath):
    return df.to_csv(destpath, mode=mode, sep="\t", index=index, header=header)
  if is_json(destpath):
    return df.to_json(destpath, orient="records", lines=True)

  raise ValueError("{0} files are not yet supported".format(_getext(destpath)))

def to_file(df, destpath, index=False):
  return _to_file(df, destpath, index)

def append_to_file(df, destpath, index=False):
  if is_csv(destpath) or is_tsv(destpath):
    is_file_empty = os.stat(destpath).st_size == 0
    return _to_file(df, destpath, index, mode='a', header=is_file_empty)

  if is_json(destpath):
    # Workaround for JSON - rewrite the file each time.
    orig_df = to_df(destpath)
    new_df = orig_df.append(df, ignore_index=True)
    return to_file(new_df, destpath)

  raise ValueError("{0} files are not yet supported".format(_getext(destpath)))

