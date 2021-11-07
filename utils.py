import leven
import hashlib

import pandas as pd
import numpy as np

__author__ = "Anonymized Author of Prose Kaleidoscopes"

# Counts the number of characters in a pandas' dataframe's field.
def wc(df, field_name, start=0, end=None):
  return sum([len(s) for s in getattr(df, field_name).to_numpy().tolist()[start:end]])

def get_words(df, field_name):
  sents = getattr(df, field_name).to_numpy().tolist()
  return [word for segs in [sent.split() for sent in sents] for word in segs]

# Returns the type-to-token ratio of a DataFrame.
def ttr(df, field_name):
  words = get_words(df, field_name)
  return len(set(words)) / len(words)

# Returns the average levenshtein distance over all sentences (per-group) in a dataset.
# Indexes on orig_index.
def avg_ldist(df, field_name, debug=False):
  orig_idxes = list(set(getattr(df, "orig_index").to_numpy().tolist()))
  num_orig_data = len(orig_idxes)
  all_avg_dists = [None] * num_orig_data
  for i in range(len(orig_idxes)):
    idx = orig_idxes[i]
    sents = getattr(df.loc[df['orig_index'] == idx], field_name).to_numpy().tolist()
    if len(sents) <= 1:
      continue

    count = len(sents)
    dist_sum = 0
    orig_sent = sents[0]
    for j in range(1, len(sents)):
      dist_sum += leven.levenshtein(orig_sent, sents[j]) / len(orig_sent)
    all_avg_dists[i] = dist_sum / count

  all_avg_dists = list(filter(lambda x : x is not None, all_avg_dists))
  if debug:
    print(all_avg_dists)

  return sum(all_avg_dists) / num_orig_data


# Returns a list of levenshtein distances - from p0 to points.
def ldists(p0, points):
  return [leven.levenshtein(p0, p) for p in points]

# Returns df1 minus df2 by label_field_name.
# e.g. new_remainder = df_minus(original, small)
def df_minus(df1, df2, label_field_name="orig_index"):
  lst = getattr(df2, label_field_name).to_numpy().tolist()
  return df1.query("{0} not in @lst".format(label_field_name))


# Returns a sort of LSH of a string.
# Algo: Char ASCII value * char position in string, summed up over a sentence
# and normalized by the length of the string times the max ASCII value
# ('~' in this case).
def cdhash(sent):
  temp_cs = [c for w in sent.split() for c in w]
  max_ascii_val = 126
  total = 0
  for i in range(len(temp_cs)):
    total += i * ord(temp_cs[i])
  return total / (len(temp_cs) * max_ascii_val)

def md5hash(sent):
  return int(hashlib.md5(sent.encode()).hexdigest(), 16) / 1e37

def lhash(sent):
  return leven.levenshtein(sent, ' ' * len(sent))

def abshash(sent):
  nullconst = 32  # ord(' ')
  cs = [ord(c) for w in sent.split() for c in w]
  return sum(cs) / (nullconst * len(cs))
