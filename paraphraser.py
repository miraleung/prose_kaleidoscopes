import argparse
import heapq
import leven
import nltk
import re
import sys
import torch
import warnings

import pandas as pd
import numpy as np

from itertools import groupby
from parrot import Parrot
from transformers import pipeline
from transformers import T5ForConditionalGeneration, T5Tokenizer

import dataset_io

__author__ = "Mira Leung"

warnings.filterwarnings("ignore")

class Paraphraser:
  PARROT = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=False)
  # Defaults to https://huggingface.co/sshleifer/distilbart-cnn-12-6
  HF_SUMMARIZER_PIPELINE = pipeline("summarization")
  MODEL_T5 = T5ForConditionalGeneration.from_pretrained("t5-base")
  TOKENIZER_T5 = T5Tokenizer.from_pretrained("t5-base")

  def __init__(self):
    try:
      nltk.data.find('tokenizers/punkt')
    except LookupError:
      nltk.download('punkt')
    try:
      nltk.data.find('corpora/stopwords')
    except LookupError:
      nltk.download('stopwords')

  def paraphrase(self, text):
    sents = list(filter(lambda x : len(x) > 0, map(lambda s : s.strip(), text.split("."))))
    if len(sents) > 1:
      return None

    results = self.PARROT.augment(
        input_phrase=text,
        diversity_ranker="levenshtein",
        do_diverse=True,
        adequacy_threshold=0.99,
        fluency_threshold=0.90)
    if results is None:
      return None

    # Format is [('text', score)]
    return [r[0] for r in results[:2]]

  def summarize_nltk(self, text):
    sents = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english')
    word_freqs = {
        v : len(list(freq))
        for v, freq in groupby(sorted(list(filter(lambda x : x not in stopwords, words))))
        }
    # Get weighted frequency.
    if len(word_freqs.values()) == 0:
      return None

    max_freq = max(word_freqs.values())
    word_freqs = { k : word_freqs[k] / max_freq for k in word_freqs }

    # Get sentence scores.
    sent_scores = {}
    for s in sents:
      for word in nltk.word_tokenize(s.lower()):
        if word not in word_freqs.keys():
          continue
        if s not in sent_scores.keys():
          sent_scores[s] = 0
        sent_scores[s] += word_freqs[word]

    summary_sents = heapq.nlargest(7, sent_scores, key=sent_scores.get)
    return ' '.join(summary_sents)

  def summarize_hf_pipeline(self, text):
    try:
      num_words = len(text.split())
      input_len = num_words + 3
      min_len = min(3, max(1, input_len))
      max_len = int(0.7 * input_len)
      summary = self.HF_SUMMARIZER_PIPELINE(
          text,
          min_length=min_len,
          max_length=max_len)
    except Exception:
      print("Could not apply HF pipeline summarization to " + text)
      return None

    aug_text = summary[0]['summary_text']
    #print("\tHF Summarizer; Orig: {0}\n\t\tAug: {1}".format(text, aug_text))
    return aug_text

  def summarize_hf_t5(self, text):
    try:
      inputs = self.TOKENIZER_T5.encode(
          "summarize: " + text, return_tensors="pt", truncation=True)
      outputs = self.MODEL_T5.generate(
          inputs,
          min_length=min(10, max(3, len(text))),
          max_length=int(0.3 * len(text)),
          length_penalty=4.0,
          num_beams=4,
          early_stopping=True)
      result = self.TOKENIZER_T5.decode(outputs[0])
    except Exception:
      print("Could not apply T5pipeline summarization to " + text)
      return None

    if result.startswith("<pad>"):
      result = result[5:]
    return result

  def augment_text(self, text, leven_distp):
    sum_nltk = self.summarize_nltk(text)
    sum_hf_pipeline = self.summarize_hf_pipeline(text)
    sum_hf_t5 = self.summarize_hf_t5(text)
    paraphrase_results = self.paraphrase(text)

    results = []
    if sum_nltk is not None:
      results += [sum_nltk]
    if sum_hf_pipeline is not None and len(sum_hf_pipeline) > 0:
      results += [sum_hf_pipeline]
    if sum_hf_t5 is not None and len(sum_hf_t5) > 0:
      results += [sum_hf_t5]
    if paraphrase_results is not None and len(paraphrase_results) > 0:
      results += paraphrase_results

    #print(results)
    # Must have a distance at lease leven_distp percent more than the original.
    filtered_results = list(filter(
      lambda x : leven.levenshtein(text, x) > leven_distp * (len(text)),
      results))

    return filtered_results

  def augment_df(self, df, field_name, destpath, leven_distp=0.1):
    num_orig_rows = len(df.values)
    text_col_index = df.columns.tolist().index(field_name)
    df_np = df.to_numpy()
    num_cols = df_np.shape[1]

    # Number of data points to write to a file in one batch.
    batch_size = 10

    # Add one for the original row, plus some extra room because EDA may
    # return more data than we think.
    new_data = np.ndarray((batch_size * 6, num_cols), dtype=df_np.dtype)
    #new_data = np.ndarray((df_np.shape[0] * 6, df_np.shape[1]),
    #                       dtype=df_np.dtype)

    curr_index = 0
    for i in range(num_orig_rows):
      if i % 10 == 0:
        sys.stdout.write("\n~~~~~~~~ Augmenting {0} / {1} ~~~~~~~~\n".format(
          i, num_orig_rows))
        sys.stdout.flush()

      curr_text = df_np[i][text_col_index]
      new_data[curr_index] = df_np[i]   # Original data.
      curr_index += 1

      # Too short, cannot be augmented.
      if len(curr_text) < 2:
        continue

      augmented_sents = self.augment_text(curr_text, leven_distp)
      if i % 10 == 0:
        sys.stdout.write(
            "\tOrig: {0}. \n\tAug:\n\t{1}\n".format(curr_text, "\n\t".join(augmented_sents)))
        sys.stdout.flush()

      for aug_sent in set(augmented_sents):
        if aug_sent == curr_text or len(aug_sent) == 0:
          continue

        new_data[curr_index] = df_np[i]
        new_data[curr_index][text_col_index] = aug_sent
        # Accumulate indices instead of using (i * n_aug + j) becausse there may
        # be fewer than n_aug sentences.
        curr_index += 1

      if i % batch_size == 0:
        print("\tWriting rows {0} / {1}".format(i, num_orig_rows))
        # Prune any leftover empty rows.
        new_data = new_data[np.array([r[0] is not None for r in new_data])]
        dataset_io.append_to_file(
            pd.DataFrame(data=new_data, columns=df.columns), destpath)

        # Reset the accumulator.
        new_data = np.ndarray((batch_size * 6, num_cols), dtype=df_np.dtype)
        curr_index = 0

    # Prune any leftover empty rows and write this out for the last time.
    # Condition: orig_index is not None (although any other column works too).
    new_data = new_data[np.array([r[0] is not None for r in new_data])]
    dataset_io.append_to_file(
        pd.DataFrame(data=new_data, columns=df.columns), destpath)
    print("\nWritten to {0}".format(destpath))

