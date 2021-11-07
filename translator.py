import sys

import googletrans

import pandas as pd
import numpy as np

from google.cloud import translate_v2 as google_translate

__author__ = "Anonymized Author of Prose Kaleidoscopes"

class PyGoogleTrans:
  def __init__(self):
    self.TRANSLATOR = googletrans.Translator()

  def _chain(self, text, lang1, lang2):
    result1 = self.TRANSLATOR.translate(text, src="en", dest=lang1)
    if result1.text == text:
      return None
    result2 = self.TRANSLATOR.translate(result1.text, src=lang1, dest=lang2)
    if result2.text == result1.text:
      return None
    result3 = self.TRANSLATOR.translate(result2.text, src=lang2, dest="en")
    if result3.text == result2.text:
      return None
    return result3.text

  def en_de(self, text):
    return self.TRANSLATOR.translate(text, src="en", dest="de")

  def de_en(self, text):
    return self.TRANSLATOR.translate(text, src="de", dest="en")

  def zh_xh(self, text):
    return self._chain(text, "zh-CN", "xh")

  def so_tl(self, text):
    return self._chain(text, "so", "tl")

  def zu_ur(self, text):
    return self._chain(text, "zu", "ur")

  def backtranslate(self, text):
    de_text = self.en_de(text)
    return self.de_en(de_text)

  def augment(self, text):
    return [self.zh_xh(text), self.so_tl(text), self.zu_ur(text)]

class RealGoogleTrans:

  def __init__(self):
    self.TRANSLATOR = google_translate.Client()

  def _chain(self, text, lang1, lang2, debug=False):
    orig_text = text
    result1 = self.TRANSLATOR.translate(
        orig_text, source_language="en", target_language=lang1)

    text1 = result1["translatedText"]
    result2 = self.TRANSLATOR.translate(
        text1, source_language=lang1, target_language=lang2)

    text2 = result2["translatedText"]
    result3 = self.TRANSLATOR.translate(
        text2, source_language=lang2, target_language="en")

    retval = result3["translatedText"]
    if debug:
      print("\t{0}  |----->  {1}  |->  {2}  |----> {3}".format(orig_text, text1, text2, retval))
    return retval

  def en_de(self, text):
    result = self.TRANSLATOR.translate(
        text, source_language="en", target_language="de")
    return result["translatedText"]

  def de_en(self, text):
    result = self.TRANSLATOR.translate(
        text, source_language="de", target_language="en")
    return result["translatedText"]

  def zh_xh(self, text):
    return self._chain(text, "zh-CN", "xh", debug=True)

  def so_tl(self, text):
    return self._chain(text, "so", "tl")

  def zu_ur(self, text):
    return self._chain(text, "zu", "ur")

  def backtranslate(self, text):
    de_text = self.en_de(text)
    return self.de_en(de_text)

  def augment(self, text):
    return [self.zh_xh(text), self.so_tl(text), self.zu_ur(text)]

class Translator:
  def __init__(self, use_actual=False):
    self.GOOGLE_TRANS = RealGoogleTrans() if use_actual else PyGoogleTrans()

  def augment_df(self, df, field_name):
    num_orig_rows = len(df.values)
    text_col_index = df.columns.tolist().index(field_name)
    df_np = df.to_numpy()
    # Add one for the original row, plus some extra room because EDA may
    # return more data than we think.
    new_data = np.ndarray((df_np.shape[0] * 6, df_np.shape[1]),
                           dtype=df_np.dtype)

    curr_index = 0
    is_gtrans_working = True
    for i in range(num_orig_rows):
      if i % 10 == 0:
        sys.stdout.write("===== Augmenting {0} / {1} =====\n".format(i, num_orig_rows))
        sys.stdout.flush()

      curr_text = df_np[i][text_col_index]
      new_data[curr_index] = df_np[i]   # Original data.
      curr_index += 1
      augmented_sents = self.GOOGLE_TRANS.augment(curr_text)

      for aug_sent in set(augmented_sents):
        if aug_sent is None:
          print("ERROR! GoogleTrans stopped working, writing existing data to "
                + "disk. Pick up from:\n\t|{0}|".format(curr_text))
          is_gtrans_working = False
          break
        if aug_sent == curr_text or len(aug_sent) == 0:
          continue

        new_data[curr_index] = df_np[i]
        new_data[curr_index][text_col_index] = aug_sent
        # Accumulate indices instead of using (i * n_aug + j) becausse there may
        # be fewer than n_aug sentences.
        curr_index += 1

      if not is_gtrans_working:
        break

    # Prune any leftover empty rows.
    # Condition: orig_index is not None (although any other column works too).
    new_data = new_data[np.array([r[0] is not None for r in new_data])]
    return pd.DataFrame(data=new_data, columns=df.columns)

class BackTranslator:
  def __init__(self, use_actual=False):
    self.GOOGLE_TRANS = RealGoogleTrans() if use_actual else PyGoogleTrans()

  def augment_df(self, df, field_name):
    num_orig_rows = len(df.values)
    text_col_index = df.columns.tolist().index(field_name)
    df_np = df.to_numpy()
    # Add one for the original row, plus some extra room because EDA may
    # return more data than we think.
    new_data = np.ndarray((df_np.shape[0] * 6, df_np.shape[1]),
                           dtype=df_np.dtype)

    curr_index = 0
    for i in range(num_orig_rows):
      if i % 10 == 0:
        sys.stdout.write("===== Backtranslating {0} / {1} =====\n".format(i, num_orig_rows))
        sys.stdout.flush()

      curr_text = df_np[i][text_col_index]
      new_data[curr_index] = df_np[i]   # Original data.
      curr_index += 1
      backtranslated_text = self.GOOGLE_TRANS.backtranslate(curr_text)

      if len(backtranslated_text) == 0:
        continue

      new_data[curr_index] = df_np[i]
      new_data[curr_index][text_col_index] = backtranslated_text
      # Accumulate indices instead of using (i * n_aug + j) becausse there may
      # be fewer than n_aug sentences.
      curr_index += 1

    # Prune any leftover empty rows.
    # Condition: orig_index is not None (although any other column works too).
    new_data = new_data[np.array([r[0] is not None for r in new_data])]
    return pd.DataFrame(data=new_data, columns=df.columns)

