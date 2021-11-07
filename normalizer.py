import inflect
import html
import pkg_resources
import re

from symspellpy import SymSpell, Verbosity

__author__ = "Anonymized Author of Prose Kaleidoscopes"

class Normalizer:
  # as per recommendation from @freylis, compile once only
  REGEX_DIGIT_CAPTURE = re.compile("(\d+)")
  REGEX_DOTS = re.compile("\.\.+")
  REGEX_HTML_TAG = re.compile('<.*?>')
  REGEX_LOWERWORD_CAPTURE = re.compile("\s([a-z]+)")
  REGEX_PUNC_LOWER_NO_SPACE = re.compile("([,;])([a-z])")
  REGEX_PUNC_UPPER_NO_SPACE = re.compile("([\.\!\?])([A-Z])")
  REGEX_UNK = re.compile("<(unk|pad)>\s")
  NUM_TO_WORD_ENGINE = inflect.engine()

  # This is actually the max number handled by inflct: nine hundred and
  # ninety-nine decillion, ...
  MAX_INFLECT_NUM = 999999999999999999999999999999999999

  def __init__(self):
    dictionary_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt")
    self.spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
    self.spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

  def remove_html_tags(self, raw_html):
    return re.sub(self.REGEX_HTML_TAG, '', raw_html)

  def normalize_html(self, text):
    return html.unescape(text)

  def convert_numbers_to_words(self, text):
    return re.sub(self.REGEX_DIGIT_CAPTURE,
        lambda x : self.NUM_TO_WORD_ENGINE.number_to_words(int(x.group(0))) if int(x.group(0)) < self.MAX_INFLECT_NUM else x.group(0),
        text)

  def fix_unicode(self, text):
    return text.encode().decode("utf-8")

  def remove_unks(self, text):
    return re.sub(self.REGEX_UNK, "", text)

  def fix_spelling(self, text):
    # Address only lower-case words, to avoid fixing names.
    return re.sub(self.REGEX_LOWERWORD_CAPTURE,
        lambda x : " " + self.spell.lookup(x.group(0),
                                           Verbosity.CLOSEST,
                                           max_edit_distance=2,
                                           include_unknown=True,
                                           transfer_casing=True)[0]._term,
        text)

  def fix_spacing(self, text):
    # Add any missing spaces between periods and text.
    result = re.sub(self.REGEX_PUNC_LOWER_NO_SPACE, r"\1 \2", text)
    result = re.sub(self.REGEX_PUNC_UPPER_NO_SPACE, r"\1 \2", result)
    result = ' '.join(result.split()).strip()
    return result.rstrip()

  def fix_ellipses(self, text):
    # Extra space will be cleaned up by the space fixer.
    # While the order dependency cuts across methods and is suboptimal, the
    # optimal approach of fixing the spacing here would incur a performance
    # hit when applied to all data points.
    return re.sub(self.REGEX_DOTS, '. ', text)

  def normalize(self, text):
    result = text

    result = self.remove_unks(result)
    result = self.fix_unicode(result)
    result = self.normalize_html(self.remove_html_tags(result))
    result = self.fix_spelling(result)
    result = self.convert_numbers_to_words(result)
    result = self.fix_ellipses(result)
    # Must go last.
    result = self.fix_spacing(result)

    return result

