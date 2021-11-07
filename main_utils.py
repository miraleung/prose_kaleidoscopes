# Utils.
import os

__author__ = "Mira Leung"

def should_continue(in_filepath, out_filepath):

  should_continue = in_filepath != out_filepath
  if should_continue:
    return True

  while not should_continue:
    should_continue = input(
        "Are you sure you want to overwrite the original dataset? [y/n] > ")
    if should_continue.lower() == "n":
      return False

    should_continue = should_continue.lower() == "y"
    print("Overwriting {0}".format(out_filepath))
    return True

  file_exists = os.path.isfile(out_filepath)
  if not file_exists:
    return True

  should_continue = False
  while not should_continue:
    should_continue = input(
        "{0} exists, are you sure you want to overwrite it? [y/n] > ".format(out_filepath))
    if should_continue.lower() == "n":
      return False

    should_continue = should_continue.lower() == "y"
    print("Overwriting {0}".format(out_filepath))
    return True

