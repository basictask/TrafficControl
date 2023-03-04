"""
Supplementary methods for the Reader class and others
These methods don't belong to any specific class operation and may be used across the entire project
@author: daniel kuknyo
"""


def letter_to_number(letter: str) -> int:
    """
    Converts a letter to a number in GeoGebra representation
    GeoGebra names points as [A, B, C, ..., Z, A1, B1, C1, ..., Z1, A2, B2, C2, ...]
    :param letter: the The letter to be converted to a number
    :return: the numeric representation of the letter starting with 0
    """
    if len(letter) == 1:  # A, B, C
        return ord(letter) - ord('A')
    else:  # A1, B1, C1
        return int(letter[1:]) * 26 + ord(letter[0]) - ord('A')


def drop_empty_keys(dct: dict) -> dict:
    """
    Drops the empty keys from a given dict and returns the dict itself
    :param dct: The dict to drop from
    :return: dict: The dict with the empty keys removed
    """
    for v in list(dct.keys()):
        if len(dct[v]) == 0:
            dct.pop(v, None)
    return dct
