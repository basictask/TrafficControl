"""
Definition of exceptions for the Reader class.
"""


class SegmentRemovalError(Exception):
    """
    In case there's a command to remove a segment between nodes where there exists no segment.
    """
    pass


class NotStoredError(Exception):
    """
    In case a data member is not stored or np.nan
    """
    pass


class IllegalMidPointError(Exception):
    """
    In case midpoint is not in the 0...1 range (exclusive)
    """
    pass
