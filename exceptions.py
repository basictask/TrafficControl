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


class TooManyPathsError(Exception):
    """
    In case there are too many paths needed to be generated in the matrix assembler
    """
    pass


class IllegalEntryPointError(Exception):
    """
    In case there's an antry point given which is not present in the graph
    """
    pass


class IllegalNodeConfigurationError(Exception):
    """
    In case some nodes are placed too close to eachother
    """
    pass


class IllegalActionException(Exception):
    """
    In case the agent has chosen an action that is not defined
    """
    pass


class NoneTypeAttributeError(Exception):
    """
    In case an object has an attribute that is set to None but that's not allowed
    """
    pass
