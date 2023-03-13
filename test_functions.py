"""
Functions that are used to test the functionality
"""
from city_constructor import Reader
from suppl import *


def test_add_remove(r: Reader, mode: str, infra: str, start: str, end: str) -> None:
    """
    A simple function to test the behavior of the add and remove funcitons
    :param r: Reader object
    :param mode: Operation mode. One of ['add', 'remove']
    :param infra: Infratructure type. One of ['lane', 'road']
    :param start: Starting node of (x, y) coordinates
    :param end: Ending node of (x, y) coordinates
    :return: None
    """
    start = l2n(start)
    end = l2n(end)
    n_lanes = str(r.matrix.loc[start, end])
    result = -1
    if mode == 'add' and infra == 'lane':
        result = r.add_lane(start, end)
    elif mode == 'remove' and infra == 'lane':
        result = r.remove_lane(start, end)
    elif mode == 'add' and infra == 'road':
        result = r.add_road(start, end)
    elif mode == 'remove' and infra == 'road':
        result = r.remove_road(start, end)
    else:
        print(f'Invalid data given: {mode} {infra} {start} {end}')
    if result:
        result = 'successful'
    else:
        result = 'unsuccessful'
    start = n2l(start)
    end = n2l(end)
    print(f'Number of lanes from {start} --> {end} = {n_lanes}')
    print(f'{mode[:-1]}ing {infra} ==> {result}\n')
