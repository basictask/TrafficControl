"""
Functions that are used to test the functionality
"""
from suppl import *
from city_constructor import Reader
from environment import Environment


def test_a_r_roads(r: Reader, mode: str, infra: str, start: str, end: str) -> None:
    """
    A simple function to test the behavior of the add and remove funcitons on road-type infrastructure (lanes and roads)
    A lane is defined as a single-directional graph edge going A --> B
    A road is defined as a bidirectional graph edge going A <--> B
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
    start = n2l(start)
    end = n2l(end)
    if result:
        result = 'successful'
    else:
        result = 'unsuccessful'
    print(f'Number of lanes from {start} --> {end} = {n_lanes}')
    print(f'{mode} {infra} ==> {result}\n')


def test_a_r_junct(r: Reader, infra: str, node: str) -> None:
    """
    A simple function to test the behavior of converting a junction to a different type
    A junction can be 1: right-hand, 2: roundabout, 3: traffic light
    :param r: Reader object to do the operation
    :param infra: Infrastructure type. One of ['righthand', 'roundabout', 'trafficlight']
    :param node: Node in string form e.g. 'A':0
    :return: None
    """
    node = l2n(node)
    n_incoming = count_incoming_lanes(r.matrix, node)  # Number of incoming lanes
    result = -1
    if infra == 'righthand':
        result = r.add_righthand(node)
    elif infra == 'roundabout':
        result = r.add_roundabout(node)
    elif infra == 'trafficlight':
        result = r.add_trafficlight(node)
    else:
        print(print(f'Invalid data given: {infra} {node}'))
    node = n2l(node)
    if result:
        result = 'successful'
    else:
        result = 'unsuccessful'
    print(f'Number of lanes incoming to {node} = {n_incoming}')
    print(f'Convert to {infra}: {result}\n')


def play_one_episode(env: Environment, agent, max_t: int) -> list:
    """
    Uses the agent to do some time steps inside the episode
    The process is:
        1. Agent observes state
        2. Agent chooses action
        3. Agent executes action
        4. Environment returns next state and reward
    :return: a list of scores that the agent has received
    """
    state = env.reset()
    scores = []
    score = 0
    eps = 0
    for t in range(max_t):
        start, end, action, _ = agent.act(state, eps)
        next_state, reward, successful = env.step(start, end, action)
        state = next_state
        score += reward
        scores.append(reward)
        print('start: {}, end: {}, action: {}, reward: {}, total: {}'.format(start, end, ACTIONS[action], reward, score))
    return scores
