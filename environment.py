"""
 ▄▄▄▄▄▄▄ ▄▄    ▄ ▄▄   ▄▄ ▄▄▄ ▄▄▄▄▄▄   ▄▄▄▄▄▄▄ ▄▄    ▄ ▄▄   ▄▄ ▄▄▄▄▄▄▄ ▄▄    ▄ ▄▄▄▄▄▄▄
█       █  █  █ █  █ █  █   █   ▄  █ █       █  █  █ █  █▄█  █       █  █  █ █       █
█    ▄▄▄█   █▄█ █  █▄█  █   █  █ █ █ █   ▄   █   █▄█ █       █    ▄▄▄█   █▄█ █▄     ▄█
█   █▄▄▄█       █       █   █   █▄▄█▄█  █ █  █       █       █   █▄▄▄█       █ █   █
█    ▄▄▄█  ▄    █       █   █    ▄▄  █  █▄█  █  ▄    █       █    ▄▄▄█  ▄    █ █   █
█   █▄▄▄█ █ █   ██     ██   █   █  █ █       █ █ █   █ ██▄██ █   █▄▄▄█ █ █   █ █   █
█▄▄▄▄▄▄▄█▄█  █▄▄█ █▄▄▄█ █▄▄▄█▄▄▄█  █▄█▄▄▄▄▄▄▄█▄█  █▄▄█▄█   █▄█▄▄▄▄▄▄▄█▄█  █▄▄█ █▄▄▄█
This is the environment that handles actions, state and rewards.
The environment is explicitly meant to be used by the agent
"""
from city_constructor import Reader
import configparser
args = configparser.ConfigParser()
args.read('config.ini')


class Environment:
    def __init__(self):
        """
        The environment reads all the parameters from a configuration file.
        """
        self.vrate = args['reader'].getint('vrate')
        self.n_steps = args['reader'].getint('n_steps')
        self.show_win = args['reader'].getboolean('show_win')
        self.paths_to_gen = args['reader'].getint('paths_to_gen')
        self.steps_per_update = args['reader'].getint('steps_per_update')
        self.path_dist = args['reader'].get('path_dist')
        self.filepath = args['reader'].get('filepath')
        self.entry_points = list(args['reader'].get('entry_points'))
        self.radius = args['reader'].getint('radius')


# if __name__ == '__main__':
#     env = Environment()
#     i = 1
