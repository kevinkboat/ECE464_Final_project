import random
from typing import Tuple

import numpy as np

""" Global variables """

# Game settings
NUM_ATTRIBUTES = int(5)  # number of unique agent attributes
GRID_SIZE = int(10)  # game grid size
ENABLED_DIRECT_COMMUNICATION = False  # allow direct agent-agent communication
AGENT_MESSAGING_COOL_DOWN_PERIOD = int(5)  # cool down time for agent messaging to prevent blocking

# Probabilities
PROB_AGENT_DO_NOTHING = 0.2  # probability that agents doesn't move when it has the chance to
PROB_ENEMY_DETECTS_HUB = 0.2
BOX_DROP_PERIOD = 7
INIT_PCT_BOXES = 0.10

LIMIT_MOVES_PER_GAME = 10000

# Instrumenting variables
count_boxes_redeemed = 0
count_interactions_agent_hub = 0
count_interactions_agent_agent = 0
count_interactions_agent_enemy = 0
count_attempted_hub_breaches = 0
count_successful_hub_breaches = 0

""" Object Classes """

# Variables below are used keep track of in game objects
hubs_in_game = list()  # hubs in the game
agents_in_game = list()  # agents in the game
enemies_in_game = list()  # enemies in game
boxes_in_game = dict()  # boxes in game
player_map = dict()  # keeps track of spaces occupied by agents, bad actors, hub


class GameBox(object):
    """An interactable in the game environment that affects agent learning"""

    def __init__(self, row: int, col: int, effect: int = 1):
        """Game Box constructor"""
        self.row, self.col, self.effect = row, col, effect
        self.attr_id = int(random.uniform(0, NUM_ATTRIBUTES - 1))
        boxes_in_game[(row, col)] = self


class Hub(object):
    """Coordinator of Knowledge sharing"""

    def __init__(self, row: int, col: int, attr: np.ndarray = None):
        """Hub constructor"""
        """
        <Attribute>: <type>     # <description>
        row: int, col: int      # position of hub
        attr: ndarray           # stored attributes 
        connected_agents: set   # set of agents connected to this hub
        unique_id: int          # id
        message_handle: str     # message handle
        broadcast_period: int   # period of hub broadcasts
        """

        if attr is None:
            # assign default attribute set
            attr = np.zeros(NUM_ATTRIBUTES)

        if validate_coordinate(row, col):  # must be in grid
            if not player_map.__contains__((row, col)):  # space must be unoccupied
                if len(attr) >= NUM_ATTRIBUTES:  # list must be long enough to store all attributes

                    # from this point, all arguments are validated
                    self.row, self.col, = row, col
                    player_map[(row, col)] = self
                    self.attr = (attr[0:NUM_ATTRIBUTES]).copy()
                    self.connected_agents = list()  # set of agents connecting to this hub
                    self.unique_id = len(hubs_in_game)
                    hubs_in_game.append(self)
                    self.message_handle = "hub_" + str(self.unique_id)
                    self.broadcast_period = 15
                    return
                else:
                    print("Error: Required list with ", NUM_ATTRIBUTES, " attributes, but only gave ", len(attr))
            else:
                print("Error: Spot at (row, col) = (", row, ", ", col, ") is occupied")

        del self

    def perform_action(self):
        """Update knowledge of connected agents. Hub attribute is updated to reflect
            group knowledge"""
        global count_interactions_agent_hub
        # Consolidate group knowledge
        for agent in self.connected_agents:
            agent.is_busy = True  # block agent activity
            self.attr = np.maximum(self.attr, agent.attr)
        # self.attr now contains group knowledge

        # Broadcast group knowledge
        for agent in self.connected_agents:
            agent.attr = self.attr.copy()

        count_interactions_agent_hub += 1

    def connect_agents(self, agents: list):
        """Establish connection between this hub and agents in the list"""
        for agent in agents:
            if type(agent) is Agent:
                agent.is_busy = True
                agent.connected_hub = self
                self.connected_agents.append(agent)
            else:
                print("Error: Tried to connect non-agent to a hub")


class Agent(object):

    def __init__(self, row: int, col: int, attr: np.ndarray = None, hub: Hub = None):
        """Agent constructor"""
        """
        <Attribute>: <type>     # <description>
        row: int, col: int      # position of agent
        attr: ndarray           # stored attributes
        connected_hub: Hub      # the hub that this agent connects to
        unique_id: int          # id
        is_busy: bool           # preoccupied with higher priority task
        messaging_cool_down_counter: int    # cool down counter to prevent blocking after agent messaging
        message_handle: str     # message handle
        """

        if attr is None:
            # assign default attribute set
            attr = np.zeros(NUM_ATTRIBUTES)

        if validate_coordinate(row, col):  # must be in grid
            if not player_map.__contains__((row, col)):  # space must be unoccupied
                if len(attr) >= NUM_ATTRIBUTES:  # list must be long enough to store all attributes

                    # from this point, all arguments are validated
                    self.row, self.col, = row, col
                    player_map[(row, col)] = self
                    self.attr = (attr[0:NUM_ATTRIBUTES]).copy()
                    self.unique_id = len(agents_in_game)
                    agents_in_game.append(self)
                    self.connected_hub = hub
                    self.is_busy = False
                    self.messaging_cool_down_counter = AGENT_MESSAGING_COOL_DOWN_PERIOD
                    self.message_handle = "agent_" + str(self.unique_id)
                    return
                else:
                    print("Error: Required list with ", NUM_ATTRIBUTES, " attributes, but only gave ", len(attr))
            else:
                print("Error: Spot at (row, col) = (", row, ", ", col, ") is occupied")

        del self

    def perform_action(self):
        """Agent interacts with its environment (counts as a move)"""
        global count_boxes_redeemed, count_interactions_agent_agent
        if not self.is_busy:
            self.is_busy = True

            # update messaging cool down counter
            if self.messaging_cool_down_counter < AGENT_MESSAGING_COOL_DOWN_PERIOD:
                self.messaging_cool_down_counter += 1

            pos = (self.row, self.col)
            empty_spots, nearby_agents = scan_nearby(self.row, self.col)

            if len(nearby_agents) > 0 and (self.messaging_cool_down_counter == AGENT_MESSAGING_COOL_DOWN_PERIOD) \
                    and ENABLED_DIRECT_COMMUNICATION:
                # choose agent to share knowledge with
                sharing_buddy = random.choice(nearby_agents)

                if sharing_buddy.messaging_cool_down_counter == AGENT_MESSAGING_COOL_DOWN_PERIOD \
                        and not sharing_buddy.is_busy:
                    sharing_buddy.is_busy = True  # block other agent

                    # share knowledge
                    self.attr = np.maximum(self.attr, sharing_buddy.attr)
                    sharing_buddy.attr = self.attr.copy()

                    # start cool down timers for both agents
                    sharing_buddy.messaging_cool_down_counter = -1
                    self.messaging_cool_down_counter = -1

                    # print(self.message_handle, ": Shared knowledge with @", sharing_buddy.message_handle)
                    count_interactions_agent_agent += 1

            elif boxes_in_game.__contains__(pos):
                # realize effect and delete game box
                box = boxes_in_game[pos]
                self.attr[box.attr_id] += box.effect
                # print(self.message_handle, ": Redeemed box_", box.attr_id, "@", pos)
                count_boxes_redeemed += 1
                del boxes_in_game[pos]

            elif len(empty_spots) > 0 and (random.random() >= PROB_AGENT_DO_NOTHING):
                # move to new spot
                old_pos = (self.row, self.col)
                new_pos = random.choice(empty_spots)
                self.row, self.col = new_pos[0], new_pos[1]
                player_map[new_pos] = self
                # update player map
                del player_map[old_pos]

            else:
                # do nothing
                pass


class Enemy(object):

    def __init__(self, row: int, col: int):
        """Enemy constructor"""
        """
        <Attribute>: <type>     # <description>
        row: int, col: int      # position of agent
        unique_id: int          # id
        message_handle: str     # message handle
        """

        if validate_coordinate(row, col):  # must be in grid
            if not player_map.__contains__((row, col)):  # space must be unoccupied
                # from this point, all arguments are validated
                self.row, self.col, = row, col
                player_map[(row, col)] = self
                self.unique_id = len(enemies_in_game)
                enemies_in_game.append(self)
                self.message_handle = "enemy_" + str(self.unique_id)
                return
            else:
                print("Error: Spot at (row, col) = (", row, ", ", col, ") is occupied")

        del self

    def perform_action(self) -> bool:
        # return value answers: Is game over?

        global count_interactions_agent_enemy, count_attempted_hub_breaches, count_successful_hub_breaches
        empty_spots, nearby_targets = scan_nearby(self.row, self.col, pass_hubs=True)

        if len(nearby_targets) > 0:
            target = random.choice(nearby_targets)
            if type(target) is Hub:
                count_attempted_hub_breaches += 1
                if random.random() < PROB_ENEMY_DETECTS_HUB:
                    print("Game over")
                    count_successful_hub_breaches += 1
                    return True
                elif len(empty_spots) > 0:
                    # move to new spot
                    old_pos = (self.row, self.col)
                    new_pos = random.choice(empty_spots)
                    self.row, self.col = new_pos[0], new_pos[1]
                    player_map[new_pos] = self

                    # update player map
                    del player_map[old_pos]
            elif type(target) is Agent:
                # kill agent
                if target.connected_hub is not None and target.connected_hub.connected_agents.__contains__(target):
                    # disconnect agent from hub
                    target.connected_hub.connected_agents.remove(target)
                del player_map[(target.row, target.col)]
                agents_in_game.remove(target)
                count_interactions_agent_enemy += 1
        elif len(empty_spots) > 0:
            # move to new spot
            old_pos = (self.row, self.col)
            new_pos = random.choice(empty_spots)
            self.row, self.col = new_pos[0], new_pos[1]
            player_map[new_pos] = self

            # update player map
            del player_map[old_pos]

        return False


""" Helper Functions """


def validate_coordinate(row: int, col: int, suppress_err_message: bool = False) -> bool:
    """Returns True if (row, col) falls within grid, else returns False"""
    is_valid = (row >= 0) and (row < GRID_SIZE) and (col >= 0) and (col < GRID_SIZE)
    if not is_valid and not suppress_err_message:
        print("Error: Invalid coordinate given for grid of size ", GRID_SIZE)

    return is_valid


def scan_nearby(row: int, col: int, step: int = 1, agents_only: bool = True, pass_hubs: bool = False) \
        -> Tuple[list, list]:
    """:returns list of unoccupied spots and players nearby"""
    empty_spots = []
    nearby_players = []

    for i in range(row - step, row + step + 1):
        if (i >= 0) and (i < GRID_SIZE):
            # no row wraparounds
            for j in range(col - step, col + step + 1):
                if ((j >= 0) and (j < GRID_SIZE)) and (i != row or j != col):
                    # no column wraparounds and ignore current position
                    pos = (i, j)
                    if player_map.__contains__(pos):
                        player = player_map[pos]
                        if (agents_only and type(player) is Agent) or (pass_hubs and type(player) is Hub):
                            nearby_players.append(player)
                    else:
                        # add spot to the list of free spaces
                        empty_spots.append(pos)

    return empty_spots, nearby_players


def main():
    stats = []
    num_iterations = 100
    for i in range(num_iterations):
        print("Iteration:", i)

        print("Setting up game players")
        center = int(GRID_SIZE / 2)
        hub1 = Hub(center, center)
        # hub1 = None
        Agent(center + 1, center, hub=hub1)
        Agent(center - 1, center, hub=hub1)
        Agent(center, center + 1, hub=hub1)
        Agent(center, center - 1, hub=hub1)
        Enemy(0, 0)
        Enemy(0, 1)

        empty_spots, _ = scan_nearby(center, center, step=int(GRID_SIZE / 2) - 1, agents_only=True)
        if len(empty_spots) > 0:
            init_num_boxes = min([int(INIT_PCT_BOXES * (GRID_SIZE * GRID_SIZE)), len(empty_spots)])
            for pos in random.sample(empty_spots, k=init_num_boxes):
                GameBox(pos[0], pos[1])

        print("Starting game")
        game_over = False
        t = 0
        while t < LIMIT_MOVES_PER_GAME and not game_over:

            if t % BOX_DROP_PERIOD == 0:
                # drop box
                empty_spots, _ = scan_nearby(center, center, step=int(GRID_SIZE / 2) - 1, agents_only=True)
                if len(empty_spots) > 0:
                    pos = random.choice(empty_spots)
                    GameBox(pos[0], pos[1])

            for hub in hubs_in_game:
                if t % hub.broadcast_period == 0:
                    hub.perform_action()

            for enemy in random.sample(enemies_in_game, k=len(enemies_in_game)):
                game_over |= enemy.perform_action()

            if not game_over:
                for agent in random.sample(agents_in_game, k=len(agents_in_game)):
                    agent.perform_action()

                    # Reset agent blocking status
                for agent in agents_in_game:
                    agent.is_busy = False

            t += 1

        stats = [count_boxes_redeemed, count_interactions_agent_hub, count_interactions_agent_agent,
                 count_interactions_agent_enemy, count_attempted_hub_breaches, count_successful_hub_breaches]

        hubs_in_game.clear()
        agents_in_game.clear()
        enemies_in_game.clear()
        boxes_in_game.clear()
        player_map.clear()

    print(stats)
    stats = np.array(stats) / num_iterations
    print(stats)


if __name__ == '__main__':
    main()
