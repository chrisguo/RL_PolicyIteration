import random
import numpy as np

"""
    Module containing class for the easy 21 environment

"""
class CardGame():
    """
    Contains methods for drawing cards and stepping in an easy 21 game, as well as fields for recording
    a history game steps and attributes for a game instance.
    One game relates to one episode in our RL problem.
    """

    def __init__(self, dealer_limit = 17, bust_limits = (0, 22)):
        self.dealer_limit = dealer_limit
        self.bust_limits = bust_limits
        self.sa_rewards = {}
        self.state_actions = []
        self.cards = []
        self.terminal = (0,0)

    def draw(self, blackProb, who = 'player', replace = True):
        """
        Function to simulate drawing a card in Easy 21
        :param blackProb: probability of drawing a black card
        :param who: who the card is dealt to
        :param replace: whether to replace card back in deck after drawing
        :return: the card colour and card numer
        """

        card_col = np.random.choice(a=[1,-1], replace = replace, p=[blackProb, 1-blackProb])
        card_num = random.randint(1,10)
        self.cards.append((card_col, card_num, who))

        return card_col, card_num

    def step(self, state, player_action, **args):
        """
        Method for simulating a turn in an easy21 game where a player is taking action
        :param state: current state of game
        :param player_action: action the player makes, 0 or 1 representing hit or stick
        :param args: any key word arguments required by the draw function
        :return: next game state and reward. Reward encoded -1, 0 , 1. Terminal state encoded (0,0)
        """

        dealerSum = state[0]
        playerSum = state[1]
        self.state_actions.append((state,player_action))

        # ERROR CHECK:if incoming state already results in player being bust before action taken...

        if not (self.bust_limits[0] < playerSum < self.bust_limits[1]):
            next_state = self.terminal
            reward = -1

         # update to player sum if player hits
        elif player_action == 0:
            col, num = self.draw(**args)
            playerSum = playerSum + num*col # new player sum after hit

            # check if player is bust after the hit
            if not (self.bust_limits[0] < playerSum < self.bust_limits[1]):
                next_state = self.terminal
                reward = -1

            else:
                next_state = (state[0], playerSum)
                reward = 0 # all intermediary rewards are zero

        # if player sticks, the dealer does their thing
        else:

            while self.bust_limits[0] < dealerSum <= self.dealer_limit:
                # dealer always hits if below 17, sticks otherwise
                col, num = self.draw(who = 'dealer', **args)
                dealerSum = dealerSum + num*col #

            next_state = self.terminal

             # check of dealer is bust
            if not(self.bust_limits[0] < dealerSum < self.bust_limits[1]):
                reward = 1

            # check other outcomes and record rewards
            elif dealerSum == playerSum:
                reward = 0

            elif dealerSum > playerSum:
                reward = -1

            elif dealerSum < playerSum:
                reward = 1

        if next_state == self.terminal:
            for p in self.state_actions:
                self.sa_rewards[p] = reward

        return next_state, reward









