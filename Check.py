
from Setup import *

"""
    Script containing methods and executable required by Q1 of the Easy 21 assignment
"""

def check_draw(its, blackProb =0.667):
    """
    Method for testing the environments draw method
    :param its: number of iterations to check
    :param blackProb: probability of the drawn card being black

    :return:checkdraw: frequencies of cards observed, stored to checkDraw.txt file also
    """
    checkdraw = defaultdict(int)
    env = CardGame() # instantiate environment

    for i in range(its): # loop through iterations and record frequency of recorded card
        card_col, card_num = env.draw(blackProb) # call the environment to draw a card
        checkdraw[(card_num, card_col)] +=1  # accumulate counts

    # store results to file
    freq= open('output/checkDraw.txt', 'w')
    freq.write('value color frequency \n')
    key = checkdraw.keys()
    for k in key:
        freq.write('% i %i %f\n' % (k[0], k[1], np.true_divide(checkdraw[k], its)))
    print 'Check draw frequency results file created with %i rows' % len(key)
    freq.close()

    return checkdraw

def check_step(its, start_state, start_action, blackProb =0.667):
    """
    Method for testing the environments step method
    :param its: number of iterations to check
    :param blackProb: probability of the drawn card being black
    :param start_action: action to begin episode from
    :param start_state: state to begin episode from

    :return:checkstep: frequencies of cards observed, stored to checkDraw.txt file also
    """

    checkstep= defaultdict(int)
    env = CardGame() # instantiate environment

    for i in range(its):
        next_state, reward = env.step(state = start_state, player_action=start_action,
                                      blackProb=blackProb)  # call the environment to step in the episode
        checkstep[next_state[0], next_state[1], reward]+=1  # accumulate counts

    # store results to file
    filename = 'output/DBLcheckDrawDealer%iPlayer%iAction%i.txt' % (start_state[0], start_state[1], start_action)
    freq= open(filename, 'w')
    freq.write('dealerCard playerSum reward frequency \n')
    key = checkstep.keys()
    for k in key:
        freq.write('% i %i %i %f\n' % (k[0], k[1],k[2],  np.true_divide(checkstep[k], its)))
    print 'Check step frequency results file %s \n created with %i rows' % (filename,len(key))
    freq.close()

    return checkstep


if __name__ == '__main__':

    """"
    Execute Q1
    """

    its = 10000
    checkDraw = check_draw(its)

    # specify test cases for step check as specified in the assignment
    startStates = [(4,21,1), (1,10,0), (1,18,1), (10,15,1)]

    # loop through test cases and produce output
    for i in startStates:
        checkStep = check_step(its, i[0:2], i[2])
