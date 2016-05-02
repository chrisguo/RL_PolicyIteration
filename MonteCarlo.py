
from Setup import *

"""
    Script containing methods and executable required by Q2 of the Easy 21 assignment
    Monte Carlo Control

"""

# TODO create another class for general policy update procedure for code duplicated across MCC, Sarsa and Linear Approx


@func_timer
def runMCC(its, Q_sa, policy, expl_const = 10):
    """
    Monte Carlo Control Algorithm
    :param its: number of iterations
    :param Q_sa: initial action value dict (set to zero)
    :param policy: initial arbitrary policy dict
    :param expl_const: exploration constant for greedy policy udpates

    :return:
           V: Final value function
           Q_sa: Final action value function
           policy: Final policy
    """

    sa_pairs = Q_sa.keys()
    sa_counts = dict(zip(sa_pairs, np.zeros(len(sa_pairs))))
    V = {}

    for i in range(0, its):

        current_state = (random.randint(1,10), random.randint(1,10)) # exploratory starts
        current_action = policy[current_state]
        env=CardGame()   # instantiate environment to run MCC against

        # begin episode
        while current_state <> (0,0): # step until terminal state produced, which results in reward

            new_state, reward = env.step(state = current_state, player_action=current_action,
                                        blackProb=(0.667)) # step in environment
            if new_state<>(0,0):
                current_action = policy[new_state]
            current_state = new_state


        # update action_value function
        for q in env.sa_rewards.keys():
            sa_counts[q] += 1
            Q_current = Q_sa[q]
            Q_sa[q] = Q_current + (1/sa_counts[q])*(env.sa_rewards[q]-Q_current)

        # update policy
        policy = epsilon_greedy(env, expl_const, sa_counts, Q_sa, policy)

        # print progress
        sys.stdout.write("\r ------- MCC EPISODE %i -------" % i)
        sys.stdout.flush()

        # Compute final value function
        for s in policy.iterkeys():

            if Q_sa[(s),0]  > Q_sa[(s),1]:
                V[s[0], s[1]] = Q_sa[(s),0]
            else:
                V[s[0], s[1]] = Q_sa[(s),1]

    return V, Q_sa, policy


def epsilon_greedy(env, expl_const, sa_counts, Q_sa, policy):
    """
    Epsilon greedy policy updates
    :param env: history of game
    :param expl_const: exploratory contant
    :param sa_counts: visit rates of states
    :param Q_sa: current action value function
    :param Q_sa: current policy

    :return:
           policy: e-greedy updated policy
    """

    for p in env.sa_rewards: # loop through each state

        p = p[0] # extract state from key
        min_rate = min(sa_counts[(p,0)],sa_counts[(p,1)])
        e = expl_const /(expl_const  + min_rate) # determine epsilon, reducing over time
        rand_action = random.randint(0,1) # determine exploratory action

        if Q_sa[(p,0)] > Q_sa[(p,1)]: # determine greedy action
            greedy_action = 0
        else:
            greedy_action = 1

        # update policy
        policy[p] = np.random.choice(a=[rand_action, greedy_action], p=[e, 1-e])

        return policy

# ------------------------------EXECUTABLE------------------------------------------------

if __name__ == '__main__':

    """"
    Execute Q2
    """

    num_its  = 10000000
    actions = (0,1)

    # initialise dictionaries
    Q_sa, policy = dict_init(state_dim = (10,21), actions = (0,1), init_ps = 0)

    # run monte carlo control for easy 21
    V_opt, Q_sa_opt, opt_policy= runMCC(num_its, Q_sa, policy)

    # store optimal action value function for use in Sarsa and linear approx MSE calcs
    with open('output/Q_sa_MC', 'wb') as g:
        pickle.dump(Q_sa_opt, g)

    # store action value function and policy to file for inspection
    Q = open('output/checkQ.txt', 'w')
    Q.write('dealerCard playerSum action value \n')
    key = Q_sa_opt.keys()
    for k in key:
        Q.write('% i %i %i %f\n' % (k[0][0], k[0][1], k[1], Q_sa_opt[k]))
    print 'Check Q results file created with %i rows' % len(key)
    Q.close()

    P = open('output/checkPolicy.txt', 'w')
    P.write('dealerCard playerSum action \n')
    key = opt_policy.keys()
    for k in key:
        P.write('% i %i %i\n' % (k[0], k[1], opt_policy[k]))
    print 'Check P results file created with %i rows' % len(key)
    P.close()

    # produce 3d plot of learned value function
    plotValueFcn(V_opt)



