
from Setup import *
from MonteCarlo import epsilon_greedy

"""
    Script containing methods and executable required by Q3 of the Easy 21 assignment
    Sarsa(lambda)

"""

@func_timer
def runSARSA(its, Q_sa, Q_MC_sa, policy, expl_const = 10,  lam=1):
    """
    Monte Carlo Control Algorithm
    :param its: number of iterations
    :param Q_sa: initial action value dict (set to zero)
    :param Q_MC_sa: optimal policy produced by Monte Carlo Control
    :param policy: initial arbitrary policy dict
    :param expl_const: exploration constant for greedy policy updates
    :param lam: trace hyper parameter

    :return:
           mse: incremental mean squared error results (e.g. every 10000 episodes) for the lambda value
           Q_sa: Final action value function
           policy: Final policy
    """

    sa_pairs = Q_sa.keys()
    sa_counts = dict(zip(sa_pairs, np.zeros(len(sa_pairs))))
    mse = []


    for i in range(0, its):

        E_sa = dict(zip(sa_pairs, np.zeros(len(sa_pairs))))
        # initialise state and action for episode
        current_state = (random.randint(1,10), random.randint(1,10)) # exploratory starts
        current_action = policy[current_state]
        env=CardGame()

        # begin episode
        while current_state <> (0,0): # step until terminal state produced, which results in reward

            new_state, reward = env.step(state = current_state, player_action=current_action,
                                        blackProb=(0.667)) # step in environment

            # update counts and get next action by policy
            sa_counts[(current_state, current_action)] += 1
            next_action  = policy[new_state]

            #incorporate elg trace using lambda
            delta = reward + Q_sa[(new_state, next_action)]- Q_sa[(current_state, current_action)]
            E_sa[(current_state, current_action)] +=1 # accumulating trace

            # update action value functin and trace each step. Bootstrapping
            Q_sa.update({k : (Q_sa[k] + np.true_divide(1,sa_counts[k])*(delta*E_sa[k])) for k in env.state_actions})
            E_sa.update({k : (lam*E_sa[k]) for k in env.state_actions})

            current_action = next_action
            current_state = new_state

        # update policy greedily
        policy = epsilon_greedy(env, expl_const, sa_counts, Q_sa, policy)

        # record MSE incrementally
        mse_check = divmod(i+1,num_its/100)
        if mse_check[1] ==0:
            # compute mse
            dif = [(Q_sa[k] - Q_MC_sa[k]) for k in Q_sa.iterkeys()]
            agg = np.sum(np.power(dif,2))
            mse.append(np.true_divide(1,len(Q_sa))*(agg))

            print 'Epoch %d, Lambda %.2f, MSE %.10f' % (i+1,lam, mse[mse_check[0]-1])

    return Q_sa, policy, mse

# ------------------------------EXECUTABLE------------------------------------------------

if __name__ == '__main__':

    """"
    Execute Q3
    """

    num_its  = 1000000
    actions = (0,1)
    # placeholders for mse results
    mse_lam = []
    mse_lam_learn = []

    # load optimal action value function provided by monte carlo control for use in MSE
    with open('output/Q_sa_MC') as f:
         Q_MC_sa = pickle.load(f)

    # create range of lambda values
    lam = np.true_divide(range(0,11, 1),10)

    # loop over lambda, run Sarsa (lambda) and accumulate results
    for l in lam:
        init_Q_sa, init_policy = dict_init(state_dim = (10,21), actions = (0,1), init_ps = 10)
        Q_sa_opt, opt_policy, mse = runSARSA(its=num_its,Q_sa=init_Q_sa, Q_MC_sa=Q_MC_sa,
                                            policy=init_policy, lam=l)
        mse_lam_learn.append(mse)
        mse_lam.append(mse[-1])

    # store granular mse results if required for subsequent plots or inspection
    with open('output/mse_lam_learn', 'wb') as g:
        pickle.dump(mse_lam_learn, g)

    # produce MSE Vs lambda and MSE learning curve plots
    plotMSE(num_its, lam, mse_lam, mse_lam_learn, 'Sarsa')





