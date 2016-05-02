__author__ = 'Peadar'
from Setup import *

# ------------------------------MONTE CARLO--------------------
@func_timer
def runFcnApprox(its, Q_approx, Q_MC_sa, policy,course_fts,
                 lam=1, e = 0.1, alpha = 0.01):
    """
    Linear function approximation algorithm
    :param its: number of iterations
    :param Q_approx: initial action value dict (set to zero)
    :param policy: initial arbitrary policy dict
    :param course_fts: List of course coded features partitioning input space
    :param e: exploration constant for greedy policy updates
    :param alpha: scalar learning rate

    :return:
           mse: incremental mean squared error results (e.g. every 10000 episodes) for the lambda value
           Q_approx: Final action value function
           policy: Final policy
    """

    W = np.zeros(len(course_fts)) # initialise weight vector for linear model
    mse = []

    for i in range(0, num_its):

         # record MSE incrementally
        mse_check = divmod(i+1,num_its/100)
        if mse_check[1] ==0:
            # compute mse
            dif = [(Q_approx[k] - Q_MC_sa[k]) for k in Q_approx.iterkeys()]
            agg = np.sum(np.power(dif,2))
            mse.append(np.true_divide(1,len(Q_approx))*(agg))
            print 'Epoch %d, Lambda %.2f, MSE %.10f' % (i+1,lam, mse[mse_check[0]-1])

        # reset traces to zero - using accumulating traces
        z = np.zeros(len(course_fts))
        # initialise state and action for episode
        current_state = (random.randint(1,10), random.randint(1,10)) #exploratory starts
        current_action = policy[current_state]

        # get indexes of initial present features
        current_features = featureFilter(current_state, current_action, course_fts)
        env=CardGame()   # instantiate environment

        # begin episode
        while current_state <> (0,0):

            z[current_features]+=1 # accumulating trace for indexes of present features

            new_state, reward = env.step(state = current_state, player_action=current_action,
                                        blackProb=(0.667)) # step in environment

            delta = reward - np.sum(W[current_features]) # used for weight updates

            if new_state <> (0,0):
                explore = np.random.choice(a=[True, False], p=[1-e, e]) # chance to explore options

                if explore == True:
                    for i in (0,1):
                        # greedily select nextr action and update action value function based on present features
                        current_features = featureFilter(new_state, i, course_fts)
                        Q_approx[(new_state,i)] = np.sum(W[current_features]) # linear approximation to action val fcn

                    if Q_approx[(new_state,0)] >= Q_approx[(new_state,1)]:
                        current_action = 0
                    else:
                        current_action = 1
                    current_features = featureFilter(new_state, current_action, course_fts)
                else:
                    current_action = random.randint(0,1)
                    current_features = featureFilter(new_state, current_action, course_fts)
                    Q_approx[(new_state, current_action)] = np.sum(W[current_features]) # linear approximation to action val fcn

                delta = delta + Q_approx[(new_state, current_action)]

            W = W + alpha*delta*z  # update weights based on trace for present features
            z = lam*z # increment trace with lambda
            current_state = new_state # reset state

        # update policy greedily given update action value function
        for p in env.sa_rewards:
            p = p[0]
            rand_action = random.randint(0,1)
            if Q_approx[(p,0)] > Q_approx[(p,1)]:
                greedy_action = 0
            else:
                greedy_action = 1

            policy[p] = np.random.choice(a=[rand_action, greedy_action], p=[e, 1-e])

    return Q_approx, policy, mse


def featureFilter(state, action, features):
    """

    :param state: current state
    :param action:  current action
    :param features: set of course features
    :return: F_a indices of present features
    """
    counter = 0
    F_a = []
    # consider all features
    for (ft_dealer, ft_player, ft_action) in features:

         # if present, record index using counter
        if (ft_dealer[0]<= state[0] <= ft_dealer[1]) and (ft_player[0]<= state[1] <= ft_player[1]) and action == ft_action:
            F_a.append(counter)
        counter += 1

    return F_a

# ------------------------------EXECUTABLE--------------------

if __name__ == '__main__':

    """
    Script containing methods and executable required by Q4 of the Easy 21 assignment
    Linear function approximation

    """

    num_its  = 1000000
    actions = (0,1)
    # placeholders for mse results
    mse_lam = []
    mse_lam_learn = []

    with open('output/Q_sa_MC') as f:
         Q_MC_sa = pickle.load(f)

    # inputs for how features encode the state space

    code_dealer = [(1,4), (4,7), (7,10)]
    code_player = [(1,6), (4,9), (7,12), (10,15),
                   (13, 18), (16,21)]
    num_features = 3*6*2
    # create list of all features
    CCFA = list(itertools.product(code_dealer, code_player,actions))

    # create range of lambda values
    lam = np.true_divide(range(0,11, 1),10)

    # loop over lambda, run lin fcn approx and accumulate results
    for l in lam:
        init_Q_sa, init_policy = dict_init(state_dim = (10,21), actions = (0,1), init_ps = 10)
        Q_sa_opt, opt_policy, mse = runFcnApprox(its=num_its, Q_approx=init_Q_sa, Q_MC_sa=Q_MC_sa,
                                            policy=init_policy, course_fts =CCFA, lam=l)

        mse_lam_learn.append(mse)
        mse_lam.append(mse[-1])

    # produce MSE Vs lambda and MSE learning curve plots
    plotMSE(num_its,lam, mse_lam, mse_lam_learn, 'LinFuncApprox')
