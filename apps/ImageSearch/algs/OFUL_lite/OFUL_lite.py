"""
* make mapping, targets to features
* work with Zappos dataset
2. generalize to n users, not 1 user
3. choose initial sampling arm
    * myApp.py getQuery/processAnswer help this
    * V, b, theta_hat need to be stored per user
    * add new key to butler.particpants[i]
* make launching easier

## 2016-05-17
### Features
* Download features from internet, assume images have been uploaded (it takes
about 3 hours to do via S3 from the university
* do_not_ask is implemented

### Bottlenecks
* argmax is really slow; it uses a for loop in Python. I'll look into using
    NumPy to speed this up. Extrapolating to 50k features, it will take about
    25 minutes to answer one question

| Trial | App.py:getQuery | myApp.py:getQuery | get_X * 2   | argmax | Total |
| load  | 2.27            | 0.24              | 1.93 (*1)   | 0      | 2.17  |
| q0    | 10.38           | 1.02              | 1.96 + 2.27 | 0      | 5.25  |
| q1    | 9.3             | 1.08              | 1.97 + 1.95 | 4.49   | 9.49  |
| q2    | 14.69           | 1.15              | 3.83 + 1.97 | 7.25   | 14.2  |
(run on 2016-05-18 10:00 on c3.large machine with 2k shoes)

| Trial | App.py:getQuery | myApp.py:getQuery | get_X * 2   | argmax | Total |
| load  | 2.28            | 0.13              | 2.0*1       | 0      | 2.38  |
| q0    | 9.69            | 1.28              | 2.03 + 2.04 | 4.54   | 9.89  |
(run on 2016-05-18 11:00 on r3.large machine with 2k shoes)

| Trial | App.py:getQuery | myApp.py:getQuery | get_X * 2   | argmax | Total |
| load  | 51.57           | 0.58              | 50.8*1      | -      | 51.38 |
| q0    | 115             | 1.11              | 50.5*1      | 58.7   |       |
| q1    | 116             | 1.24              | 51.1 + 50.8 | 63     |       |
(run on 2016-05-11:30 on r3.large machine with 50k shoes)

inverting V does not take the most time in argmax_reward
time to invert (1000, 1000) matrix = 0.0263409614563

*summary:* from this, we need to (a) optimize getting X and (b) optimize
argmax_reward

After speeding up get_X to on the order of 0.2secs:

| Trial | App.py getQuery |
| load | 0.36 secs |
| q0 | not written down|
| q1 | get_X: 0.144 + 0.02, time_to_invert: 2.2secs, argmax_reward = 65secs

So most of the time is spent in OFUL:argmax

After speeding up argmax:

| Function            | Time 0   | Time 1 | Time 2 | Time 3 |
| ------------------- | -----    | -----  | ------ |        |
| myApp:processAnswer | 1.00     | 1.1    | 0.4    | 0.33   |
| alg:processAnswer   | 2.35     | 1.2    | 1.50   | 1.14   |
| App:processAnswer   | 3.38     | 2.3    | 1.9    |        |
| myApp:getQuery      | 1.29     | 1.43   | 0.73   | 0.497  |
| alg:getQuery        | 0.49     | 0.39   | 1.56   | 0.45   |
| App:getQuery        | 3.42     | 3.2    | 2.7    |        |
| ------------------- | -------- |        |        |        |
| Total               | 6.8      |        |        |        |

0. speeds up calculating V^{-1}
1. speeds up arg max
2.5 Moves from 2k to 50k shoes?
2. speeds up storing long lists in database
3. merge Kevin's changes from #101 in.

| Task                  | Time |
| load                  | 1.7  |
| choosing initial shoe | 6.5  |
| q1 yes/no             | 3.6  |
| q2 yes/no             | 3.6  |
| q3 yes/no             | 3.8  |

alg:processAnswer bottleneck: PermStore:set_many, MongoDB:update_one
alg:getQuery 0.5s in PermStore:get, MongoDB:find_one
"""

from __future__ import division
import numpy as np
import next.utils as utils
import time

# TODO: change this to 1
reward_coeff = 1.00

def timeit(fn_name=''):
    def timeit_(func, *args, **kwargs):
        def timing(*args, **kwargs):
            start = time.time()
            r = func(*args, **kwargs)
            utils.debug_print('')
            utils.debug_print("function {} took {} seconds".format(fn_name, time.time() - start))
            utils.debug_print('')
            return r
        return timing
    return timeit_


def CalcSqrtBeta(d, t, scale, R, ridge, delta, S_hat=1.0):
    return scale * (R * np.sqrt(d * np.log((1 + t / (ridge * d)) / delta)) + np.sqrt(ridge) * S_hat)


class OFUL_lite:
    def initExp(self, butler, params=None, n=None, R=None, ridge=None,
                failure_probability=None):
        """
        initialize the experiment

        (int) n : number of arms
        (float) R : sub-Gaussian parameter, e.g. E[exp(t*X)]<=exp(t^2 R^2/2),
                    defaults to R=0.5 (satisfies X \in [0,1])
        (float) failure_probability : confidence
                imp note: delta
        (dict) params : algorithm-specific parameters (if none provided in
                        alg_list of init experiment, params=None)

        Expected output (comma separated):
          (boolean) didSucceed : did everything execute correctly
        """
        # setting the target matrix, a description of each target
        # X = np.asarray(params['X'])
        #X = get_feature_vectors()
        X = butler.db.X
        # theta_star = np.asarray(params['theta_star'])
        d = X.shape[1]  # number of dimensions in feature
        n = X.shape[0]

        #lambda_ = ridge
        lambda_ = 1.0
        R = 1.0

        # initial sampling arm
        # theta_hat = X[:, np.random.randint(X.shape[1])]
        # theta_hat = np.random.randn(d)
        # theta_hat /= np.linalg.norm(theta_hat)

        to_save = {'R': R, 'd': d, 'n': n,
                   'lambda_': lambda_,
                   'total_pulls': 0.0,
                   'rewards': [],
                   'max_dist_comp': 1000,
                   'ask_indices': range(n),
                   'arms_pulled': [],
                   'failure_probability': failure_probability}

        for name in to_save:
            butler.algorithms.set(key=name, value=to_save[name])

        return True

    @timeit(fn_name='alg:getQuery')
    def getQuery(self, butler, participant_uid):
        """
        A request to ask which index/arm to pull

        Expected input:
          (list of int) do_not_ask_list : indices in {0,...,n-1} that the
                algorithm must not return. If there does not exist an index
                that is not in do_not_ask_list then any index is acceptable
                (this changes for each participant so they are not asked the
                same question twice)

        Expected output (comma separated):
          (int) target_index : idnex of arm to pull (in 0,n-1)

         particpant_doc is butler.participants corresponding to this
         participant

        if we want, we can find some way to have different arms
        pulled using the butler
        """
        expected_rewards = np.asarray(butler.participants.get(uid=participant_uid, key='expected_rewards'))
        do_not_ask = butler.participants.get(uid=participant_uid, key='do_not_ask')
        utils.debug_print('dna: ', do_not_ask)
        expected_rewards[np.asarray(do_not_ask)] = -np.inf
        i_x = np.argmax(expected_rewards)
        butler.participants.append(uid=participant_uid,
                                   key='do_not_ask', value=i_x)
        return i_x

    @timeit(fn_name='alg:processAnswer')
    def processAnswer(self, butler, target_id=None,
                      target_reward=None, participant_uid=None):
        """
        reporting back the reward of pulling the arm suggested by getQuery

        Expected input:
          (int) target_index : index of arm pulled
          (int) target_reward : reward of arm pulled

        Expected output (comma separated):
          (boolean) didSucceed : did everything execute correctly
        """

        if not target_id:
            participant_doc = butler.participants.get(uid=participant_uid)
            # utils.debug_print('pargs in processAnswer:', participant_doc)
            # X = get_feature_vectors()
            X = butler.db.X
            participant_uid = participant_doc['participant_uid']

            n = X.shape[0]
            d = X.shape[1]
            lambda_ = butler.algorithms.get(key='lambda_')

            utils.debug_print('setting t for first time')
            target_id = butler.participants.get(uid=participant_uid, key='i_hat')
            expected_rewards = X.dot(X[target_id,:])
            expected_rewards[target_id] = -np.inf
            data = {'t': 1,
                    'b': np.zeros(d),
                    'invV': np.eye(d)/lambda_,
                    'x_invVt_norm': np.ones(n)/lambda_,
                    'do_not_ask': [target_id],
                    'expected_rewards': expected_rewards
                    }
            participant_doc.update(data)
            #for key in data.keys():
            #    butler.participants.set(uid=participant_uid, key=key)

            butler.participants.set_many(uid=participant_doc['participant_uid'],
                                         key_value_dict=participant_doc)

            return True

        task_args = {
            'butler': butler,
            'target_id': target_id,
            'target_reward': target_reward,
            'participant_uid': participant_uid
        }


        butler.job('modelUpdate', task_args, ignore_result=True)

        return True

    def modelUpdate(self, butler, target_id, target_reward, participant_uid):
        participant_doc = butler.participants.get(uid=participant_uid)
        X = butler.db.X
        n = X.size[0]
        do_not_ask = participant_doc['do_not_ask']
        max_dist_comp = butler.algorithms.get(key='max_dist_comp')
        sub_inds = np.random.choice(np.setdiff1d(range(n), do_not_ask), max_dist_comp)

        reward = target_reward
        participant_uid = participant_doc['participant_uid']
        i_hat = butler.participants.get(uid=participant_uid, key='i_hat')

        d = X.shape[1]
        lambda_ = butler.algorithms.get(key='lambda_')
        R = butler.algorithms.get(key='R')

        butler.participants.increment(uid=participant_uid, key='t')

        scale = 1.0

        b = np.array(participant_doc['b'], dtype=float)
        invV = np.array(participant_doc['invV'], dtype=float)
        x_invVt_norm = np.array(participant_doc['x_invVt_norm'], dtype=float)

        arm_pulled = X[target_id, :]
        utils.debug_print('size of X:', X.shape)
        utils.debug_print('size of arm_pulled: ', arm_pulled.shape)

        u = invV.dot(arm_pulled)
        utils.debug_print('size of np.dot(X, u):', np.dot(X, u).shape)
        invV -= np.outer(u, u) / (1 + np.inner(arm_pulled, u))

        x_invVt_norm -= np.dot(X, u) ** 2 / (1 + np.inner(arm_pulled, u))

        b += reward * arm_pulled
        theta_hat = X[i_hat, :] + invV.dot(b)

        sqrt_beta = CalcSqrtBeta(d, participant_doc['t'], scale, R, lambda_,
                                 butler.algorithms.get(key='failure_probability'))
        expected_rewards = np.dot(X, theta_hat) + sqrt_beta * np.sqrt(x_invVt_norm)
        expected_rewards[do_not_ask] = -np.inf * np.ones(n)
        expected_rewards[sub_inds] = np.dot(X[sub_inds,:], theta_hat) + sqrt_beta * np.sqrt(x_invVt_norm[sub_inds])

        # save the results
        data = {'x_invVt_norm': x_invVt_norm,
                'b': b,
                'invV': invV,
                'theta_hat': theta_hat,
                'expected_rewards': expected_rewards
                }
        participant_doc.update(data)

        butler.participants.set_many(uid=participant_doc['participant_uid'],
                                     key_value_dict=participant_doc)
        return True

    def getModel(self, butler):
        """
        uses current model to return empirical estimates with uncertainties

        Expected output:
          (list float) mu : list of floats representing the emprirical means
          (list float) prec : list of floats representing the precision values
                              (or standard deviation)
        """
        # TODO: I can't see the results without this
        # (and we also need to change the label name if we want to see results,
        # correct?)
        return 0.5  # mu.tolist(), prec


