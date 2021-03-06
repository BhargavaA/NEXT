import time
import numpy.random as ra
import numpy as np
import next.utils as utils
from numpy import linalg


class MyAlg:
    def initExp(self, butler, n, d, failure_probability):
        params = butler.algorithms.get(key='params')
        R = params[u'R']
        ridge = params[u'ridge']
        multiplier = params[u'multiplier']

        butler.algorithms.set(key='n', value=n)
        butler.algorithms.set(key='d', value=d)
        butler.algorithms.set(key='R', value=R)
        butler.algorithms.set(key='ridge', value=ridge)
        butler.algorithms.set(key='multiplier', value=multiplier)
        butler.algorithms.set(key='delta', value=failure_probability)

        if butler.experiment.get(key='plot_data') is None:
            butler.experiment.set(key='plot_data', value=[])

        return True

    def getQuery(self, butler, participant_uid):
        arm_order = butler.participants.get(uid=participant_uid, key='arm_order')
        do_not_ask = butler.participants.get(uid=participant_uid, key='do_not_ask')

        num_return = 16
        counter = 0
        return_arms = []
        for next_arm in arm_order:
            if next_arm not in do_not_ask and next_arm not in return_arms:
                counter += 1
                return_arms.append(next_arm)
                if counter >= num_return:
                    break

        return return_arms

    def processAnswer(self, butler, arm_id, reward, num_responses, init_id, participant_uid):
        if num_responses == 1:
            d = butler.algorithms.get(key='d')
            n = butler.algorithms.get(key='n')
            ridge = butler.algorithms.get(key='ridge')
            invVt = np.eye(d) / ridge
            b = np.zeros(d)
            x_invVt_norm = np.ones(n) / ridge

            butler.participants.set(uid=participant_uid, key='received_rewards', value=[])
            butler.participants.set(uid=participant_uid, key='invVt', value=invVt)
            butler.participants.set(uid=participant_uid, key='b', value=b)
            butler.participants.set(uid=participant_uid, key='x_invVt_norm', value=x_invVt_norm)
            butler.participants.set(uid=participant_uid, key='logDetV', value=d * np.log(ridge))

        butler.participants.append(uid=participant_uid, key='do_not_ask', value=arm_id)
        butler.participants.append(uid=participant_uid, key='received_rewards', value=reward)
        butler.participants.increment(uid=participant_uid, key='num_reported_answers')

        update_plot_data = {'rewards': reward,
                            'participant_uid': participant_uid,
                            'initial_arm': init_id,
                            'arm_pulled': arm_id,
                            'alg': 'TS',
                            'time': num_responses
        }

        butler.experiment.append(key='plot_data', value=update_plot_data)

        task_args = {
            'arm_id': arm_id,
            'reward': reward,
            'participant_uid': participant_uid
        }

        butler.job('modelUpdate', task_args, ignore_result=True)

        return True

    def modelUpdate(self, butler, task_args):
        arm_id = task_args['arm_id']
        reward = task_args['reward']
        participant_uid = task_args['participant_uid']

        # Algorithm related constants - don't need to be updated
        d = butler.algorithms.get(key='d')
        R = butler.algorithms.get(key='R')
        delta = butler.algorithms.get(key='delta')
        multiplier = butler.algorithms.get(key='multiplier')

        # Algorithm related variables, need to be updated per participant
        invVt = np.array(butler.participants.get(uid=participant_uid, key='invVt'))
        b = np.array(butler.participants.get(uid=participant_uid, key='b'))
        x_invVt_norm = butler.participants.get(uid=participant_uid, key='x_invVt_norm')
        logDetV = butler.participants.get(uid=participant_uid, key='logDetV')

        # Algorithm related variables that don't need to be updated
        features = np.load('features.npy')
        t = butler.participants.get(uid=participant_uid, key='num_reported_answers')

        xt = features[arm_id, :]
        u = invVt.dot(xt)
        v = u.dot(xt)
        invVt -= np.outer(u, u) / (1 + v)
        x_invVt_norm -= np.dot(features, u) ** 2 / (1 + v)
        b += reward * xt
        logDetV += np.log(1 + v)

        matR = linalg.cholesky(invVt).T
        theta_hat = invVt.dot(b)

        sqrt_beta = R * np.sqrt(9 * d * np.log(t/delta))
        tmp = ra.normal(size=(d,))
        theta_til = (multiplier ** 2) * sqrt_beta * np.dot(tmp, matR) + theta_hat

        expected_rewards = np.dot(features, theta_til)

        butler.participants.set(uid=participant_uid, key='arm_order', value=np.argsort(expected_rewards)[::-1])
        butler.participants.set(uid=participant_uid, key='invVt', value=invVt)
        butler.participants.set(uid=participant_uid, key='b', value=b)
        butler.participants.set(uid=participant_uid, key='x_invVt_norm', value=x_invVt_norm)
        butler.participants.set(uid=participant_uid, key='logDetV', value=logDetV)

        return True

    def getModel(self, butler):
        return butler.algorithms.get(key=['received_rewards', 'num_reported_answers'])


