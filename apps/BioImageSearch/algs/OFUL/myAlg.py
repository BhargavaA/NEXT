import time
import numpy.random as ra
import numpy as np
import next.utils as utils


class MyAlg:
    def initExp(self, butler, n, d, failure_probability):
        params = butler.algorithms.get(key='params')
        R = params[u'R']
        S = params[u'S']
        ridge = params[u'ridge']

        butler.algorithms.set(key='n', value=n)
        butler.algorithms.set(key='d', value=d)
        butler.algorithms.set(key='R', value=R)
        butler.algorithms.set(key='S', value=S)
        butler.algorithms.set(key='ridge', value=ridge)
        butler.algorithms.set(key='delta', value=failure_probability)
        return True

    def getQuery(self, butler, participant_uid):
        arm_order = butler.participants.get(uid=participant_uid, key='arm_order')
        do_not_ask = butler.participants.get(uid=participant_uid, key='do_not_ask')

        for next_arm in arm_order:
            if next_arm not in do_not_ask:
                break

        butler.participants.append(uid=participant_uid, key='do_not_ask', value=next_arm)
        return [next_arm]

    def processAnswer(self, butler, arm_id, reward, num_responses, init_id, participant_uid):
        if num_responses == 1:
            d = butler.algorithms.get(key='d')
            n = butler.algorithms.get(key='n')
            ridge = butler.algorithms.get(key='ridge')
            invVt = np.eye(d)*ridge
            b = np.zeros(d)
            x_invVt_norm = np.ones(n) / ridge

            butler.participants.set(uid=participant_uid, key='received_rewards', value=[])
            butler.participants.set(uid=participant_uid, key='invVt', value=invVt)
            butler.participants.set(uid=participant_uid, key='b', value=b)
            butler.participants.set(uid=participant_uid, key='x_invVt_norm', value=x_invVt_norm)

        butler.participants.append(uid=participant_uid, key='received_rewards', value=reward)
        butler.participants.increment(uid=participant_uid, key='num_reported_answers')

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

        d = butler.algorithms.get(key='d')
        R = butler.algorithms.get(key='R')
        S = butler.algorithms.get(key='S')
        delta = butler.algorithms.get(key='delta')
        ridge = butler.algorithms.get(key='ridge')

        invVt = np.array(butler.participants.get(uid=participant_uid, key='invVt'))
        b = np.array(butler.participants.get(uid=participant_uid, key='b'))
        features = np.load('features.npy')
        x_invVt_norm = butler.participants.get(uid=participant_uid, key='x_invVt_norm')
        t = butler.participants.get(uid=participant_uid, key='num_reported_answers')

        xt = features[arm_id, :]
        u = invVt.dot(xt)
        invVt -= np.outer(u, u) / (1 + np.inner(xt, u))
        x_invVt_norm -= np.dot(features, u) ** 2 / (1 + np.inner(xt, u))
        b += reward * xt
        theta_hat = invVt.dot(b)
        utils.debug_print((1 + t / (ridge * d)))
        sqrt_beta = R * np.sqrt(d * np.log((1 + t / (ridge * d)) / delta)) + np.sqrt(ridge) * S
        expected_rewards = np.dot(features, theta_hat) + sqrt_beta * np.sqrt(x_invVt_norm)

        butler.participants.set(uid=participant_uid, key='arm_order', value=np.argsort(expected_rewards)[::-1])
        butler.participants.set(key='invVt', value=invVt)
        butler.participants.set(key='b', value=b)

        return True

    def getModel(self, butler):
        return butler.algorithms.get(key=['received_rewards', 'num_reported_answers'])


