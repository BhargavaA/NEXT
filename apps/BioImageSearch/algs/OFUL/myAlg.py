import time
import numpy.random as ra
import numpy as np
import next.utils as utils


class MyAlg:
    def initExp(self, butler, n, d, failure_probability):
        butler.algorithms.set(key='n', value=n)
        butler.algorithms.set(key='d', value=d)
        butler.algorithms.set(key='delta', value=failure_probability)
        return True

    def getQuery(self, butler, participant_uid):
        arm_order = butler.participants.get(uid=participant_uid, key='arm_order')
        do_not_ask = butler.participants.get(uid=participant_uid, key='do_not_ask')
        epsilon = butler.algorithms.get(key='params')['epsilon']

        if ra.rand() <= epsilon:
            next_arm = np.random.choice(np.setdiff1d(arm_order, do_not_ask))
        else:
            for next_arm in arm_order:
                if next_arm not in do_not_ask:
                    break

        butler.participants.append(uid=participant_uid, key='do_not_ask', value=next_arm)
        return [next_arm]

    def processAnswer(self, butler, arm_id, reward, num_responses, init_id, participant_uid):
        if num_responses == 1:
            d = butler.algorithms.get(key='d')
            ridge = butler.algorithms.get(key='params')[u'ridge']
            invVt = np.eye(d)*ridge
            b = np.zeros(d)
            # theta_hat = np.array(init_context)

            butler.participants.set(uid=participant_uid, key='received_rewards', value=[])
            butler.participants.set(uid=participant_uid, key='invVt', value=invVt)
            butler.participants.set(uid=participant_uid, key='b', value=b)
            # butler.participants.set(uid=participant_uid, key='theta_hat', value=theta_hat)

        butler.participants.append(key='received_rewards', value=reward)
        butler.participants.increment(key='num_reported_answers')

        task_args = {
            'arm_context': arm_id,
            'reward': reward,
            'participant_uid': participant_uid
        }

        butler.job('modelUpdate', task_args, ignore_result=True)

        return True

    def modelUpdate(self, butler, task_args):
        arm_context = task_args['arm_context']
        reward = task_args['reward']
        participant_uid = task_args['participant_uid']

        invVt = np.array(butler.participants.get(uid=participant_uid, key='invVt'))
        b = np.array(butler.participants.get(uid=participant_uid, key='b'))
        features = np.load('features.npy')

        u = invVt.dot(features[arm_context,:])
        invVt -= np.outer(u, u) / (1 + np.inner(features[arm_context,:], u))

        # x_invVt_norm -= np.dot(X, u) ** 2 / (1 + np.inner(arm_pulled, u))

        b += reward * arm_context
        theta_hat = invVt.dot(b)

        expected_rewards = np.dot(features, theta_hat)
        butler.participants.set(uid=participant_uid, key='arm_order', value=np.argsort(expected_rewards)[::-1])

        return True

    def getModel(self, butler):
        return butler.algorithms.get(key=['received_rewards', 'num_reported_answers'])


