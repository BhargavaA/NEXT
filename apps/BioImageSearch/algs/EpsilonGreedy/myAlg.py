import time
import numpy.random as ra
import numpy as np
import next.utils as utils


class MyAlg:
    def initExp(self, butler, n, d, failure_probability):
        unasked_arms = range(n)
        expected_rewards = ra.normal(0, 1, n)
        butler.algorithms.set(key='n', value=n)
        butler.algorithms.set(key='d', value=d)
        butler.algorithms.set(key='delta', value=failure_probability)
        butler.algorithms.set(key='unasked_arms', value=unasked_arms)
        butler.algorithms.set(key='expected_rewards', value=expected_rewards.tolist())
        butler.algorithms.set(key='num_reported_answers', value=0)
        butler.algorithms.set(key='received_rewards', value=[])
        return True

    def getQuery(self, butler):
        expected_rewards = np.array(butler.algorithms.get(key='expected_rewards'))
        unasked_arms = np.array(butler.algorithms.get(key='unasked_arms'))
        epsilon = butler.experiment.get(key='epsilon')
        if ra.rand() <= epsilon:
            next_arm = np.random.choice(unasked_arms)
        else:
            next_arm = unasked_arms[np.argmax(expected_rewards[unasked_arms])]

        unasked_arms = np.setdiff1d(unasked_arms, next_arm)
        butler.algorithms.set(key='unasked_arms', value=unasked_arms)
        return [next_arm]

    def processAnswer(self, butler, arm_context, reward, num_responses, init_context, participant_uid):
        if num_responses == 1:
            d = butler.experiment.get(key='d')
            ridge = butler.experiment.get(key='ridge')
            invVt = np.eye(d)*ridge
            b = np.zeros(d)
            theta_hat = np.array(init_context)

            butler.participants.set(uid=participant_uid, key='received_rewards', value=[])
            butler.participants.set(uid=participant_uid, key='invVt', value=invVt)
            butler.participants.set(uid=participant_uid, key='b', value=b)
            butler.participants.set(uid=participant_uid, key='theta_hat', value=theta_hat)

        butler.participants.append(key='received_rewards', value=reward)
        butler.participants.increment(key='num_reported_answers')

        task_args = {
            'arm_context': arm_context,
            'reward': reward,
            'participant_uid': participant_uid
        }

        butler.job('modelUpdate', task_args, ignore_result=True)

        return True

    def modelUpdate(self, butler, task_args):
        arm_context = task_args['arm_context']
        reward = task_args['reward']
        participant_uid = task_args['participant_uid']

        invVt = butler.participants.get(uid=participant_uid, key='invVt')
        b = butler.participants.get(uid=participant_uid, key='b')

        u = invVt.dot(arm_context)
        invVt -= np.outer(u, u) / (1 + np.inner(arm_context, u))

        # x_invVt_norm -= np.dot(X, u) ** 2 / (1 + np.inner(arm_pulled, u))

        b += reward * arm_pulled
        theta_hat = X[i_hat, :] + invV.dot(b)


        expected_rewards = np.dot(X, theta_hat)


        return True

    def getModel(self, butler):
        return butler.algorithms.get(key=['received_rewards', 'num_reported_answers'])


