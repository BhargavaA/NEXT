import time
from numpy.random import shuffle
import numpy as np
import next.utils as utils
from numpy.linalg import inv


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

        # butler.participants.append(uid=participant_uid, key='do_not_ask', value=next_arm)
        return return_arms

    def processAnswer(self, butler, arm_id, reward, num_responses, init_id, participant_uid, relevant_indices):
        if num_responses == 1:
            d = butler.algorithms.get(key='d')
            n = butler.algorithms.get(key='n')
            ridge = butler.algorithms.get(key='ridge')
            invVt = np.eye(d) / ridge
            b = np.zeros(d)
            x_invVt_norm = np.ones(n) / ridge
            revealed_indices = []

            butler.participants.set(uid=participant_uid, key='received_rewards', value=[])
            butler.participants.set(uid=participant_uid, key='arms_pulled', value=[])
            butler.participants.set(uid=participant_uid, key='invVt', value=invVt)
            butler.participants.set(uid=participant_uid, key='b', value=b)
            butler.participants.set(uid=participant_uid, key='x_invVt_norm', value=x_invVt_norm)
            butler.participants.set(uid=participant_uid, key='revealed_indices', value=revealed_indices)

            self.update_parameters(butler, participant_uid, revealed_indices)

        revealed_indices = butler.participants.get(key='revealed_indices')
        prev_d = len(revealed_indices)
        revealed_indices = list(set().union(revealed_indices, relevant_indices))
        new_d = len(revealed_indices)

        butler.participants.append(uid=participant_uid, key='do_not_ask', value=arm_id)

        butler.participants.append(uid=participant_uid, key='received_rewards', value=reward)
        butler.participants.append(uid=participant_uid, key='arms_pulled', value=arm_id)
        butler.participants.increment(uid=participant_uid, key='num_reported_answers')
        butler.participants.set(uid=participant_uid, key='revealed_indices', value=revealed_indices)
        butler.participants.set(uid=participant_uid, key='prev_d', value=prev_d)
        butler.participants.set(uid=participant_uid, key='new_d', value=new_d)

        update_plot_data = {'rewards': reward,
                            'participant_uid': participant_uid,
                            'initial_arm': init_id,
                            'arm_pulled': arm_id,
                            'alg': 'FFOFUL',
                            'time': num_responses
                            }

        butler.experiment.append(key='plot_data', value=update_plot_data)

        task_args = {
            'arm_id': arm_id,
            'reward': reward,
            'participant_uid': participant_uid,
            'prev_d': prev_d,
            'new_d': new_d,
            'revealed_indices': revealed_indices
        }

        butler.job('modelUpdate', task_args, ignore_result=True)

        return True

    def modelUpdate(self, butler, task_args):
        arm_id = task_args['arm_id']
        reward = task_args['reward']
        participant_uid = task_args['participant_uid']
        prev_d = task_args['prev_d']
        new_d = task_args['new_d']
        revealed_indices = task_args['revealed_indices']

        R = butler.algorithms.get(key='R')
        S = butler.algorithms.get(key='S')
        delta = butler.algorithms.get(key='delta')
        ridge = butler.algorithms.get(key='ridge')
        t = butler.participants.get(uid=participant_uid, key='num_reported_answers')

        if new_d == prev_d:
            invVt = np.array(butler.participants.get(uid=participant_uid, key='invVt'))
            b = np.array(butler.participants.get(uid=participant_uid, key='b'))
            features = np.load('small_features.npy')
            x_invVt_norm = butler.participants.get(uid=participant_uid, key='x_invVt_norm')

            xt = features[arm_id, :]
            u = invVt.dot(xt)
            invVt -= np.outer(u, u) / (1 + np.inner(xt, u))
            x_invVt_norm -= np.dot(features, u) ** 2 / (1 + np.inner(xt, u))
            b += reward * xt
            theta_hat = invVt.dot(b)
            sqrt_beta = R * np.sqrt(new_d * np.log((1 + t / (ridge * new_d)) / delta)) + np.sqrt(ridge) * S
            expected_rewards = np.dot(features, theta_hat) + sqrt_beta * np.sqrt(x_invVt_norm)
        else:
            self.update_parameters(butler, participant_uid, revealed_indices)
            features = np.load('small_features.npy')
            invVt = np.array(butler.participants.get(uid=participant_uid, key='invVt'))
            x_invVt_norm = butler.participants.get(uid=participant_uid, key='x_invVt_norm')
            b = np.array(butler.participants.get(uid=participant_uid, key='b'))
            theta_hat = invVt.dot(b)
            sqrt_beta = R * np.sqrt(new_d * np.log((1 + t / (ridge * new_d)) / delta)) + np.sqrt(ridge) * S
            expected_rewards = np.dot(features, theta_hat) + sqrt_beta * np.sqrt(x_invVt_norm)

        butler.participants.set(uid=participant_uid, key='arm_order', value=np.argsort(expected_rewards)[::-1])
        butler.participants.set(uid=participant_uid, key='invVt', value=invVt)
        butler.participants.set(uid=participant_uid, key='b', value=b)
        butler.participants.set(uid=participant_uid, key='x_invVt_norm', value=x_invVt_norm)


        return True

    def getModel(self, butler):
        return butler.algorithms.get(key=['received_rewards', 'num_reported_answers'])

    def update_parameters(self, butler, participant_uid, revealed_indices):
        if len(revealed_indices) == 0:
            n = butler.experiment.get(key='n')
            arm_order = range(n)
            shuffle(arm_order)
            invVt = 1.0
            b = 0.0
            x_invVt_norm = np.ones(n)
            small_features = np.ones(n)
        else:
            new_d = len(revealed_indices)
            features = np.load('features.npy')
            small_features = features[:, revealed_indices].toarray()
            rewards = butler.participants.get(uid=participant_uid, key='received_rewards')
            arms = butler.participants.get(uid=participant_uid, key='arms_pulled')
            ridge = butler.algorithms.get(key='ridge')
            Vt = np.eye(new_d) * ridge
            b = np.zeros(new_d)

            for i, arm in enumerate(arms):
                xt = small_features[i, :]
                Vt += np.outer(xt, xt)
                b += rewards[i] * xt

            invVt = inv(Vt)
            x_invVt_norm = np.sum(np.dot(small_features, invVt) * small_features, axis=1)

        butler.participants.set(uid=participant_uid, key='invVt', value=invVt)
        butler.participants.set(uid=participant_uid, key='b', value=b)
        butler.participants.set(uid=participant_uid, key='x_invVt_norm', value=x_invVt_norm)
        np.save('small_features.npy', small_features)

        return True




