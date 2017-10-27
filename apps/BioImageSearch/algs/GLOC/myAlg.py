import time
from numpy.linalg import norm
import numpy as np
import next.utils as utils
import cvxpy as cvx


class MyAlg:
    def initExp(self, butler, n, d, failure_probability):
        params = butler.algorithms.get(key='params')
        R = params[u'R']
        S = params[u'S']
        ridge = params[u'ridge']
        c = params[u'c']
        kappa = params[u'kappa']

        butler.algorithms.set(key='n', value=n)
        butler.algorithms.set(key='d', value=d)
        butler.algorithms.set(key='R', value=R)
        butler.algorithms.set(key='S', value=S)
        butler.algorithms.set(key='ridge', value=ridge)
        butler.algorithms.set(key='c', value=c)
        butler.algorithms.set(key='kappa', value=kappa)
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
            S = butler.algorithms.get(key='S')

            invVt = np.eye(d) / ridge
            Vt = np.eye(d) * ridge
            b = np.zeros(d)
            x_invVt_norm = np.ones(n) / ridge
            XTz = np.zeros(d)

            theta_s = np.random.normal(0, 1, d)
            theta_s = S * theta_s / norm(theta_s)

            butler.participants.set(uid=participant_uid, key='received_rewards', value=[])
            butler.participants.set(uid=participant_uid, key='invVt', value=invVt)
            butler.participants.set(uid=participant_uid, key='Vt', value=Vt)
            butler.participants.set(uid=participant_uid, key='b', value=b)
            butler.participants.set(uid=participant_uid, key='x_invVt_norm', value=x_invVt_norm)
            butler.participants.set(uid=participant_uid, key='theta_s', value=theta_s)
            # butler.participants.set(uid=participant_uid, key='theta_hat', value=theta_s)
            butler.participants.set(uid=participant_uid, key='XTz', value=XTz)
            butler.participants.set(uid=participant_uid, key='sum_z_t_sq', value=0.)
            butler.participants.set(uid=participant_uid, key='radius_problem_constant', value=0.)

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
        ridge = butler.algorithms.get(key='ridge')
        c = butler.algorithms.get(key='c')
        kappa = butler.algorithms.get(key='kappa')

        sum_z_t_sq = butler.participants.get(uid=participant_uid, key='sum_z_t_sq')
        radius_problem_constant = butler.participants.get(uid=participant_uid, key='radius_problem_constant')
        theta_s = butler.participants.get(uid=participant_uid, key='theta_s')
        Vt = np.array(butler.participants.get(uid=participant_uid, key='Vt'))
        invVt = np.array(butler.participants.get(uid=participant_uid, key='invVt'))
        b = np.array(butler.participants.get(uid=participant_uid, key='b'))
        features = np.load('features.npy')
        x_invVt_norm = butler.participants.get(uid=participant_uid, key='x_invVt_norm')
        XTz = butler.participants.get(uid=participant_uid, key='XTz')
        t = butler.participants.get(uid=participant_uid, key='num_reported_answers')

        xt = features[arm_id, :]
        u = invVt.dot(xt)
        invVt -= np.outer(u, u) / (1 + np.inner(xt, u))
        x_invVt_norm -= np.dot(features, u) ** 2 / (1 + np.inner(xt, u))
        b += reward * xt

        Vt += np.outer(xt, xt)

        loss_prime = (1 / (1 + np.exp(-np.dot(theta_s, xt))) - reward)
        theta_prime = theta_s - loss_prime * np.dot(invVt, xt) / kappa

        theta_s = self.solve_by_cvx(theta_prime, Vt, d, S)

        g_t = (1 / (1 + np.exp(-np.dot(theta_s, xt))) - reward)
        z = np.dot(theta_s, xt)
        XTz += z * xt
        sum_z_t_sq += z ** 2
        theta_hat = np.dot(invVt, XTz)

        radius_problem_constant += (g_t ** 2) * np.dot(xt, np.dot(invVt, xt))

        radius = c * self._calc_radius_sq_new(R, t, kappa, radius_problem_constant, theta_hat, sum_z_t_sq, XTz, ridge, S)

        expected_rewards = np.dot(features, theta_hat) + np.sqrt(radius) * np.sqrt(
            np.sum(np.dot(features, invVt) * features, axis=1))

        butler.participants.set(uid=participant_uid, key='arm_order', value=np.argsort(expected_rewards)[::-1])
        butler.participants.set(uid=participant_uid, key='invVt', value=invVt)
        butler.participants.set(uid=participant_uid, key='b', value=b)
        butler.participants.set(uid=participant_uid, key='radius_problem_constant', value=radius_problem_constant)
        butler.participants.set(uid=participant_uid, key='Vt', value=Vt)
        butler.participants.set(uid=participant_uid, key='sum_z_t_sq', value=sum_z_t_sq)
        butler.participants.set(uid=participant_uid, key='theta_s', value=theta_s)
        butler.participants.set(uid=participant_uid, key='x_invVt_norm', value=x_invVt_norm)
        butler.participants.set(uid=participant_uid, key='XTz', value=XTz)
        # butler.participants.set(uid=participant_uid, key='', value=)
        # butler.participants.set(uid=participant_uid, key='', value=)

        return True

    def getModel(self, butler):
        return butler.algorithms.get(key=['received_rewards', 'num_reported_answers'])

    @staticmethod
    def solve_by_cvx(theta_prime, Vt, d, S):
        th = cvx.Variable(d)
        obj = cvx.Minimize(cvx.quad_form(th - theta_prime, Vt))
        cons = [cvx.norm(th) <= S]

        prob = cvx.Problem(obj, cons)
        prob.solve()

        return np.array(th.value).flatten()

    @staticmethod
    def _calc_radius_sq_new(R, t, kappa, radius_problem_constant, theta_hat, sum_z_t_sq, XTz, ridge, S):
        dt = 0.2
        if (t == 1):
            radius_sq = 0
        else:
            #- still, omitting some terms.
            B = (0.5/kappa)*radius_problem_constant + 2*kappa*(S**2) * ridge
            inner = 1 + (2/kappa)*B + 4*R**4/(kappa**4 * dt**2)
            extra = (sum_z_t_sq - theta_hat.dot(XTz))
            assert (extra > -1e-8);
            radius_sq = 1 + (4.0/kappa)*B \
                          + (8*R**2/kappa**2)*np.log((2/dt)*np.sqrt( inner )) \
                          - extra
        return radius_sq


