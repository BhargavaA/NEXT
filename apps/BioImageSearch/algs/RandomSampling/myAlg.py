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
    next_arm = unasked_arms[np.argmax(expected_rewards[unasked_arms])]
    unasked_arms = np.setdiff1d(unasked_arms, next_arm)
    butler.algorithms.set(key='unasked_arms', value=unasked_arms)
    return [next_arm]

  def processAnswer(self, butler, arm_context, reward, num_responses, init_context, participant_uid):
    if num_responses == 1:
        butler.participants.set(uid=participant_uid, key='received_rewards', value=[])
    butler.participants.append(key='received_rewards', value=reward)
    butler.participants.increment(key='num_reported_answers')
    return True

  def getModel(self, butler):
    return butler.algorithms.get(key=['received_rewards', 'num_reported_answers'])


