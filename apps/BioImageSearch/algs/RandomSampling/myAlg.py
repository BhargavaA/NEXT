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

  def getQuery(self, butler, participant_uid):
    arm_order = butler.participants.get(uid=participant_uid, key='arm_order')
    do_not_ask = butler.participants.get(uid=participant_uid, key='do_not_ask')
    ask_list = np.setdiff1d(arm_order, do_not_ask)
    butler.participants.append(key='do_not_ask', value=ask_list[0])
    return [ask_list[0]]

  def processAnswer(self, butler, arm_id, reward, num_responses, init_id, participant_uid):
    if num_responses == 1:
        butler.participants.set(uid=participant_uid, key='received_rewards', value=[])
    butler.participants.append(key='received_rewards', value=reward)
    butler.participants.increment(key='num_reported_answers')
    return True

  def getModel(self, butler):
    return butler.algorithms.get(key=['received_rewards', 'num_reported_answers'])


