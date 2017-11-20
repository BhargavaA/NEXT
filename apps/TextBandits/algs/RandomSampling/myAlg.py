import time
import numpy.random as ra
import numpy as np
import next.utils as utils


class MyAlg:
  def initExp(self, butler, n, d, failure_probability):
    butler.algorithms.set(key='n', value=n)
    butler.algorithms.set(key='d', value=d)
    butler.algorithms.set(key='delta', value=failure_probability)

    if butler.experiment.get(key='plot_data') is None:
        butler.experiment.set(key='plot_data', value=[])

    return True

  def getQuery(self, butler, participant_uid):
    arm_order = butler.participants.get(uid=participant_uid, key='arm_order')
    do_not_ask = butler.participants.get(uid=participant_uid, key='do_not_ask')

    if arm_order is None:
        n = butler.algorithms.get(key='n')
        arm_order = range(n)
        ra.shuffle(arm_order)

    for next_arm in arm_order:
        if next_arm not in do_not_ask:
            break

    return next_arm

  def processAnswer(self, butler, arm_id, reward, num_responses, init_id, participant_uid, relevant_indices):
    utils.debug_print('ris:', relevant_indices)
    if num_responses == 1:
        butler.participants.set(uid=participant_uid, key='received_rewards', value=[])
    butler.participants.append(key='received_rewards', value=reward)
    butler.participants.increment(key='num_reported_answers')
    butler.participants.append(uid=participant_uid, key='do_not_ask', value=arm_id)

    update_plot_data = {'rewards': reward,
                        'participant_uid': participant_uid,
                        'initial_arm': init_id,
                        'arm_pulled': arm_id,
                        'alg': 'TS',
                        'time': num_responses
                        }

    butler.dashboard.append(key='plot_data', value=update_plot_data)

    return True

  def getModel(self, butler):
    return butler.algorithms.get(key=['received_rewards', 'num_reported_answers'])


