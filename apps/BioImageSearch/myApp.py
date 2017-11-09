from __future__ import print_function
import json
import next.utils as utils
import next.apps.SimpleTargetManager
import numpy as np


class MyApp:
    def __init__(self, db):
        self.app_id = 'BioImageSearch'
        self.TargetManager = next.apps.SimpleTargetManager.SimpleTargetManager(db)

    def initExp(self, butler, init_algs, args):
        exp_uid = butler.exp_uid
        if 'targetset' in args['targets'].keys():
            n = len(args['targets']['targetset'])
            targets = args['targets']['targetset']
            self.TargetManager.set_targetset(exp_uid, targets)
            d = args['d']
            features = np.zeros((n, d))
            for i in range(n):
                features[i, :] = targets[i]['context']

            np.save('features.npy', features)

        else:
            n = args['targets']['n']
        args['n'] = n
        del args['targets']

        alg_data = {}
        algorithm_keys = ['n', 'd','failure_probability']
        for key in algorithm_keys:
            if key in args:
                alg_data[key] = args[key]

        init_algs(alg_data)
        return args

    def getQuery(self, butler, alg, args):
        experiment = butler.experiment.get()
        n = experiment['args']['n']
        exp_uid = experiment['exp_uid']
        participant_uid = args['participant_uid']
        num_responses = butler.participants.get(uid=participant_uid, key='num_responses')
        init_arm = int(args['init_arm'])
        print('init_arm:', init_arm)
        if num_responses == 0 or num_responses is None:
            butler.participants.set(uid=participant_uid, key='init_arm', value=init_arm)
            arm_order = range(n)
            np.random.shuffle(arm_order)
            butler.participants.set(uid=participant_uid, key='arm_order', value=arm_order)
            butler.participants.set(uid=participant_uid, key='do_not_ask', value=[init_arm])
            print('Initialized lists in getQuery')

        alg_response = alg({'participant_uid': participant_uid})
        exp_uid = butler.exp_uid

        if num_responses == 0 or num_responses is None:
            butler.participants.set(uid=participant_uid, key='init_arm', value=init_arm)
            arm_order = range(n)
            np.random.shuffle(arm_order)
            butler.participants.set(uid=participant_uid, key='arm_order', value=arm_order)

        utils.debug_print('Alg_resp:', alg_response)
        target_indices = alg_response

        targets = [self.TargetManager.get_target_item(exp_uid, a) for a in target_indices]
        init_target = init_arm and self.TargetManager.get_target_item(exp_uid, init_arm)

        return_dict = {
            'target_indices': target_indices,
            'targets': targets,
            'init_target': init_target,
            'instructions': 'Is this the kind of image you are looking for?',
            'count': 1,
        }

        return return_dict

    def processAnswer(self, butler, alg, args):
        query = butler.queries.get(uid=args['query_uid'])
        participant_uid = query['participant_uid']
        exp_uid = query['exp_uid']
        targets = query['target_indices']
        rewards = args['target_rewards']
        num_responses = butler.participants.increment(uid=participant_uid, key='num_responses')
        # print(targets, rewards)

        experiment = butler.experiment.get()
        num_responses = butler.participants.get(uid=participant_uid, key='num_responses')
        init_arm = butler.participants.get(uid=participant_uid, key='init_arm')
        utils.debug_print('Num user responses: %d' % num_responses)
        num_reported_answers = butler.experiment.increment(key='num_reported_answers_for_' + query['alg_label'])
        init_context = self.TargetManager.get_target_item(exp_uid, init_arm)['context']


        n = experiment['args']['n']
        if num_reported_answers % ((n+4)/4) == 0:
            butler.job('getModel', json.dumps(
                {'exp_uid': butler.exp_uid, 'args': {'alg_label': query['alg_label'], 'logging': True}}))

        for target, reward in zip(targets, rewards):
            arm_context = self.TargetManager.get_target_item(exp_uid, target)['context']
            alg({'arm_id': target, 'reward': reward, 'num_responses': num_responses, 'init_id': init_arm,
                 'participant_uid': participant_uid})

        return {'target_ids': targets, 'target_rewards': rewards}

    def getModel(self, butler, alg, args):
        return alg()

    def format_responses(self, responses):
        formatted = []
        for response in responses:
            if 'target_winner' not in response:
                continue
            targets = {'target_' + target['label']: target['primary_description']
                       for target in response['target_indices']}
            ids = {target['label'] + '_id': target['target_id'] for target in response['target_indices']}
            winner = {t['target_id'] == response['target_winner']: (t['primary_description'], t['target_id'])
                      for t in response['target_indices']}
            response.update({'target_winner': winner[True][0], 'winner_id': winner[True][1]})

            for key in ['q', '_id', 'target_indices']:
                if key in response:
                    del response[key]
            response.update(targets)
            response.update(ids)
            formatted += [response]

        return formatted
