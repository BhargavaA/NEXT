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
            self.TargetManager.set_targetset(exp_uid, args['targets']['targetset'])
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
        num_answers = butler.participants.get(uid=participant_uid, key='num_reported_answers')
        utils.debug_print('num_answers:', num_answers)
        if num_answers is None:
            butler.participants.set(uid=participant_uid, key='num_reported_answers', value=0)
            next_arm = np.random.choice(range(n))
            utils.debug_print('First arm shown: %d' % next_arm)
            target = self.TargetManager.get_target_item(exp_uid=exp_uid, target_id=next_arm)
            targets_list = {
                'index': next_arm,
                'target': target,
                'instructions': 'Pick similar images'
            }
            return_dict = {
                'initial_query': True,
                'targets': [targets_list],
                'instructions': 'When ready, please click on the image to start',
                'target_indices': [target]
            }
            butler.experiment.set(key='init_arm', value=next_arm)
        else:
            alg_response = alg()
            exp_uid = butler.exp_uid
            init_arm = butler.experiment.get(key='init_arm')
            next_arm = self.TargetManager.get_target_item(exp_uid, alg_response[0])
            target = self.TargetManager.get_target_item(exp_uid, next_arm)
            init_target = self.TargetManager.get_target_item(exp_uid, init_arm)
            targets_list = {
                'index': next_arm,
                'target': target
            }
            return_dict = {
                'initial_query': False,
                'targets': targets_list,
                'main_target': init_target,
                'instructions': 'Is this the kind of image you are looking for?',
                'count': 1,
                'target_indices': [target]
            }
            # return {'next_arm': [next_arm]}

        return return_dict

    def processAnswer(self, butler, alg, args):
        query = butler.queries.get(uid=args['query_uid'])
        target = query['target_indices']
        reward = args['answer']['target_reward']

        experiment = butler.experiment.get()
        num_reported_answers = butler.experiment.increment(key='num_reported_answers_for_' + query['alg_label'])

        n = experiment['args']['n']
        if num_reported_answers % ((n+4)/4) == 0:
            butler.job('getModel', json.dumps(
                {'exp_uid': butler.exp_uid, 'args': {'alg_label': query['alg_label'], 'logging': True}}))

        alg({'arm_id': target, 'reward': reward})
        return {'target_id': target, 'target_reward': reward}

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
