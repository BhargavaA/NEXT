"""
Usage:
    launch.py <target_filename>

Arguments:
    target_filename (required) : path to a json file of targets

Options:
    -h, --help

Example:

    > cd ~/Developer/NEXT/examples/
    > python launch_bio_image_search.py /Users/aniruddha/Downloads/bio_dictionary.json

"""

from __future__ import print_function
import os
import csv
import sys
from collections import OrderedDict
import numpy as np
import requests
import json

sys.path.append('../../next/lib')
from docopt import docopt


def launch(targets_filename=None, upload=False):

    with open(targets_filename) as f:
        target_dictionary = json.load(f)

    targetset = []
    keys = target_dictionary.keys()
    n = len(keys)
    d = len(target_dictionary[keys[0]]['context'])
    for i, key in enumerate(keys):
        targetset.append({'target_id': str(i),
                          'primary_type': 'image',
                          'primary_description': 'http://localhost:8999/' + target_dictionary[key]['name'],
                          'context': str(target_dictionary[key]['context']),
                          'alt_type': 'text',
                          'alt_description': 'Image {}'.format(i)})

    supported_alg_ids = ['RandomSampling']

    alg_list = []
    for alg_id in supported_alg_ids:
        alg_item = {}
        alg_item['alg_id'] = alg_id
        alg_item['alg_label'] = alg_id
        alg_item['test_alg_label'] = alg_id
        if alg_id == 'EpsilonGreedy':
            alg_item['params'] = {
                'epsilon': 0.1,
                'ridge': 1.0
            }
        elif alg_id == 'OFUL':
            alg_item['params'] = {
                'ridge': 1.0,
                'R': 1.0,
                'S': 1.0,
                'L': 1.0,
                'c': 1.0
            }
        # alg_item['params'] = {}
        alg_list.append(alg_item)

    algorithm_management_settings = {}
    params = []
    for algorithm in alg_list:
        params += [{'alg_label': algorithm['alg_label'],
                    'proportion': 1.0 / len(alg_list)}]

    algorithm_management_settings['mode'] = 'fixed_proportions'
    algorithm_management_settings['params'] = params
    # Create experiment dictionary
    initExp = {}
    initExp['app_id'] = 'BioImageSearch'
    initExp['args'] = {}
    initExp['args']['failure_probability'] = 0.05
    initExp['args']['participant_to_algorithm_management'] = 'one_to_one'
    initExp['args']['algorithm_management_settings'] = algorithm_management_settings
    initExp['args']['alg_list'] = alg_list
    initExp['args']['num_tries'] = 50
    initExp['args']['d'] = d
    initExp['args']['n'] = n
    initExp['args']['targets'] = {'targetset': targetset}
    # initExp['args'] = {
    #     'failure_probability': 0.05,
    #     'participant_to_algorithm_management': 'one_to_one',
    #     'algorithm_management_settings': algorithm_management_settings,
    #     'alg_list': alg_list,
    #     'num_tries': 50,
    #     'd': d,
    #     'n': n,
    #     'targets': {'targetset': targetset}
    # }

    # initExp['args']['R'] = 1.
    # initExp['args']['rating_scale'] = {'labels': [{'label':'yes', 'reward': 1},
    #                                              {'label':'no', 'reward': -1}]}
    # with open('initExp.json', 'w') as f:
    #     json.dump(initExp, f)

    host_url = os.environ.get('NEXT_BACKEND_GLOBAL_HOST', 'localhost')
    host_url = 'http://' + host_url + ':8000'

    print("Host URL: %s"%host_url)
    print('Initializing experiment')
    response = requests.post(host_url + '/api/experiment',
                             json.dumps(initExp),
                             headers={'content-type': 'application/json'})
    # response = requests.post(host_url + '/assistant/init/experiment',
    #                          json.dumps(initExp),
    #                          headers={'content-type':'application/json'})

    initExp_response_dict = json.loads(response.text)
    print('initExp_response_dict', initExp_response_dict)
    exp_uid = initExp_response_dict['exp_uid']

if __name__ == "__main__":
    args = docopt(__doc__)
    upload = False
    print(args['<target_filename>'])

    print(args, '\n')
    launch(targets_filename=args['<target_filename>'], upload=upload)
