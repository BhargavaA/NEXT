"""
Usage:
    launch.py <target_filename> <context_file>

Arguments:
    target_filename (required) : path to a csv file of targets
    context_file (required) : path to the context matrix

Options:
    -h, --help

Example:

    > cd ~/Developer/NEXT/examples/
    > python launch.py strange_fruit_triplet/targets.csv strange_fruit_triplet/features.npy

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


def launch(targets_filename=None, context_filename=None, upload=False):
    supported_alg_ids = ['RandomSampling']

    alg_list = []
    for alg_id in supported_alg_ids:
        alg_item = {}
        alg_item['alg_id'] = alg_id
        alg_item['alg_label'] = alg_id
        alg_item['test_alg_label'] = alg_id
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
    initExp['app_id'] = 'PoolBasedTripletMDS'
    initExp['args'] = {}
    initExp['args']['failure_probability'] = .05
    # initExp['args']['R'] = 1.
    # initExp['args']['rating_scale'] = {'labels': [{'label':'yes', 'reward': 1},
    #                                              {'label':'no', 'reward': -1}]}
    initExp['args']['participant_to_algorithm_management'] = 'one_to_one'
    initExp['args']['algorithm_management_settings'] = algorithm_management_settings
    initExp['args']['alg_list'] = alg_list
    initExp['args']['num_tries'] = 50
    initExp['args']['d'] = 2

    target_locations = []
    with open(targets_filename) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader.fieldnames):
            target_locations.append(row)

    with open(context_filename) as f:
        features = np.load(f)

    targetset = []

    for i, row in enumerate(target_locations):
        print(row)
        targetset.append({'target_id': str(i),
                          'primary_type': 'image',
                          'primary_description': 'http://' + row,
                          'context': str(features[i,:]),
                          'alt_type': 'text',
                          'alt_description': 'Image {}'.format(i)})
    initExp['args']['targets'] = {'targetset': targetset}

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
    print(args['<context_file>'])

    print(args, '\n')
    launch(targets_filename=args['<target_filename>'], context_filename=args['<context_file>'],
           upload=upload)
