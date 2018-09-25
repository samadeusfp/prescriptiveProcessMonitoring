import EncoderFactory
from DatasetManager import DatasetManager

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion

import time
import os
import sys
from sys import argv
import pickle
import csv

from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
import hyperopt
from multiprocessing import Process as Process


def get_min_nonmonotonic(dataSetName):
    if str.startswith(dataSetName, "traffic_fines"):
        return 3
    elif str.startswith(dataSetName, "bpic2017"):
        return 18


def calculate_cost(x, costs):
    return costs[int(x['prediction']), int(x['actual'])](x)

def get_max_nonmonotonic(dataSetName):
    if str.startswith(dataSetName,"traffic_fines"):
        return 2
    elif str.startswith(dataSetName,"bpic2017"):
        return 12


def evaluate_model_cost(args):
    conf_threshold = args['conf_threshold']
    c_action = args['c_action']
    c_miss = args['c_miss']
    c_com = args['c_com']
    event_threshold = args['event_threshold']

    if early_type == "linear":
        costs = np.matrix([[lambda x: 0,
                            lambda x: c_miss],
                           [lambda x: c_action * (x['prefix_nr'] - 1) / x['case_length'] + c_com,
                            lambda x: c_action * (x['prefix_nr'] - 1) / x['case_length'] + (x['prefix_nr'] - 1) / x[
                                'case_length'] * c_miss
                            ]])
    elif early_type == "nonmonotonic":
        costs = np.matrix([[lambda x: 0,
                                        lambda x: c_miss],
                                       [lambda x: c_action * (
                                                   x['case_length'] - min(x['prefix_nr'], nonmonotonic_threshold) / x[
                                               'case_length']) + (c_com * (
                                                   x['case_length'] - x['prefix_nr'] / x['case_length'])),
                                        lambda x: c_action * (
                                                    x['case_length'] - min(x['prefix_nr'], nonmonotonic_threshold) / x[
                                                'case_length']) + max((
                                                                              x['prefix_nr'] - 1) / x['case_length'],
                                                                      max_nonmonotonic_threshold) * c_miss
                                        ]])
    else:
        costs = np.matrix([[lambda x: 0,
                            lambda x: c_miss],
                           [lambda x: c_action + c_com,
                            lambda x: c_action + (x['prefix_nr'] - 1) / x['case_length'] * c_miss
                            ]])

    # trigger alarms according to conf_threshold
    dt_final = pd.DataFrame()
    unprocessed_case_ids = set(dt_preds.case_id.unique())
    for nr_events in range(1, dt_preds.prefix_nr.max() + 1):
        tmp = dt_preds[(dt_preds.case_id.isin(unprocessed_case_ids)) & (dt_preds.prefix_nr == nr_events) & (dt_preds.Activity == event_threshold)]
        tmp = tmp[tmp.predicted_proba >= conf_threshold]
        tmp["prediction"] = 1
        dt_final = pd.concat([dt_final, tmp], axis=0)
        unprocessed_case_ids = unprocessed_case_ids.difference(tmp.case_id)
    tmp = dt_preds[(dt_preds.case_id.isin(unprocessed_case_ids)) & (dt_preds.prefix_nr == 1)]
    tmp["prediction"] = 0
    dt_final = pd.concat([dt_final, tmp], axis=0)

    case_lengths = dt_preds.groupby("case_id").prefix_nr.max().reset_index()
    case_lengths.columns = ["case_id", "case_length"]
    dt_final = dt_final.merge(case_lengths)

    cost = dt_final.apply(calculate_cost, costs=costs, axis=1).sum()

    return {'loss': cost, 'status': STATUS_OK, 'model': dt_final}


def run_experiment(c_miss_weight, c_action_weight, c_com_weight, early_type):
    c_miss = c_miss_weight / (c_miss_weight + c_action_weight + c_com_weight)
    c_action = c_action_weight / (c_miss_weight + c_action_weight + c_com_weight)
    c_com = c_com_weight / (c_miss_weight + c_action_weight + c_com_weight)

    space = {'conf_threshold': hp.uniform("conf_threshold", 0, 1),
             'c_action': c_action,
             'c_miss': c_miss,
             'c_com': c_com,
            'event_threshold': hp.choice("event_threshold", dt_preds.Activity.unique())}
    trials = Trials()
    best = fmin(evaluate_model_cost, space, algo=tpe.suggest, max_evals=500, trials=trials)

    best_params = hyperopt.space_eval(space, best)

    outfile = os.path.join(params_dir, "optimal_confs_%s_%s_%s_%s_%s_%s.pickle" % (
        dataset_name, c_miss_weight, c_action_weight, c_postpone_weight, c_com_weight, early_type))
    # write to file
    with open(outfile, "wb") as fout:
        pickle.dump(best_params, fout)


print('Preparing data...')
start = time.time()

dataset_name = argv[1]
preds_dir = argv[2]
params_dir = argv[3]

# create output directory
if not os.path.exists(os.path.join(params_dir)):
    os.makedirs(os.path.join(params_dir))

# read the data
dataset_manager = DatasetManager(dataset_name)

# prepare the dataset
dt_preds = pd.read_csv(os.path.join(preds_dir, "preds_val_%s.csv" % dataset_name), sep=";")

# set nonomonotic-threshold
nonmonotonic_threshold = get_min_nonmonotonic(dataset_name)
max_nonmonotonic_threshold = get_max_nonmonotonic(dataset_name)

print('Optimizing parameters...')
processes = []
cost_weights = [(1, 1), (2, 1), (3, 1), (5, 1), (10, 1), (20, 1), (40, 1)]
c_com_weights = [1 / 40.0, 1 / 20.0, 1 / 10.0, 1 / 5.0, 1 / 3.0, 1 / 2.0, 1, 2, 3, 5, 10, 20, 40, 0]
c_postpone_weight = 0
for c_miss_weight, c_action_weight in cost_weights:
    for c_com_weight in c_com_weights:
        for early_type in ["const", "linear", "nonmonotonic"]:
            p = Process(target=run_experiment, args=(c_miss_weight, c_action_weight, c_com_weight, early_type))
            p.start()
            processes.append(p)
for p in processes:
    p.join()

