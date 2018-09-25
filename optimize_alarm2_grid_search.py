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

from conf_constant_costfunctions import get_constant_costfunctions

from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
import hyperopt
from multiprocessing import Process as Process


def calculate_cost(x, costs):
    return costs[int(x['prediction']), int(x['actual'])](x)


def evaluate_model_cost(args):
    conf_threshold = args['conf_threshold']
    c_action = args['c_action']
    c_miss = args['c_miss']
    c_com = args['c_com']

    c_action = c_action * 1.2
    c_com = c_com * 0.5

    costs = np.matrix([[lambda x: 0,
                        lambda x: c_miss],
                       [lambda x: c_action + c_com,
                        lambda x: c_action
                        ]])

    # trigger alarms according to conf_threshold
    dt_final = pd.DataFrame()
    unprocessed_case_ids = set(dt_preds.case_id.unique())
    for nr_events in range(1, dt_preds.prefix_nr.max() + 1):
        tmp = dt_preds[(dt_preds.case_id.isin(unprocessed_case_ids)) & (dt_preds.prefix_nr == nr_events)]
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
             'c_com': c_com}
    #trials = Trials()
    #best = fmin(evaluate_model_cost, space, algo=tpe.suggest, max_evals=50, trials=trials)

    best_threshold_alarm1 = 0.0
    best_cost = 99999999999999.9

    for threshold_alarm1 in range(0,101):
        conf_threshold = threshold_alarm1 * 0.01
        args = {'conf_threshold': conf_threshold,
             'c_action': c_action,
             'c_miss': c_miss,
             'c_com': c_com
        }
        cost = evaluate_model_cost(args)
        if cost['loss'] < best_cost:
            best_threshold_alarm1 = conf_threshold
            best_cost = cost['loss']

    best_params=  {'conf_threshold': best_threshold_alarm1,
             'c_action': c_action,
             'c_miss': c_miss,
             'c_com': c_com
        }

    #best_params = hyperopt.space_eval(space, best)
    #best_cost = evaluate_model_cost(best_params)

    best_params['cost_training'] = best_cost

    #best_params = hyperopt.space_eval(space, best)
    best_cost = evaluate_model_cost(best_params)

    best_params['cost_training'] = best_cost['loss']

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

# set nonomonotic constants
aConst, bConst, cConst, dConst, eConst, fConst = get_constant_costfunctions(dataset_name)

print('Optimizing parameters...')
processes = []
cost_weights = [(10, 1), (10, 2), (10, 3), (10, 4), (10, 5)]
c_com_weights = [1, 2, 3, 4, 5, 10, 20, 30, 40]
c_postpone_weight = 0
for c_miss_weight, c_action_weight in cost_weights:
    for c_com_weight in c_com_weights:
        for early_type in ["const"]:
            p = Process(target=run_experiment, args=(c_miss_weight, c_action_weight, c_com_weight, early_type))
            p.start()
            processes.append(p)
for p in processes:
    p.join()
