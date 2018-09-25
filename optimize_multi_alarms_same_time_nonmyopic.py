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

from hyperopt import Trials, STATUS_OK,STATUS_FAIL, tpe, fmin, hp
import hyperopt
from multiprocessing import Process as Process


def calculate_cost(x, costs):
    return costs[int(x['prediction']), int(x['actual'])](x)

def evaluate_model_cost(args):
    conf_threshold = args['conf_threshold']
    c_action = args['c_action']
    c_miss = args['c_miss']
    c_com = args['c_com']
    myopic_param = args['myopic_param']

    only_alarm1 = False



    #conf_threshold[0] == alarm2 && conf_threshold[1] == alarm1
    if conf_threshold[0] > conf_threshold[1]:
        only_alarm1 = True
        #return {'loss': 99999999999999.9, 'status': STATUS_FAIL, 'model': pd.DataFrame()}

    c_action2 = c_action * 1.2
    c_com2 = c_com * 0.5

    costs = np.matrix([[lambda x: 0,
                        lambda x: c_miss],
                       [lambda x: c_action2 + c_com2,
                        lambda x: c_action2
                        ],
                       [lambda x: c_action + c_com,
                        lambda x: c_action
                        ]])

    # trigger alarms according to conf_threshold
    dt_final = pd.DataFrame()
    unprocessed_case_ids = set(dt_preds.case_id.unique())
    case_counter = pd.DataFrame()
    case_counter["case_id"] = dt_preds.case_id.unique()
    case_counter["counter"] = 0
    for nr_events in range(1, dt_preds.prefix_nr.max() + 1):
        counter_alarms = 0
        if only_alarm1:
            counter_alarms = 1
            threshold = conf_threshold[1]
            tmp = dt_preds[(dt_preds.case_id.isin(unprocessed_case_ids)) & (dt_preds.prefix_nr == nr_events)]
            tmp = tmp[tmp.predicted_proba >= threshold]
            tmp["prediction"] = 1 + counter_alarms
            case_counter.loc[case_counter.case_id.isin(tmp.case_id), ['counter']] = case_counter["counter"] + 1
            tmp_case_counter = case_counter[case_counter.counter > myopic_param]
            tmp = tmp[tmp.case_id.isin(tmp_case_counter.case_id)]
            dt_final = pd.concat([dt_final, tmp], axis=0)
            unprocessed_case_ids = unprocessed_case_ids.difference(tmp.case_id)
        else:
            tmp = dt_preds[(dt_preds.case_id.isin(unprocessed_case_ids)) & (dt_preds.prefix_nr == nr_events)]
            tmp = tmp[tmp.predicted_proba >= min(conf_threshold)]
            case_counter.loc[case_counter.case_id.isin(tmp.case_id), ['counter']] = case_counter["counter"] + 1
            for threshold in conf_threshold:
                tmp = dt_preds[(dt_preds.case_id.isin(unprocessed_case_ids)) & (dt_preds.prefix_nr == nr_events)]
                tmp = tmp[tmp.predicted_proba >= threshold]
                tmp["prediction"] = 1 + counter_alarms
                tmp_case_counter = case_counter[case_counter.counter > myopic_param]
                tmp = tmp[tmp.case_id.isin(tmp_case_counter.case_id)]
                dt_final = pd.concat([dt_final, tmp], axis=0)
                unprocessed_case_ids = unprocessed_case_ids.difference(tmp.case_id)
                counter_alarms = counter_alarms + 1
    tmp = dt_preds[(dt_preds.case_id.isin(unprocessed_case_ids)) & (dt_preds.prefix_nr == 1)]
    tmp["prediction"] = 0
    dt_final = pd.concat([dt_final, tmp], axis=0)

    case_lengths = dt_preds.groupby("case_id").prefix_nr.max().reset_index()
    case_lengths.columns = ["case_id", "case_length"]
    dt_final = dt_final.merge(case_lengths)

    cost = dt_final.apply(calculate_cost, costs=costs, axis=1).sum()

    return {'loss': cost, 'status': STATUS_OK, 'model': dt_final}


def run_experiment(c_miss_weight, c_action_weight, c_com_weight, early_type):


    conf_thresholds = []
    for i in range(0,2):
        string_conf_threshold = "conf_threshold" + str(i)
        conf_thresholds.append(hp.uniform(string_conf_threshold, 0, 1))

    c_miss = c_miss_weight / (c_miss_weight + c_action_weight + c_com_weight)
    c_action = c_action_weight / (c_miss_weight + c_action_weight + c_com_weight)
    c_com = c_com_weight / (c_miss_weight + c_action_weight + c_com_weight)


    space = {'conf_threshold': conf_thresholds,
             'c_action': c_action,
             'c_miss': c_miss,
             'c_com': c_com}
 #   trials = Trials()
 #   best = fmin(evaluate_model_cost, space, algo=tpe.suggest, max_evals=1200, trials=trials)
    best_threshold_alarm1 = 0.0
    best_threshold_alarm2 = 0.0
    best_cost = 99999999999999.9
    best_myopic = 0

    for threshold_alarm1 in range(50,101):
        for threshold_alarm2 in range(threshold_alarm1-1,101):
            for myopic_param in range(0,6):
                conf_thresholds[0] = threshold_alarm2 * 0.01
                conf_thresholds[1] = threshold_alarm1 * 0.01
                args = {'conf_threshold': conf_thresholds,
                 'c_action': c_action,
                 'c_miss': c_miss,
                 'c_com': c_com,
                 'myopic_param': myopic_param

                }
                cost = evaluate_model_cost(args)
                if cost['loss'] < best_cost:
                    best_threshold_alarm1 = conf_thresholds[1]
                    best_threshold_alarm2 = conf_thresholds[0]
                    best_cost = cost['loss']
                    best_myopic = myopic_param
                    print("Best myopic" + str(best_myopic))


    conf_thresholds[0] = best_threshold_alarm2
    conf_thresholds[1] = best_threshold_alarm1



    best_params=  {'conf_threshold': conf_thresholds,
             'c_action': c_action,
             'c_miss': c_miss,
             'c_com': c_com,
            'cost_training': best_cost,
            'myopic_param': best_myopic
    }

    #best_params = hyperopt.space_eval(space, best)
    #best_cost = evaluate_model_cost(best_params)

    #best_params['cost_training'] = best_cost


    outfile = os.path.join(params_dir, "optimal_confs_%s_%s_%s_%s_%s_%s.pickle" % (
        dataset_name, c_miss_weight, c_action_weight, c_postpone_weight, c_com_weight, early_type))
    # write to file
    with open(outfile, "wb") as fout:
        print(outfile)
        print(repr(best_params))
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