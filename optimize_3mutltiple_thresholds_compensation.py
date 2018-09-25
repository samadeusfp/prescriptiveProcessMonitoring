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



def calculate_expected_costs(x, costs, alarm):
    return ((x.predicted_proba) * costs[alarm,1](x)) + ((1 - x.predicted_proba) * costs[alarm,0](x))

def get_min_nonmonotonic(dataSetName):
    if str.startswith(dataSetName,"traffic_fines"):
        return 3
    elif str.startswith(dataSetName,"bpic2017"):
        return 18

def get_max_nonmonotonic(dataSetName):
    if str.startswith(dataSetName,"traffic_fines"):
        return 2
    elif str.startswith(dataSetName,"bpic2017"):
        return 12

def calculate_cost(x, costs):
    actual = int(x['actual'])
    alarmArray = []
    if int(x['prediction']) == 1:
        if x["alarm1"] == 1:
            alarmArray.append(1)
        if x["alarm2"] == 1:
            alarmArray.append(2)
        if x["alarm3"] == 1:
            alarmArray.append(3)
        noAlarmSelected = True
        selectedAlarm = 0
        for alarm in alarmArray:
            if noAlarmSelected:
                selectedAlarm = alarm
                noAlarmSelected = False
            elif calculate_expected_costs(x,costs,alarm) < calculate_expected_costs(x,costs,selectedAlarm):
                selectedAlarm = alarm
        optimalCosts = costs[selectedAlarm, actual](x)
    else:
        optimalCosts = costs[0, actual](x)
    return optimalCosts


def evaluate_model_cost(args):
    conf_threshold_1 = args['conf_threshold_1']
    conf_threshold_2 = args['conf_threshold_2']
    conf_threshold_3 = args['conf_threshold_3']
    c_action = args['c_action']
    c_miss = args['c_miss']
    c_com = args['c_com']


    if early_type == "linear":
        costs = np.matrix([[lambda x: 0,
                            lambda x: c_miss],
                           [lambda x: c_action * (x['prefix_nr'] - 1) / x['case_length'] + c_com,
                            lambda x: c_action * (x['prefix_nr'] - 1) / x['case_length'] + (
                                    x['prefix_nr'] - 1) / x['case_length'] * c_miss
                            ],
                           [lambda x: c_action * 3 * (x['prefix_nr'] - 1) / x['case_length'] + c_com / 2,
                            lambda x: c_action * 3 * (x['prefix_nr'] - 1) / x['case_length'] + (
                                    x['prefix_nr'] - 1) / x['case_length'] * c_miss
                            ],
                           [lambda x: ((c_action / 2) * (x['prefix_nr'] - 1) / x['case_length']) + c_com * 1.5,  # 0:2
                            lambda x: ((c_action / 2) * (x['prefix_nr'] - 1) / x['case_length']) + (x['prefix_nr'] - 1) / x['case_length'] * c_miss
                            ]
                           ])
    elif early_type == "nonmonotonic":
        costs = np.matrix([[lambda x: 0,
                            lambda x: c_miss],
                           [lambda x: c_action * (x['case_length'] - min(x['prefix_nr'], nonmonotonic_threshold)/ x['case_length']) + (c_com * (x['case_length'] - x['prefix_nr']/ x['case_length'])),
                            lambda x: c_action * (x['case_length'] - min(x['prefix_nr'], nonmonotonic_threshold)/ x['case_length']) + max((
                                    x['prefix_nr'] - 1) / x['case_length'],max_nonmonotonic_threshold) * c_miss
                            ],
                           [lambda x: c_action * (x['case_length'] - min(x['prefix_nr'], nonmonotonic_threshold)/ x['case_length']) * 3 + (c_com * (x['case_length'] - x['prefix_nr']/ x['case_length'])) / 2,  # 0:2
                            lambda x: c_action * (x['case_length'] - min(x['prefix_nr'], nonmonotonic_threshold)/ x['case_length']) * 3 + max((
                                    x['prefix_nr'] - 1) / x['case_length'],max_nonmonotonic_threshold) * c_miss
                            ],
                           [lambda x: (c_action * (x['case_length'] - min(x['prefix_nr'], nonmonotonic_threshold)/ x['case_length']) / 2) + (c_com * (x['case_length'] - x['prefix_nr']/ x['case_length'])) * 1.5,  # 0:2
                            lambda x: (c_action * (x['case_length'] - min(x['prefix_nr'], nonmonotonic_threshold)/ x['case_length']) / 2) + max((
                                    x['prefix_nr'] - 1) / x['case_length'],max_nonmonotonic_threshold) * c_miss
                            ]
                           ])
    else:
        costs = np.matrix([[lambda x: 0,
                            lambda x: c_miss],
                           [lambda x: c_action + c_com,  # 0:1
                            lambda x: c_action + (x['prefix_nr'] - 1) / x['case_length'] * c_miss
                            ],
                           [lambda x: c_action * 3 + c_com / 2,  # 0:2
                            lambda x: c_action * 3 + (x['prefix_nr'] - 1) / x['case_length'] * c_miss
                            ],
                           [lambda x: (c_action/2) + c_com*1.5,  # 0:2
                            lambda x: (c_action/2) + (x['prefix_nr'] - 1) / x['case_length'] * c_miss
                            ]
                           ])

    # trigger alarms according to conf_threshold
    dt_final = pd.DataFrame()
    unprocessed_case_ids_alarm1 = set(dt_preds.case_id.unique())
    unprocessed_case_ids_alarm2 = set(dt_preds.case_id.unique())
    unprocessed_case_ids_alarm3 = set(dt_preds.case_id.unique())
    unprocessed_case_ids = set(dt_preds.case_id.unique())
    #alarm3
    for nr_events in range(1, dt_preds.prefix_nr.max() + 1):
        tmp = dt_preds[(dt_preds.case_id.isin(unprocessed_case_ids_alarm3)) & (dt_preds.prefix_nr == nr_events)]
        tmp = tmp[tmp.predicted_proba >= conf_threshold_3]
        tmp["prediction"] = 1
        tmp["alarm3"] = 1
        dt_final = pd.concat([dt_final, tmp], axis=0)
        unprocessed_case_ids_alarm3 = unprocessed_case_ids_alarm3.difference(tmp.case_id)
        unprocessed_case_ids = unprocessed_case_ids.difference(tmp.case_id)
    #alarm2
    for nr_events in range(1, dt_preds.prefix_nr.max() + 1):
        tmp = dt_preds[(dt_preds.case_id.isin(unprocessed_case_ids_alarm2)) & (dt_preds.prefix_nr == nr_events)]
        tmp = tmp[tmp.predicted_proba >= conf_threshold_2]
        tmp["prediction"] = 1
        tmp["alarm2"] = 1
        dt_final = pd.concat([dt_final, tmp], axis=0)
        unprocessed_case_ids_alarm2 = unprocessed_case_ids_alarm2.difference(tmp.case_id)
        unprocessed_case_ids = unprocessed_case_ids.difference(tmp.case_id)
    #alarm1
    for nr_events in range(1, dt_preds.prefix_nr.max() + 1):
        tmp = dt_preds[(dt_preds.case_id.isin(unprocessed_case_ids_alarm1)) & (dt_preds.prefix_nr == nr_events)]
        tmp = tmp[tmp.predicted_proba >= conf_threshold_1]
        tmp["prediction"] = 1
        tmp["alarm1"] = 1
        dt_final = pd.concat([dt_final, tmp], axis=0)
        unprocessed_case_ids_alarm1 = unprocessed_case_ids_alarm1.difference(tmp.case_id)
        unprocessed_case_ids = unprocessed_case_ids.difference(tmp.case_id)
    tmp = dt_preds[(dt_preds.case_id.isin(unprocessed_case_ids)) & (dt_preds.prefix_nr == 1)]
    tmp["prediction"] = 0
    dt_final = pd.concat([dt_final, tmp], axis=0)

    case_lengths = dt_preds.groupby("case_id").prefix_nr.max().reset_index()
    case_lengths.columns = ["case_id", "case_length"]
    dt_final = dt_final.merge(case_lengths)

    cost = dt_final.apply(calculate_cost, costs=costs, axis=1).sum()

    return {'loss': cost, 'status': STATUS_OK, 'model': dt_final}

def run_experiment(c_miss_weight,c_action_weight,c_com_weight,early_type):
    c_miss = c_miss_weight / (c_miss_weight + c_action_weight + c_com_weight)
    c_action = c_action_weight / (c_miss_weight + c_action_weight + c_com_weight)
    c_action = c_action_weight / (c_miss_weight + c_action_weight + c_com_weight)
    c_com = c_com_weight / (c_miss_weight + c_action_weight + c_com_weight)


    space = {'conf_threshold_1': hp.uniform("conf_threshold_1", 0, 1),
             'conf_threshold_2': hp.uniform("conf_threshold_2", 0, 1),
             'conf_threshold_3': hp.uniform("conf_threshold_3", 0, 1),
             'c_action': c_action,
             'c_miss': c_miss,
             'c_com': c_com}
    trials = Trials()
    best = fmin(evaluate_model_cost, space, algo=tpe.suggest, max_evals=100, trials=trials)

    best_params = hyperopt.space_eval(space, best)

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

#set nonomonotic-threshold
nonmonotonic_threshold = get_min_nonmonotonic(dataset_name)
max_nonmonotonic_threshold = get_max_nonmonotonic(dataset_name)



print('Optimizing parameters...')
cost_weights = [(1, 1), (2, 1), (3, 1), (5, 1), (10, 1), (20, 1), (40, 1)]
c_com_weights = [1 / 40.0, 1 / 20.0, 1 / 10.0, 1 / 5.0, 1 / 3.0, 1 / 2.0, 1, 2, 3, 5, 10, 20, 40, 0]
#cost_weights = [(10, 1)]
#c_com_weights = [2]
c_postpone_weight = 0
processes = []
for c_miss_weight, c_action_weight in cost_weights:
    for c_com_weight in c_com_weights:
        for early_type in ["linear", "const","nonmonotonic"]:
             p = Process(target=run_experiment,args=(c_miss_weight, c_action_weight, c_com_weight, early_type))
             p.start()
             processes.append(p)
for p in processes:
    p.join()






