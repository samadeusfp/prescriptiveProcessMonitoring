import EncoderFactory
from DatasetManager import DatasetManager

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
from sklearn.pipeline import FeatureUnion

import time
import os
import sys
import csv
from sys import argv
import pickle
from conf_constant_costfunctions import get_constant_costfunctions
from DatasetManager import DatasetManager


dataset_name = argv[1]
predictions_dir = argv[2]
conf_threshold_dir = argv[3]
results_dir = argv[4]

method = "alarm1"

# create results directory
if not os.path.exists(os.path.join(results_dir)):
    os.makedirs(os.path.join(results_dir))

dataset_manager = DatasetManager(dataset_name)
data = dataset_manager.read_dataset()

# load predictions
dt_preds = pd.read_csv(os.path.join(predictions_dir, "preds_train_%s.csv" % dataset_name), sep=";")



# set nonomonotic constants
aConst, bConst, cConst, dConst, eConst, fConst = get_constant_costfunctions(dataset_name)

cost_weights = [(10, 1), (10, 2), (10, 3), (10, 4), (10, 5)]
c_com_weights = [1, 2, 3, 4, 5, 10, 20, 30, 40]
c_postpone_weight = 0
for c_miss_weight, c_action_weight in cost_weights:
    for c_com_weight in c_com_weights:
        for early_type in ["const"]:

              # load the optimal confidence threshold
            conf_file = os.path.join(conf_threshold_dir, "optimal_confs_%s_%s_%s_%s_%s_%s.pickle" % (
               dataset_name, c_miss_weight, c_action_weight, c_postpone_weight, c_com_weight, early_type))

            with open(conf_file, "rb") as fin:
                conf_threshold = pickle.load(fin)['conf_threshold']

            # trigger alarms according to conf_threshold
            dt_final = pd.DataFrame()
            unprocessed_case_ids = set(dt_preds.case_id.unique())
            for nr_events in range(1, dt_preds.prefix_nr.max() + 1):
                tmp = dt_preds[(dt_preds.case_id.isin(unprocessed_case_ids)) & (dt_preds.prefix_nr == nr_events)]
                tmp = tmp[tmp.predicted_proba >= conf_threshold]
                tmp1 = tmp[tmp.actual == 1]
                tmp2 = tmp[tmp.actual == 0]
                tmp1["prediction"] = 1
                tmp2["prediction"] = 2
                dt_final = pd.concat([dt_final, tmp1], axis=0)
                dt_final = pd.concat([dt_final, tmp2], axis=0)
                unprocessed_case_ids = unprocessed_case_ids.difference(tmp.case_id)
            dt_events = pd.DataFrame()
            dt_events = data[(data["Case ID"].isin(dt_final["case_id"])) & (data["event_nr"].isin(dt_final["prefix_nr"]))]

                # write results to file
            out_filename = os.path.join(results_dir, "filtered_events_%s_%s_%s_%s_%s_%s_%s.csv" % (dataset_name, method, c_miss_weight, c_action_weight, c_postpone_weight, c_com_weight, early_type))

            dt_events.to_csv(out_filename, sep=';')