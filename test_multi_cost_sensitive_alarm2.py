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
import xgboost as xgb


def calculate_cost(x, costs):
    return costs[int(x['prediction']), int(x['actual'])](x)


def calculate_cost_baseline(x, costs):
    return costs[0, int(x['actual'])](x)


def cost_sensitive_eval(y_predicted, y_true):
    labels = y_true.get_label()
    # return a pair metric_name, result
    #false negative: abs(c_in_alarm2 - c_in_alarm1)
    #                   abs(1.2*2-2)
    fn = sum((labels >= 0.5) & (y_predicted < 0.5)) * cost_false_negative
    #false positive abs((c_in_alarm2 + c_com_alarm2) - (c_in_alarm1 + c_com_alarm2))
    #                   1.2 + 5                     -       1       +   10
    fp = sum((labels <= 0.5) & (y_predicted > 0.5)) * cost_false_positive
    return 'cost', (fn + fp)

dataset_name = argv[1]
predictions_dir = argv[2]
conf_threshold_dir = argv[3]
train_data_dir = argv[4]
optimal_params_dir = argv[5]
results_dir = argv[6]

trial_nr = 1

method = "cost_sensitive_multi"

# create results directory
if not os.path.exists(os.path.join(results_dir)):
    os.makedirs(os.path.join(results_dir))


# load predictions
dt_preds = pd.read_csv(os.path.join(predictions_dir, "preds_%s.csv" % dataset_name), sep=";")

# write results to file
out_filename = os.path.join(results_dir, "results_%s_%s.csv" % (dataset_name, method))

# set nonomonotic constants
aConst, bConst, cConst, dConst, eConst, fConst = get_constant_costfunctions(dataset_name)

with open(out_filename, 'w') as fout:
    writer = csv.writer(fout, delimiter=';', quotechar='', quoting=csv.QUOTE_NONE)
    writer.writerow(["dataset", "method", "metric", "value", "c_miss", "c_action", "c_postpone", "c_com", "early_type",
                     "threshold"])

    cost_weights = [(10, 1), (10, 2)]
    c_com_weights = [1, 2, 3, 4, 5, 10, 20, 30, 40]
    c_postpone_weight = 0
    alarm_type = ["alarm1","alarm2"]
    for c_miss_weight, c_action_weight in cost_weights:
        for c_com_weight in c_com_weights:
            for early_type in ["const"]:
                for v_alarm_type in alarm_type:

                    # Alarm2 instead of 1
                    cost_false_positive = float(c_action_weight) * 0.2
                    # Alarm 1 instead of 2
                    cost_false_negative = (float(c_action_weight) + float(c_com_weight)) - (
                                (1.2 * float(c_action_weight)) + (float(c_com_weight) * 0.5))

                    train_data_file = os.path.join(train_data_dir, "filtered_events_%s_%s_%s_%s_%s_%s_%s.csv" % (
                    dataset_name,v_alarm_type, c_miss_weight, c_action_weight, c_postpone_weight, c_com_weight, early_type))


                    optimal_params_filename = os.path.join(optimal_params_dir, "xgboost_params_%s_%s_%s_%s_%s_%s_%s.pickle" % (
                    dataset_name,v_alarm_type, c_miss_weight, c_action_weight, c_postpone_weight, c_com_weight, early_type))
                    print(optimal_params_filename)

                    if os.path.exists(optimal_params_filename):

                        ###############Initiate XGB for alarm choice#######
                        # read the data
                        dataset_manager = DatasetManager(dataset_name)

                        data = dataset_manager.read_dataset_file(train_data_file)

                        min_prefix_length = 1
                        max_prefix_length = int(np.ceil(data.groupby(dataset_manager.case_id_col).size().quantile(0.9)))

                        cls_encoder_args = {'case_id_col': dataset_manager.case_id_col,
                                            'static_cat_cols': dataset_manager.static_cat_cols,
                                            'static_num_cols': dataset_manager.static_num_cols,
                                            'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                                            'dynamic_num_cols': dataset_manager.dynamic_num_cols,
                                            'fillna': True}

                        train_ratio = 0.85
                        # generate data where each prefix is a separate instance
                        train, val = dataset_manager.split_val(data, train_ratio)
                        dt_train = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length)
                        dt_val = dataset_manager.generate_prefix_data(val, min_prefix_length, max_prefix_length)

                        # train the model with pre-tuned parameters
                        with open(optimal_params_filename, "rb") as fin:
                            best_params = pickle.load(fin)

                        # encode all prefixes

                        feature_combiner = FeatureUnion(
                            [(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in
                             ["static", "agg"]])

                        X_train = feature_combiner.fit_transform(dt_train)
                        y_train = np.array(dataset_manager.get_label_numeric(dt_train))
                        X_val = feature_combiner.fit_transform(dt_val)
                        y_val = np.array(dataset_manager.get_label_numeric(dt_val))

                        print(best_params)

                        cls = xgb.XGBClassifier(objective='binary:logistic',
                                                n_estimators=best_params['n_estimators'],
                                                learning_rate=best_params['learning_rate'],
                                                subsample=best_params['subsample'],
                                                max_depth=int(best_params['max_depth']),
                                                colsample_bytree=best_params['colsample_bytree'],
                                                min_child_weight=int(best_params['min_child_weight']),
                                                seed=22,
                                                n_jobs=-1)

                        eval_set = [(X_val, y_val)]
                        cls.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=10,
                                eval_metric=cost_sensitive_eval)
                        print("Done training classifier")

                        ###############End XGB for alarm choice#######

                        c_miss = c_miss_weight / (c_miss_weight + c_action_weight + c_com_weight)
                        c_action = c_action_weight / (c_miss_weight + c_action_weight + c_com_weight)
                        c_com = c_com_weight / (c_miss_weight + c_action_weight + c_com_weight)

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
                            tmp["prediction"] = 1
                            dt_final = pd.concat([dt_final, tmp], axis=0)
                            unprocessed_case_ids = unprocessed_case_ids.difference(tmp.case_id)
                        # read the data
                        dataset_manager_multi = DatasetManager(dataset_name)
                        data_multi = dataset_manager_multi.read_dataset()
                        dt_multi = pd.DataFrame()
                        dt_multi = data_multi[(data_multi["Case ID"].isin(dt_final["case_id"])) & (data_multi["event_nr"].isin(dt_final["prefix_nr"]))]
                        X_test = feature_combiner.fit_transform(dt_multi)
                        print("Feature Combiner sucessful")
                        dt_final["prediction"] = cls.predict(X_test) + 1
                        print(dt_final["prediction"].value_counts())
                        tmp = dt_preds[(dt_preds.case_id.isin(unprocessed_case_ids)) & (dt_preds.prefix_nr == 1)]
                        tmp["prediction"] = 0
                        dt_final = pd.concat([dt_final, tmp], axis=0)

                        case_lengths = dt_preds.groupby("case_id").prefix_nr.max().reset_index()
                        case_lengths.columns = ["case_id", "case_length"]
                        dt_final = dt_final.merge(case_lengths)

                        # calculate precision, recall etc.
                        prec, rec, fscore, _ = precision_recall_fscore_support(dt_final.actual, dt_final.prediction,
                                                                               pos_label=1, average="macro")
                        #tn, fp, fn, tp = confusion_matrix(dt_stasts.actual, dt_final.prediction).ravel()

                        # calculate earliness based on the "true alarms" only
                        tmp = dt_final[(dt_final.prediction == 1) & (dt_final.actual == 1)]
                        earliness = (1 - ((tmp.prefix_nr - 1) / tmp.case_length))
                        tmp = dt_final[(dt_final.prediction == 1)]
                        earliness_alarms = (1 - ((tmp.prefix_nr - 1) / tmp.case_length))

                        writer.writerow([dataset_name, method, "prec", prec, c_miss_weight, c_action_weight, c_postpone_weight,
                                         c_com_weight, early_type, conf_threshold])
                        writer.writerow(
                            [dataset_name, method, "rec", rec, c_miss_weight, c_action_weight, c_postpone_weight, c_com_weight,
                             early_type, conf_threshold])
                        writer.writerow(
                            [dataset_name, method, "fscore", fscore, c_miss_weight, c_action_weight, c_postpone_weight,
                             c_com_weight, early_type, conf_threshold])
                        # writer.writerow(
                        #     [dataset_name, method, "tn", tn, c_miss_weight, c_action_weight, c_postpone_weight, c_com_weight,
                        #      early_type, conf_threshold])
                        # writer.writerow(
                        #     [dataset_name, method, "fp", fp, c_miss_weight, c_action_weight, c_postpone_weight, c_com_weight,
                        #      early_type, conf_threshold])
                        # writer.writerow(
                        #     [dataset_name, method, "fn", fn, c_miss_weight, c_action_weight, c_postpone_weight, c_com_weight,
                        #      early_type, conf_threshold])
                        # writer.writerow(
                        #     [dataset_name, method, "tp", tp, c_miss_weight, c_action_weight, c_postpone_weight, c_com_weight,
                        #     early_type, conf_threshold])
                        writer.writerow(
                            [dataset_name, method, "earliness_mean", earliness.mean(), c_miss_weight, c_action_weight,
                             c_postpone_weight, c_com_weight, early_type, conf_threshold])
                        writer.writerow([dataset_name, method, "earliness_std", earliness.std(), c_miss_weight, c_action_weight,
                                         c_postpone_weight, c_com_weight, early_type, conf_threshold])
                        writer.writerow([dataset_name, method, "earliness_alarms_mean", earliness_alarms.mean(), c_miss_weight,
                                         c_action_weight, c_postpone_weight, c_com_weight, early_type, conf_threshold])
                        writer.writerow([dataset_name, method, "earliness_alarms_std", earliness_alarms.std(), c_miss_weight,
                                         c_action_weight, c_postpone_weight, c_com_weight, early_type, conf_threshold])

                        cost = dt_final.apply(calculate_cost, costs=costs, axis=1).sum()
                        writer.writerow([dataset_name, method, "cost", cost, c_miss_weight, c_action_weight, c_postpone_weight,
                                         c_com_weight, early_type, conf_threshold])
                        writer.writerow([dataset_name, method, "cost_avg", cost / len(dt_final), c_miss_weight, c_action_weight,
                                         c_postpone_weight, c_com_weight, early_type, conf_threshold])

                        cost_baseline = dt_final.apply(calculate_cost_baseline, costs=costs, axis=1).sum()
                        writer.writerow([dataset_name, method, "cost_baseline", cost_baseline, c_miss_weight, c_action_weight,
                                         c_postpone_weight, c_com_weight, early_type, conf_threshold])
                        writer.writerow(
                            [dataset_name, method, "cost_avg_baseline", cost_baseline / len(dt_final), c_miss_weight,
                             c_action_weight, c_postpone_weight, c_com_weight, early_type, conf_threshold])
