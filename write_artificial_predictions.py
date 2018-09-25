import EncoderFactory
from DatasetManager import DatasetManager

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion

from sklearn.ensemble import RandomForestClassifier

import random
import time
import os
import sys
from sys import argv
import pickle
from multiprocessing import Pool as ThreadPool
from multiprocessing import Process as Process


def write_test_set_predictions(preds, y_test, dt_test_prefixes, dataset_name, dataset_manager, results_dir):
    print("write test set predictions")
    dt_preds = pd.DataFrame({"predicted_proba": preds, "actual": y_test,
                             "prefix_nr": dt_test_prefixes.groupby(dataset_manager.case_id_col).first()["prefix_nr"],
                             "case_id": dt_test_prefixes.groupby(dataset_manager.case_id_col).first()["orig_case_id"]})

    dt_preds.to_csv(os.path.join(results_dir, "preds_%s.csv" % dataset_name), sep=";", index=False)
    print("write test set predictions finished")


def write_error_for_prefix_length(dataset_name, min_prefix_length, max_prefix_length, dt_preds_val, results_dir):
    print("write errors for every prefix length")
    with open(os.path.join(results_dir, "errors_%s.csv" % dataset_name), 'w') as fout:
        fout.write("dataset;prefix_nr;mean_error;std_error\n")

        for i in range(min_prefix_length, max_prefix_length + 1):
            tmp = dt_preds_val[dt_preds_val.prefix_nr == i]
            mean = np.mean(tmp.actual - tmp.predicted_proba)
            std = np.std(tmp.actual - tmp.predicted_proba)
            fout.write("%s;%s;%s;%s\n" % (dataset_name, i, mean, std))
    print("write errors for every prefix length finished")


def write_auc(dataset_name, min_prefix_length, max_prefix_length, dt_preds, results_dir):
    print("write AUC for every prefix length")
    with open(os.path.join(results_dir, "results_%s.csv" % dataset_name), 'w') as fout:
        fout.write("dataset;nr_events;auc\n")

        for i in range(min_prefix_length, max_prefix_length + 1):
            tmp = dt_preds[dt_preds.prefix_nr == i]
            if len(tmp.actual.unique()) > 1:
                auc = roc_auc_score(tmp.actual, tmp.predicted_proba)
                fout.write("%s;%s;%s\n" % (dataset_name, i, auc))
    print("write AUC for every prefix length finished")


def write_delta(min_prefix_length, max_prefix_length, dataset_name, dt_preds_val, results_dir):
    print("write deltas for every prefix length")
    with open(os.path.join(results_dir, "deltas_%s.csv" % dataset_name), 'w') as fout:
        mean_cols = ["mean_delta_%s" % i for i in range(min_prefix_length, max_prefix_length)]
        std_cols = ["std_delta_%s" % i for i in range(min_prefix_length, max_prefix_length)]
        fout.write("dataset;prefix_nr;%s;%s\n" % (";".join(mean_cols), ";".join(std_cols)))

        for k in range(min_prefix_length, max_prefix_length):
            tmp_k = dt_preds_val[dt_preds_val.prefix_nr == k]
            means = []
            stds = []
            for i in range(k + 1, max_prefix_length + 1):
                tmp_i = dt_preds_val[dt_preds_val.prefix_nr == i]
                tmp_merged = tmp_k.merge(tmp_i, on="case_id", suffixes=["_k", "_i"])
                mean = np.mean(tmp_merged.predicted_proba_i - tmp_merged.predicted_proba_k)
                std = np.std(tmp_merged.predicted_proba_i - tmp_merged.predicted_proba_k)
                means.append(mean)
                stds.append(std)
            for i in range(k - 1):
                means.append(mean)
                stds.append(std)
            fout.write("%s;%s;%s;%s\n" % (
                dataset_name, k, ";".join([str(val) for val in means]), ";".join([str(val) for val in stds])))
    print("write deltas for every prefix length finished")


def predict_proba(X, Y, accuracy):
    results = []
    cases = []
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        if y == 1:
            if (accuracy * 10) >= random.randint(1, 10):
                results.append(random.randint(40, 100) / 100)
            else:
                results.append(random.randint(0, 40) / 100)
        else:
            if (accuracy * 10) <= random.randint(1, 10):
                results.append(random.randint(40, 100) / 100)
            else:
                results.append(random.randint(0, 40) / 100)
    return results


def do_prediction(dataset_name, results_dir, accuracy):
    split_type = "temporal"

    train_ratio = 0.8
    val_ratio = 0.2

    print('Preparing data...')
    start = time.time()

    # read the data
    dataset_manager = DatasetManager(dataset_name)
    data = dataset_manager.read_dataset()

    min_prefix_length = 1
    max_prefix_length = int(np.ceil(data.groupby(dataset_manager.case_id_col).size().quantile(0.9)))

    cls_encoder_args = {'case_id_col': dataset_manager.case_id_col,
                        'static_cat_cols': dataset_manager.static_cat_cols,
                        'static_num_cols': dataset_manager.static_num_cols,
                        'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                        'dynamic_num_cols': dataset_manager.dynamic_num_cols,
                        'fillna': True}

    print(time.time() - start)
    print("split into training and test")
    if split_type == "temporal":
        train, test = dataset_manager.split_data_strict(data, train_ratio, split=split_type)
    else:
        train, test = dataset_manager.split_data(data, train_ratio, split=split_type)

    train, val = dataset_manager.split_val(train, val_ratio)

    # generate data where each prefix is a separate instance
    dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length)
    dt_val_prefixes = dataset_manager.generate_prefix_data(val, min_prefix_length, max_prefix_length)
    dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)

    print(time.time() - start)
    print("encode all prefixes")
    # encode all prefixes
    feature_combiner = FeatureUnion(
        [(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in ["static", "agg"]], n_jobs=-1)

    arrayPrefixes = [dt_train_prefixes, dt_test_prefixes, dt_val_prefixes]
    X_train = feature_combiner.fit_transform(dt_train_prefixes)
    X_test = feature_combiner.fit_transform(dt_test_prefixes)
    X_val = feature_combiner.fit_transform(dt_val_prefixes)

    y_train = dataset_manager.get_label_numeric(dt_train_prefixes)
    y_test = dataset_manager.get_label_numeric(dt_test_prefixes)
    y_val = dataset_manager.get_label_numeric(dt_val_prefixes)

    print(time.time() - start)
    print("get predictions for test set")
    # get predictions for test set
    preds_train = predict_proba(X_train, y_train, accuracy)
    preds_val = predict_proba(X_val, y_val, accuracy)
    preds = predict_proba(X_test, y_test, accuracy)

    print(time.time() - start)
    print("write train-val set predictions")
    # write train-val set predictions
    dt_preds = pd.DataFrame({"predicted_proba": preds_train, "actual": y_train,
                             "prefix_nr": dt_train_prefixes.groupby(dataset_manager.case_id_col).first()["prefix_nr"],
                             "case_id": dt_train_prefixes.groupby(dataset_manager.case_id_col).first()["orig_case_id"]})
    dt_preds_val = pd.DataFrame({"predicted_proba": preds_val, "actual": y_val,
                                 "prefix_nr": dt_val_prefixes.groupby(dataset_manager.case_id_col).first()["prefix_nr"],
                                 "case_id": dt_val_prefixes.groupby(dataset_manager.case_id_col).first()[
                                     "orig_case_id"]})
    # dt_preds = pd.concat([dt_preds, dt_preds_val], axis=0)
    dt_preds.to_csv(os.path.join(results_dir, "preds_train_%s.csv" % dataset_name), sep=";", index=False)
    dt_preds_val.to_csv(os.path.join(results_dir, "preds_val_%s.csv" % dataset_name), sep=";", index=False)

    print(time.time() - start)
    # write test set predictions
    pWriteTestSetPredictions = Process(target=write_test_set_predictions,
                                       args=(
                                       preds, y_test, dt_test_prefixes, dataset_name, dataset_manager, results_dir))
    pWriteTestSetPredictions.start()

    print(time.time() - start)
    # write AUC for every prefix length
    pWriteAuc = Process(target=write_auc,
                        args=(dataset_name, min_prefix_length, max_prefix_length, dt_preds, results_dir))
    pWriteAuc.start()

    print(time.time() - start)
    # write errors for every prefix length
    pWrite_error_for_prefix_length = Process(target=write_error_for_prefix_length,
                                             args=(dataset_name, min_prefix_length, max_prefix_length, dt_preds_val,
                                                   results_dir))
    pWrite_error_for_prefix_length.start()

    print(time.time() - start)
    # write deltas for every prefix length
    pWriteDelta = Process(target=write_delta,
                          args=(min_prefix_length, max_prefix_length, dataset_name, dt_preds_val, results_dir))
    pWriteDelta.start()

    print(time.time() - start)


dataset_name = argv[1]
results_dir = argv[2]

positiveMultiplierNextEvent = 1.05
negativeMultiplierNextEvent = 0.95

for accuracy in range(5, 10):
    accuracy = accuracy / 10
    result_dir_acc = results_dir + str(accuracy)
    # create results directory
    if not os.path.exists(os.path.join(result_dir_acc)):
        os.makedirs(os.path.join(result_dir_acc))
    pDoPreds = Process(target=do_prediction, args=(dataset_name, result_dir_acc, accuracy))
    pDoPreds.start()
