import pandas as pd
from sys import argv
import pickle
import os
from DatasetManager import DatasetManager


dataset_name = argv[1]
optimal_alarm_file = argv[2]

dt_preds = pd.read_csv(optimal_alarm_file, sep="\t")
out_filename = optimal_alarm_file.replace("optimal_alarm_","filtered_events_")

# read the data
dataset_manager = DatasetManager(dataset_name)
data = dataset_manager.read_dataset()

dt_final = pd.DataFrame()

dt_final = data[(data["Case ID"].isin(dt_preds["case_id"])) & (data["event_nr"].isin(dt_preds["prefix_nr"]))]
dt_final.to_csv(out_filename, sep=';')

