import numpy as np
import pickle 
import matplotlib.pyplot as plt
from tsml_eval.evaluation import evaluate_classifiers_by_problem
import pandas as pd
from sklearn.metrics import accuracy_score
from aeon.datasets import load_classification
from tsc_datasets import univariate_equal_length
from data_loader import load_datasets
from tsml_eval.evaluation.storage import load_classifier_results
from tsml_eval.experiments import (
    experiments,
    get_classifier_by_name,
    run_classification_experiment,
)


#clfs = ['DrCIF', 'RSTSF', 'QUANT', 'STSF', 'CIF', 'TSF' ]

datasets = list(univariate_equal_length)
datasets_dict = load_datasets(datasets)
clfs=[ "STSF", "CIF", "TSF", "DrCIF"]
for clf in clfs:

    classifier = get_classifier_by_name(clf)

    # record memory usage every 0.1 seconds, just here for notebook speed   

    # does not need to be changed for usage
    experiments.MEMRECORD_INTERVAL = 0.1

    for dataset_name, data in datasets_dict.items():

        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test'] # C:\Users\BOUKA\OneDrive\Bureau\CODE\tsml_eval

        run_classification_experiment( X_train, y_train, X_test, y_test, classifier, "C:/Users/BOUKA/OneDrive/Bureau/CODE/tsml_eval/", dataset_name=dataset_name, resample_id=0)


