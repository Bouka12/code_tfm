# run_experiment.py

from data_loader import load_datasets
from tsc_datasets import univariate_equal_length
from config import experiments, experiment_components
from train_and_predict import train_and_predict
from evaluation import evaluate_datasets, metrics
random_state =2025
data_dict = load_datasets(univariate_equal_length)

def run_exp(data_dict, experiment):
    """
    -   data_dict: "dict" dictionary of datasets
    -   experiment: "str" code of the dataset
    """
    exp = experiment_components(experiment)

    interval_selection_func = exp["interval_selection_func"]
    tsrep_funcs = exp["tsrep_funcs"]
    interval_features_func= exp["interval_features_func"]

    preds_dict, preds_proba_dict, tr_time_dict = train_and_predict(data_dict , extract_interval_features=interval_features_func, select_intervals= interval_selection_func,ts_representations=tsrep_funcs, random_state=random_state)
    #def train_and_predict(data_dict , extract_interval_features , select_intervals ,ts_representations, random_state):
    evaluate_datasets(data_dict=data_dict, preds_proba_dict=preds_proba_dict, preds_dict=preds_dict, tr_time_dict=tr_time_dict, metrics=metrics,experiment=experiment)

# test with experiment = "IS0_TSR1_IF5" tested status: WORKING
# test with experiment = "IS0_TSR1_IF1" tested status: WORKING
# test with experiment = "IS0_TSR1_IF2" tested status: WORKING
# test with experiment = "IS0_TSR2_IF5" testes status: WORKING
# test with experiment = "IS0_TSR2_IF3" tested status: WOKRING
# test with experiment = "IS0_TSR2_IF4" tested status: WOKRING

# test with experiment = "IS2_TSR2_IF4" tested status: WORKING
# test with experiment = "IS2_TSR1_IF3" 
'''
test_exp = "IS0_TSR2_IF4"
run_exp(data_dict=data_dict, experiment=test_exp)
'''
for exp in experiments:
    print(f"... {exp} running ...")
    run_exp(data_dict=data_dict, experiment=exp)

