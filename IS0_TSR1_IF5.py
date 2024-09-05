"""
IS0 : DrCIF;
TSR1 : Base time series + 1st Difference + Periodogram
IF5 : quantiles
"""
from data_loader import load_datasets
from evaluation import metrics, evaluate_datasets
from tsc_datasets import univariate_equal_length
from tsfresh_ import extract_interval_features
from sklearn.ensemble import ExtraTreesClassifier
from select_intervals import select_intervals_dr_cif

# load the data
random_state = 2025
data = load_datasets(univariate_equal_length)

# TSR0: `base time series`
n_representation = 1
# IS0: DrCIF `random selection`
# IF6: TSFresh
def train_and_evaluate(data_dict = data, extract_interval_features = extract_interval_features, select_intervals = select_intervals_dr_cif , random_state=random_state):
    preds_dict = {}
    for dataset_name, data in data_dict.items():
        print(f"Processing dataset: {dataset_name}")
        
        # Load the training and test data
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        
        # Select intervals for both training and test data
        intervals = select_intervals(X_train, n_representation, random_state)
        print( f"X_train of {dataset_name} = {X_train.shape}")
        print( f"X_train of {dataset_name} = {X_train.shape}")

        #X_train_ =X_train.reshape(X_train.shape[0],1, X_train.shape[1])
        #X_test_ = X_test.reshape(X_test.shape[0],1, X_test.shape[1])

        # Extract features from intervals for training data
        X_train_features = extract_interval_features(X_train, intervals)

        # Extract features from intervals for test data
        X_test_features = extract_interval_features(X_test, intervals)
        
        # Train ExtraTreesClassifier
        clf = ExtraTreesClassifier(random_state=random_state)
        clf.fit(X_train_features, y_train)
        
        # Predict on test data
        y_pred = clf.predict(X_test_features)
        preds_dict[dataset_name] = y_pred
    return preds_dict

preds_dict = train_and_evaluate(data_dict = data, extract_interval_features = extract_interval_features, select_intervals = select_intervals_dr_cif , random_state=random_state)
evaluate_datasets(data_dict = data, preds_dict=preds_dict, metrics=metrics)