import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from extract_features import extract_interval_features_func


def train_and_predict(data_dict, extract_interval_features, select_intervals, ts_representations, random_state):
    preds_dict = {}
    preds_proba_dict = {}
    
    for dataset_name, data in data_dict.items():
        print(f"Processing dataset: {dataset_name}")
        
        # Load the training and test data
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']

        X_train_features_set = []
        X_test_features_set = []
        
        # Loop over time series representations
        for tsrep in ts_representations:
            print(f" Transforming the train and test of {dataset_name} using {tsrep}")
            
            print(f"X_train_tr before squeexing = {X_train.shape}")
            X_train= X_train.reshape(X_train.shape[0], X_train.shape[-1])
            X_test = X_test.reshape(X_test.shape[0],X_test.shape[-1] )
            print(f"X_train_tr shape after squeezing = {X_train.shape}")
            # Apply the transformation to a copy of the original data (keep independent transformations)
            X_train_tr = tsrep(X_train.copy())
            X_test_tr = tsrep(X_test.copy())


            # Select intervals for the transformed data
            print(f" Interval selection using: {select_intervals}")
            intervals = select_intervals(X_train_tr, len(ts_representations), random_state)
            print(f"The number of intervals = {len(intervals)}")
            print(f"the intervals are  = {intervals}")
            
            # Extract features from intervals for training data
            print(f"extract interval features: {extract_interval_features}")
            X_train_features = extract_interval_features_func(X_train_tr, extract_interval_features, intervals)
            print(f" The shape f the X_train_features = {X_train_features.shape} and type ={type(X_train_features)}")
            # Extract features from intervals for test data
            X_test_features = extract_interval_features_func(X_test_tr, extract_interval_features, intervals)
            print(f" The shape f the X_train_features = {X_test_features.shape} and type ={type(X_test_features)}")

            # Append the features for this representation
            X_train_features_set.append(X_train_features)
            X_test_features_set.append(X_test_features)
        
        # Concatenate all features for each representation along the feature axis (axis=1)
        X_train_final = np.hstack(X_train_features_set)
        X_test_final = np.hstack(X_test_features_set)
        
        # Train ExtraTreesClassifier
        clf = ExtraTreesClassifier(random_state=random_state)
        clf.fit(X_train_final, y_train)
        
        # Predict on test data
        y_pred = clf.predict(X_test_final)
        y_preds_proba = clf.predict_proba(X_test_final)
        preds_proba_dict[dataset_name] = y_preds_proba
        preds_dict[dataset_name] = y_pred

    return preds_dict, preds_proba_dict