from data_loader import load_datasets
from tsc_datasets import univariate_equal_length
from aeon.visualisation import plot_critical_difference
import matplotlib.pyplot as plt
import numpy as np
data_dict = load_datasets(univariate_equal_length)
data_length= []
train_data_instances = []
test_data_instances =[]
for dataset in data_dict.keys():
    print(f"train shape: {data_dict[dataset]['X_train'].shape}")
    train_instances, train_length = data_dict[dataset]['X_train'].shape[0], data_dict[dataset]['X_train'].shape[2]
    test_instances, test_length = data_dict[dataset]['X_test'].shape[0], data_dict[dataset]['X_test'].shape[2]
    data_length.append(train_length)
    train_data_instances.append(train_instances)
    test_data_instances.append(test_instances)

print(f"max train instances: {np.max(train_data_instances)}")
print(f"min train instances: {np.min(train_data_instances)}")
print(f"max test instances: {np.max(test_data_instances)}")
print(f"min test instances: {np.min(test_data_instances)}")
# length
print(f"max length of data: {np.max(data_length)}")
print(f"min data length: {np.min(data_length)}")