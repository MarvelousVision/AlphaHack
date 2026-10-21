import json
from data_tools.preprocessing import prepare_data

X_train, y, X_test, numeric, cat, task, ids, tgt, id_name = prepare_data('../data/task2')
X_train, X_test = X_train.to_pandas(), X_test.to_pandas()
METRIC = 'RMSE' if task == 'regression' else 'f1_macro'
WAY_TO_JSON = '../experiments.json'
with open(WAY_TO_JSON, 'w') as f:
    json.dump(dict(), f)
COUNTER = 0