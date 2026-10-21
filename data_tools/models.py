import json
import numpy as np
import pandas as pd
from preprocessing import prepare_data
from typing import Optional, Any
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

X_train, y, X_test, cat, task, ids, tgt = prepare_data('../data/task2')
X_train, X_test = X_train.to_pandas(), X_test.to_pandas()
METRIC = 'RMSE' if task == 'regression' else 'f1_macro'
WAY_TO_JSON = '../experiments.json'
with open(WAY_TO_JSON, 'w') as f:
    json.dump(dict(), f)
COUNTER = 0


def encode(what = None) -> Any:
    if what is None: return LabelEncoder().fit_transform(y)
    return ColumnTransformer(
        transformers=[
            ('encoder', OneHotEncoder(), cat)
        ],
        remainder='passthrough'
    ).fit_transform(what)


def float_y() -> np.ndarray:
    return np.ravel(y.to_numpy().astype(float))


def unimodel(model: Optional[str] = None, to_drop: Optional[list[str] | str] = None, **kwargs) -> None:

    x = X_train if to_drop is None else X_train.drop(columns=to_drop)

    global COUNTER
    COUNTER += 1
    with open(WAY_TO_JSON, 'r') as f:
        previos = json.load(f)

    if model == 'lnr':
        score = cross_val_score(LinearRegression(**kwargs),
                                encode(x),
                                float_y(),
                                scoring='neg_root_mean_squared_error',
                                verbose=True)
    elif model == 'cbr':
        score = cross_val_score(CatBoostRegressor(cat_features=cat, **kwargs),
                                x,
                                float_y(),
                                scoring='neg_root_mean_squared_error',
                                verbose=True)
    elif model == 'cbc':
        score = cross_val_score(CatBoostClassifier(cat_features=cat, **kwargs),
                                x,
                                encode(),
                                scoring=make_scorer(f1_score, average='macro'),
                                verbose=True)
    else:
        previos[COUNTER] = {
            'model': None,
            'args': None,
            'to_drop': None,
            'metric_name': None,
            'mean_score': None,
            'max_score': None,
            'min_score': None,
            'score_std': None
        }
        with open(WAY_TO_JSON, 'w') as f:
            json.dump(previos, f)
        return

    if METRIC == 'RMSE': score *= -1
    previos[COUNTER] = {
        'model': model,
        'args': kwargs,
        'to_drop': to_drop,
        'metric_name': METRIC,
        'mean_score': np.mean(score),
        'max_score': np.max(score),
        'min_score': np.min(score),
        'score_std': np.std(score)
    }
    with open(WAY_TO_JSON, 'w') as f:
        json.dump(previos, f)


def the_end(num: int = 1) -> pd.DataFrame:
    res = pd.DataFrame({'id': ids['id']})

    with open(WAY_TO_JSON, 'r') as f:
        info = json.load(f)[str(num)]

    if info['to_drop'] is None:
        train, test = X_train, X_test
    else:
        train, test = X_train.drop(columns=info['to_drop']), X_test.drop(columns=info['to_drop'])

    model = info['model']
    if model == 'cbr':
        estimator = CatBoostRegressor(cat_features=cat, verbose=True, **info['args'])
        estimator.fit(train, float_y())
        preds = estimator.predict(test)
    elif model == 'cbc':
        estimator = CatBoostClassifier(cat_features=cat, verbose=True, **info['args'])
        estimator.fit(train, encode())
        preds = estimator.predict(test)
    elif model is None:
        return pd.DataFrame()
    else:
        optimus = ColumnTransformer(
            transformers=[
                ('encoder', OneHotEncoder(), cat)
            ],
            remainder='passthrough'
        )
        optimus.fit(train)
        train = optimus.transform(train)
        test = optimus.transform(test)

        if model == 'lnr':
            estimator = LinearRegression(**info['args'])
            estimator.fit(train, float_y())
            preds = estimator.predict(test)

    res[tgt] = preds

    return res