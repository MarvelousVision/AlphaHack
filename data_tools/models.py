import json
import numpy as np
import pandas as pd
from typing import Optional, Any
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from solution import X_train, y, X_test, cat, numeric, task, ids, tgt, id_name, METRIC, WAY_TO_JSON

MODELS = {'lnr': LinearRegression}

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


def unimodel(model: Optional[str] = None, to_drop: Optional[list[str] | str] = None, **kwargs) -> dict:

    if to_drop is None:
        x = X_train
        cats = cat
        numerics = numeric
    else:
        x = X_train.drop(columns=to_drop)
        cats = [c for c in cat if c not in to_drop]
        numerics = [n for n in numeric if n not in to_drop]

    with open(WAY_TO_JSON, 'r') as f:
        previos = json.load(f)

    cnt = max(
        (int(key) for key in previos.keys()),
        default=0
    ) + 1

    if task == 'regression':
        target = float_y()
    else:
        encoder = LabelEncoder()
        target = encoder.fit_transform(y)

    if model in ['cbr', 'cbc']:
        optimus = 'passthrough'

        estimator = CatBoostRegressor(cat_features=cats, **kwargs) if model == 'cbr' else CatBoostClassifier(cat_features=cats, **kwargs)

    else:
        optimus = ColumnTransformer(
            transformers=[
                ('encoder', OneHotEncoder(handle_unknown='ignore'), cats),
                ('scaler', StandardScaler(), numerics)
            ],
            remainder='passthrough'
        )

        estimator = MODELS.get(model, None)

        if estimator is None:
            previos[cnt] = {
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
            return previos

        estimator = estimator(**kwargs)

    pipe = Pipeline(
        steps = [
            ('preprocessor', optimus),
            ('estimator', estimator)
        ]
    )
    scoring = 'neg_root_mean_squared_error' if METRIC == 'RMSE' else make_scorer(f1_score, average='macro')
    score = cross_val_score(pipe, x, target, scoring=scoring)

    if METRIC == 'RMSE': score *= -1
    previos[cnt] = {
        'model': model,
        'args': kwargs,
        'to_drop': to_drop,
        'metric_name': METRIC,
        'mean_score': float(np.mean(score)),
        'max_score': float(np.max(score)),
        'min_score': float(np.min(score)),
        'score_std': float(np.std(score))
    }
    with open(WAY_TO_JSON, 'w') as f:
        json.dump(previos, f)
    return previos


def the_end(num: int = 1) -> pd.DataFrame:
    res = pd.DataFrame({id_name: ids[id_name]})

    with open(WAY_TO_JSON, 'r') as f:
        info = json.load(f)[str(num)]

    model = info['model']
    if model is None:
        return pd.DataFrame()

    to_drop = info['to_drop']
    if to_drop is None:
        train, test = X_train, X_test
        cats = cat
        numerics = numeric
    else:
        train, test = X_train.drop(columns=to_drop), X_test.drop(columns=to_drop)
        cats = [c for c in cat if c not in to_drop]
        numerics = [n for n in numeric if n not in to_drop]

    if task == 'regression':
        target = float_y()
    else:
        encoder = LabelEncoder()
        target = encoder.fit_transform(y)

    if model in ['cbr', 'cbc']:
        optimus = 'passthrough'

        estimator = CatBoostRegressor(cat_features=cats, **info['args']) if model == 'cbr' else CatBoostClassifier(cat_features=cats, **info['args'])

    else:
        optimus = ColumnTransformer(
            transformers=[
                ('encoder', OneHotEncoder(handle_unknown='ignore'), cats),
                ('scaler', StandardScaler(), numerics)
            ],
            remainder='passthrough'
        )

        estimator = MODELS.get(model, None)

        if estimator is None:
            return pd.DataFrame()

        estimator = estimator(**info['args'])

    pipe = Pipeline(
        steps=[
            ('preprocessor', optimus),
            ('estimator', estimator)
        ]
    )

    pipe.fit(train, target)
    res[tgt] = pipe.predict(test) if task == 'regression' else encoder.inverse_transform(pipe.predict(test))

    return res

print(unimodel(model='cbc', to_drop=['spkts', 'dpkts'], iterations=1))
print(the_end().head())