import json
import numpy as np
import pandas as pd
from typing import Optional, Any
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.svm import SVC, SVR, LinearSVR, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBRegressor, XGBClassifier
from solution import X_train, y, X_test, cat, numeric, task, ids, tgt, id_name, METRIC, WAY_TO_JSON

MODELS = {
    'lnr': {'model': LinearRegression,
            'task': 'regression'},
    'dtr': {'model': DecisionTreeRegressor,
            'task': 'regression'},
    'dtc': {'model': DecisionTreeClassifier,
            'task': ['binary', 'multiclass']},
    'rfr': {'model': RandomForestRegressor,
            'task': 'regression'},
    'rfc': {'model': RandomForestClassifier,
            'task': ['binary', 'multiclass']},
    'svr': {'model': SVR,
            'task': 'regression'},
    'svc': {'model': SVC,
            'task': ['binary', 'multiclass']},
    'lvr': {'model': LinearSVR,
             'task': 'regression'},
    'lvc': {'model': LinearSVC,
            'task': ['binary', 'multiclass']},
    'lbr': {'model': LGBMRegressor,
            'task': 'regression'},
    'lbc': {'model': LGBMClassifier,
            'task': ['binary', 'multiclass']},
    'xbr': {'model': XGBRegressor,
            'task': 'regression'},
    'xbc': {'model': XGBClassifier,
            'task': ['binary', 'multiclass']}
}

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
    elif isinstance(to_drop, str):
        x = X_train.drop(to_drop, axis=1)
        cats = [c for c in cat if c != to_drop]
        numerics = [n for n in numeric if n != to_drop]
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
        target = encoder.fit_transform(y[tgt].to_numpy().ravel())

    if model in ['cbr', 'cbc']:
        optimus = 'passthrough'

        if model == 'cbr':
            if task != 'regression':
                raise ValueError('Функция отсутствует или не соответствует задаче!')
            estimator = CatBoostRegressor(cat_features=cats, **kwargs)
        else:
            if task == 'regression':
                raise ValueError('Функция отсутствует или не соответствует задаче!')
            estimator = CatBoostClassifier(cat_features=cats, **kwargs)

    else:
        optimus = ColumnTransformer(
            transformers=[
                ('encoder', OneHotEncoder(handle_unknown='ignore'), cats),
                ('scaler', StandardScaler(), numerics)
            ],
            remainder='passthrough'
        )

        estimator = MODELS.get(model, {})

        if estimator == {} or task not in estimator.get('task', ''):
            raise ValueError('Функция отсутствует или не соответствует задаче!')

        estimator = estimator['model'](**kwargs)

    pipe = Pipeline(
        steps = [
            ('preprocessor', optimus),
            ('estimator', estimator)
        ]
    )
    scoring = 'neg_root_mean_squared_error' if METRIC == 'RMSE' else make_scorer(f1_score, average='macro')
    cv = (
    KFold(n_splits=3, shuffle=True, random_state=42)
    if task == "regression"
    else StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
)
    score = cross_val_score(
        pipe,
        x,
        target,
        scoring=scoring,
        cv=cv,
        n_jobs=1,
    )
    
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
    elif isinstance(to_drop, str):
        train, test = X_train.drop(to_drop, axis=1), X_test.drop(to_drop, axis=1)
        cats = [c for c in cat if c != to_drop]
        numerics = [n for n in numeric if n != to_drop]
    else:
        train, test = X_train.drop(columns=to_drop), X_test.drop(columns=to_drop)
        cats = [c for c in cat if c not in to_drop]
        numerics = [n for n in numeric if n not in to_drop]

    if task == 'regression':
        target = float_y()
    else:
        encoder = LabelEncoder()
        target = encoder.fit_transform(y[tgt].to_numpy().ravel())

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

        estimator = estimator['model'](**info['args'])

    pipe = Pipeline(
        steps=[
            ('preprocessor', optimus),
            ('estimator', estimator)
        ]
    )

    pipe.fit(train, target)
    res[tgt] = pipe.predict(test) if task == 'regression' else encoder.inverse_transform(pipe.predict(test))

    return res