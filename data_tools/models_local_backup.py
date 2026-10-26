import json
import time
import optuna
import numpy as np
import pandas as pd
from typing import Optional, Any
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn import clone
from sklearn.svm import SVC, SVR, LinearSVR, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBRegressor, XGBClassifier

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


def unimodel(model: str,
             search_space: dict,
             time_budget_sec: float,
             task: str,
             cat: list,
             numeric: list,
             train: pd.DataFrame,
             target: np.ndarray,
             to_drop: Optional[list[str] | str] = None) -> dict:

    start = time.time()

    if to_drop is None:
        x = train
        cats = cat
        numerics = numeric
    elif isinstance(to_drop, str):
        x = train.drop(to_drop, axis=1)
        cats = [c for c in cat if c != to_drop]
        numerics = [n for n in numeric if n != to_drop]
    else:
        x = train.drop(columns=to_drop)
        cats = [c for c in cat if c not in to_drop]
        numerics = [n for n in numeric if n not in to_drop]

    y = target.ravel()

    if model not in MODELS: raise ValueError(f'Model {model} is not supported.')
    if task not in MODELS[model]['task']: raise ValueError(f'Model {model} is not for {task}.')

    if task == 'regression':
        y = y.astype(float)
        scoring = 'neg_root_mean_squared_error'
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
    else:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        scoring = 'f1_macro'
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    tree_models = (
        DecisionTreeRegressor, DecisionTreeClassifier,
        RandomForestRegressor, RandomForestClassifier,
        LGBMRegressor, LGBMClassifier,
        XGBRegressor, XGBClassifier,
        CatBoostRegressor, CatBoostClassifier,
    )

    model_class = MODELS[model]['model']
    model_object = model_class()

    if issubclass(model_class, (CatBoostRegressor, CatBoostClassifier)):
        model_object.set_params(
            cat_features=cats,
            verbose=False,
            allow_writing_files=False,
        )

        pipeline = Pipeline([('model', model_object)])

    else:
        preprocessing = ColumnTransformer(
            transformers=[
                (
                    'cats',
                    OneHotEncoder(handle_unknown='ignore'),
                    cats,
                ),
                (
                    'numerics',
                    'passthrough'
                    if issubclass(model_class, tree_models)
                    else StandardScaler(),
                    numerics,
                ),
            ],
            sparse_threshold=1.0,
        )

        pipeline = Pipeline([
            ('preprocessing', preprocessing),
            ('model', model_object),
        ])

    def objective(trial):
        params = {}

        for parameter, variants in search_space.items():
            if isinstance(variants[0], float):
                value = trial.suggest_float(parameter, *variants)
            elif isinstance(variants[0], int):
                value = trial.suggest_int(parameter, *variants)
            else:
                value = trial.suggest_categorical(parameter, [*variants])

            params[f'model__{parameter}'] = value

        candidate = clone(pipeline).set_params(**params)

        return cross_val_score(
            candidate,
            x,
            y,
            scoring=scoring,
            cv=cv,
            n_jobs=1
        ).mean()

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        objective,
        n_trials=30,
        timeout=time_budget_sec
    )

    return {
        'model': model,
        'best_score': study.best_value if task != 'regression' else -study.best_value,
        'best_params': study.best_params,
        'duration_sec': time.time() - start,
        'completed_trials': len(study.trials)
            }


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