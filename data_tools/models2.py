import optuna
import numpy as np
import pandas as pd
from typing import Any
from time import monotonic
from sklearn.base import clone
from sklearn.svm import SVC, SVR, LinearSVR, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from solution import X_train, y, X_test, cat, numeric, task, ids, tgt, id_name

MODELS = {
    'lnr': {'model': LinearRegression,
            'task': 'regression',
            'params': (('fit_intercept', True, False),)},
    'dtr': {'model': DecisionTreeRegressor,
            'task': 'regression',
            'params': (('max_depth', 2, 30),
                       ('min_samples_split', 0.01, 0.2))},
    'dtc': {'model': DecisionTreeClassifier,
            'task': ('binary', 'multiclass'),
            'params': (('max_depth', 2, 30),
                       ('min_samples_split', 0.01, 0.2))},
    'rfr': {'model': RandomForestRegressor,
            'task': 'regression',
            'params': (('max_depth', 2, 30),
                       ('min_samples_split', 0.01, 0.2))},
    'rfc': {'model': RandomForestClassifier,
            'task': ('binary', 'multiclass'),
            'params': (('max_depth', 2, 30),
                       ('min_samples_split', 0.01, 0.2))},
    'svr': {'model': SVR,
            'task': 'regression',
            'params': (('C', 0.01, 10),
                       ('max_iter', 100, 1000))},
    'svc': {'model': SVC,
            'task': ('binary', 'multiclass'),
            'params': (('C', 0.01, 10),
                       ('max_iter', 100, 1000))},
    'lvr': {'model': LinearSVR,
             'task': 'regression',
            'params': (('C', 0.01, 10),
                       ('max_iter', 100, 1000))},
    'lvc': {'model': LinearSVC,
            'task': ('binary', 'multiclass'),
            'params': (('C', 0.01, 10),
                       ('max_iter', 100, 1000))},
    'lbr': {'model': LGBMRegressor,
            'task': 'regression',
            'params': (('n_estimators', 100, 1000),)},
    'lbc': {'model': LGBMClassifier,
            'task': ('binary', 'multiclass'),
            'params': (('n_estimators', 100, 1000),)},
    'xbr': {'model': XGBRegressor,
            'task': 'regression',
            'params': (('n_estimators', 100, 1000),)},
    'xbc': {'model': XGBClassifier,
            'task': ('binary', 'multiclass'),
            'params': (('n_estimators', 100, 1000),)}
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


def choir(m1: str, m2: str, m3: str, time_left: int) -> pd.DataFrame:
    if time_left <= 0:
        raise ValueError('time_left must be positive.')

    started = monotonic()
    model_names = (m1, m2, m3)

    for model in model_names:
        if model not in MODELS:
            raise ValueError(f'Model {model} is not supported.')
        if task not in MODELS[model]['task']:
            raise ValueError(f'Model {model} is not for {task}.')

    test_ids = ids.to_numpy().ravel()
    features = cat + numeric

    train = X_train.select(features).to_pandas()
    test = X_test.select(features).to_pandas()
    target = y[tgt].to_numpy()

    if task == 'regression':
        target = target.astype(float)
        scoring = 'neg_root_mean_squared_error'
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
    else:
        label_encoder = LabelEncoder()
        target = label_encoder.fit_transform(target)
        scoring = 'f1_macro'
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    tree_models = (
        DecisionTreeRegressor, DecisionTreeClassifier,
        RandomForestRegressor, RandomForestClassifier,
        LGBMRegressor, LGBMClassifier,
        XGBRegressor, XGBClassifier,
        CatBoostRegressor, CatBoostClassifier,
    )

    predictions = []
    scores = []
    durations = []

    for index, name in enumerate(model_names):
        remaining = time_left - (monotonic() - started)

        if durations and remaining <= max(durations) + 300:
            break

        model_started = monotonic()
        config = MODELS[name]
        model_class = config['model']
        model = model_class()

        if issubclass(model_class, (CatBoostRegressor, CatBoostClassifier)):
            model.set_params(
                cat_features=cat,
                verbose=False,
                allow_writing_files=False,
            )

            model_train = train.copy()
            model_test = test.copy()

            for column in cat:
                model_train[column] = (
                    model_train[column].fillna('__MISSING__').astype(str)
                )
                model_test[column] = (
                    model_test[column].fillna('__MISSING__').astype(str)
                )

            pipeline = Pipeline([('model', model)])

        else:
            model_train = train
            model_test = test

            preprocessing = ColumnTransformer(
                transformers=[
                    (
                        'cat',
                        OneHotEncoder(handle_unknown='ignore'),
                        cat,
                    ),
                    (
                        'numeric',
                        'passthrough'
                        if issubclass(model_class, tree_models)
                        else StandardScaler(),
                        numeric,
                    ),
                ],
                sparse_threshold=1.0,
            )

            pipeline = Pipeline([
                ('preprocessing', preprocessing),
                ('model', model),
            ])

        def objective(trial):
            params = {}

            for parameter, low, high in config['params']:
                if isinstance(low, bool):
                    value = trial.suggest_categorical(parameter, [low, high])
                elif isinstance(low, int) and isinstance(high, int):
                    value = trial.suggest_int(parameter, low, high)
                else:
                    value = trial.suggest_float(parameter, low, high)

                params[f'model__{parameter}'] = value

            candidate = clone(pipeline).set_params(**params)

            return cross_val_score(
                candidate,
                model_train,
                target,
                scoring=scoring,
                cv=cv,
                n_jobs=1,
                error_score='raise',
            ).mean()

        remaining = time_left - (monotonic() - started)

        tuning_budget = max(1.0, (remaining - 300) / (4 - index))

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(
            objective,
            n_trials=30,
            timeout=tuning_budget,
        )

        best_model = clone(pipeline).set_params(**{
            f'model__{parameter}': value
            for parameter, value in study.best_params.items()
        })
        best_model.fit(model_train, target)

        predictions.append(
            np.asarray(best_model.predict(model_test)).ravel()
        )
        scores.append(study.best_value)
        durations.append(monotonic() - model_started)

    if task == 'regression':
            result = np.mean(predictions, axis=0)
    else:
        order = np.argsort(-np.asarray(scores), kind='stable')
        votes = np.asarray(predictions, dtype=int)[order]
        result = votes[0].copy()

        if len(votes) == 3:
            agreement = votes[1] == votes[2]
            result[agreement] = votes[1, agreement]

        result = label_encoder.inverse_transform(result)

    return pd.DataFrame({
        id_name: test_ids,
        tgt: result,
    })