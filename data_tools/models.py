import numpy as np
from typing import Optional, Any
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from preprocessing import prepare_data

X, y, cats = prepare_data('../data/task1')

def encode(what: str) -> Any:
    if what == 'y': return LabelEncoder().fit_transform(y)
    to_return = ColumnTransformer(
        transformers=[
            ('encoder', OneHotEncoder(), cats)
        ],
        remainder='passthrough'
    ).fit_transform(X.to_pandas().drop('id', axis=1))
    return to_return

def float_y() -> np.ndarray:
    return np.ravel(y.to_numpy().astype(float))

def unimodel(model: Optional[str] = None, to_drop: Optional[list[str] | str] = None, **kwargs) -> dict:

    if to_drop is not None:
        global X
        X = X.drop(to_drop)

    if model == 'lnr':
        score = np.mean(cross_val_score(LinearRegression(**kwargs),
                                encode('X'),
                                float_y(),
                                scoring='neg_root_mean_squared_error',
                                n_jobs=-1))
    elif model == 'cbr':
        score = np.mean(cross_val_score(CatBoostRegressor(cat_features=cats, **kwargs),
                                X.to_pandas(),
                                float_y(),
                                scoring='neg_root_mean_squared_error',
                                n_jobs=-1))
    elif model == 'cbc':
        score = np.mean(cross_val_score(CatBoostClassifier(cat_features=cats, **kwargs),
                                X.to_pandas(),
                                encode('y'),
                                scoring=make_scorer(f1_score, average='macro'),
                                n_jobs=-1))
    else:
        return {'model': None, 'score': -1}

    return {'model': model, 'args': kwargs, 'score': score}

print(unimodel('lnr'))