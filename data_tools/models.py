from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from typing import Optional

def unimodel(model: Optional[str] = None, drop: Optional[list[str] | str] = None, **kwargs) -> dict:

    if drop is not None: pass

    if model == 'lnr':
        pass
    elif model == 'lgr':
        pass
    elif model == 'cbr':
        pass
    elif model == 'cbc':
        pass
    else:
        return {'model': None, 'metric': -1}