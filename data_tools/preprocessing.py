import numpy as np
import polars as pl
from pathlib import Path
from typing import Any
from itertools import combinations_with_replacement
from statsmodels.stats.stattools import jarque_bera
from catboost import CatBoostClassifier, CatBoostRegressor

def prepare_data(
        directory: str,
        train_file: str = "train.csv",
        test_file: str = "test.csv",
        types_file: str = "data_dictionary.csv",
        alpha: float | int = 0.05,
        quantity_threshold: int = 100,
        importance_threshold: float = 0.1
) -> Any:
    '''
    Принимает на вход путь к директории с данными, (на всякий случай) названия файлов с ними и
    ещё пороговое p-значение для теста Харке-Бера

    Делит признаки на числовые, бинарные и категориальные (других вроде нет)

    Категориальные признаки заполняет модой

    Бинарные признаки заполняет модой в зависимости от категориальных

    Числовые признаки заполняет средним в зависимости от категориальных и бинарных признаков;
    проверяет тестом Харке-Бера и логарифмирует, если они рапределены ненормально;
    добавляет признаки с их взаимодействиями;
    по логике заполняет пропуски в новых признаках нулями или очень большими числами
    '''
    directory = Path(directory)

    types = pl.read_csv(directory / types_file)
    y = types.filter(pl.col("role").str.contains("target"))
    tgt = y['feature'][0]
    id_row = types.filter(pl.col("role").str.contains("id"))
    id_name = id_row['feature'][0]
    if y['dtype'][0] == 'numeric':
        task = 'regression'
    elif y['dtype'][0] == 'binary':
        task = 'binary'
    else:
        task = 'multiclass'
    columns = types["feature"].to_list()
    types = [t for c, t in zip(columns, types["dtype"].to_list()) if c not in (id_name, tgt)]
    columns.remove(id_name)
    columns.remove(tgt)

    numeric = [c for c, t in zip(columns, types) if t == "numeric"]
    binary = [c for c, t in zip(columns, types) if t == "binary"]
    category = [c for c, t in zip(columns, types) if t == "category"]

    train_data = pl.read_csv(
        directory / train_file,
        infer_schema=False,
        null_values=["", "NA", "NaN", "nan", "NULL", "null"]
    )
    test_data = pl.read_csv(
        directory / test_file,
        infer_schema=False,
        null_values=["", "NA", "NaN", "nan", "NULL", "null"]
    )
    ids = test_data.select(id_name)
    train_data = train_data.drop_nulls(subset=[tgt])
    y_col = train_data.select(tgt)
    train_data = train_data.select(columns)
    test_data = test_data.select(columns)

    for c in numeric:
        value = pl.col(c).cast(pl.Float64, strict=False)
        train_data = train_data.with_columns(
            pl.when(value.is_finite()).then(value).otherwise(None).alias(c)
        )
        test_data = test_data.with_columns(
            pl.when(value.is_finite()).then(value).otherwise(None).alias(c)
        )

    missing_train = pl.DataFrame({f"missing__{c}": train_data[c].is_null().cast(pl.String) for c in columns})
    missing_test = pl.DataFrame({f"missing__{c}": test_data[c].is_null().cast(pl.String) for c in columns})

    log_columns = []
    for c in numeric:
        x = train_data[c].drop_nulls().to_numpy()
        if len(x) >= 8 and np.ptp(x) > 0:
            if jarque_bera(x)[1] < alpha:
                log_columns.append(c)

    for c in category:
        observed = train_data[c].drop_nulls()
        modes = observed.mode().sort()
        fill = modes[0] if len(modes) else '__MISSING__'
        train_data = train_data.with_columns(pl.col(c).fill_null(fill))
        test_data = test_data.with_columns(pl.col(c).fill_null(fill))

    for targets, groups, use_mean in [
        (binary, category, False),
        (numeric, category + binary, True),
    ]:
        for c in targets:
            observed = train_data[c].drop_nulls()

            if use_mean:
                fallback = observed.mean()
                if fallback is None or not np.isfinite(fallback):
                    fallback = 0.0
                aggregate = pl.col(c).mean()
            else:
                modes = observed.mode().sort()
                fallback = modes[0] if len(modes) else '__MISSING__'
                aggregate = pl.col(c).mode().sort().first()

            if groups:
                statistics = (
                    train_data
                    .filter(pl.col(c).is_not_null())
                    .group_by(groups)
                    .agg(aggregate.alias(c))
                )

                lookup = {
                    tuple(row[:-1]): row[-1]
                    for row in statistics.iter_rows()
                }

                train_values = [
                    row[-1] if row[-1] is not None
                    else lookup.get(tuple(row[:-1]), fallback)
                    for row in train_data.select(groups + [c]).iter_rows()
                ]
                test_values = [
                    row[-1] if row[-1] is not None
                    else lookup.get(tuple(row[:-1]), fallback)
                    for row in test_data.select(groups + [c]).iter_rows()
                ]

                train_data = train_data.with_columns(
                    pl.Series(c, train_values, dtype=train_data.schema[c])
                )
                test_data = test_data.with_columns(
                    pl.Series(c, test_values, dtype=test_data.schema[c])
                )
            else:
                train_data = train_data.with_columns(pl.col(c).fill_null(fallback))
                test_data = test_data.with_columns(pl.col(c).fill_null(fallback))

    numeric_count = len(numeric)
    limit = 1e12

    if numeric:
        x_train = train_data.select(numeric).to_numpy().copy()
        x_test = test_data.select(numeric).to_numpy().copy()

        for j, c in enumerate(numeric):
            if c in log_columns:
                x_train[:, j] = (
                        np.sign(x_train[:, j])
                        * np.log1p(np.abs(x_train[:, j]))
                )
                x_test[:, j] = (
                        np.sign(x_test[:, j])
                        * np.log1p(np.abs(x_test[:, j]))
                )

        x_train = np.clip(
            np.nan_to_num(
                x_train, nan=0.0, posinf=limit, neginf=-limit
            ),
            -limit, limit
        )
        x_test = np.clip(
            np.nan_to_num(
                x_test, nan=0.0, posinf=limit, neginf=-limit
            ),
            -limit, limit
        )

        train_data = train_data.with_columns(
            pl.DataFrame(x_train, schema=numeric, orient="row")
        )
        test_data = test_data.with_columns(
            pl.DataFrame(x_test, schema=numeric, orient="row")
        )

    # Индикаторы пропусков тоже участвуют в отборе.
    train_data = pl.concat(
        [train_data, missing_train],
        how="horizontal"
    )
    test_data = pl.concat(
        [test_data, missing_test],
        how="horizontal"
    )

    cat = category + binary + missing_train.columns

    def select_features(train, test, numeric, cat):
        target = y_col.to_series().to_numpy().ravel()

        if task == "regression":
            target = target.astype(float)

            model = CatBoostRegressor(
                iterations=300,
                depth=6,
                random_seed=42,
                verbose=False,
                allow_writing_files=False
            )
        else:
            model = CatBoostClassifier(
                iterations=300,
                depth=6,
                random_seed=42,
                verbose=False,
                allow_writing_files=False
            )

        model.fit(
            train.to_pandas(),
            target,
            cat_features=cat
        )

        importance = model.get_feature_importance(
            type="PredictionValuesChange"
        )

        selected = [
            column
            for column, value in zip(train.columns, importance)
            if value > importance_threshold
        ]

        if not selected:
            raise ValueError(
                "После отбора не осталось признаков. "
                "Уменьшите importance_threshold."
            )

        numeric = [c for c in numeric if c in selected]
        cat = [c for c in cat if c in selected]

        return (
            train.select(selected),
            test.select(selected),
            numeric,
            cat
        )

    def add_interactions(train, test, numeric):
        expressions = []
        names = []

        for i, j in combinations_with_replacement(
                range(len(numeric)), 2
        ):
            name = f"interaction__{i}__{j}"

            if name in train.columns:
                raise ValueError(
                    f"Имя нового признака уже используется: {name}"
                )

            expressions.append(
                (
                        pl.col(numeric[i]) * pl.col(numeric[j])
                )
                .clip(-limit, limit)
                .alias(name)
            )
            names.append(name)

        if expressions:
            train = train.with_columns(expressions)
            test = test.with_columns(expressions)

        return train, test, numeric + names

    if numeric_count < quantity_threshold:
        train_data, test_data, numeric = add_interactions(
            train_data, test_data, numeric
        )
        train_data, test_data, numeric, cat = select_features(
            train_data, test_data, numeric, cat
        )
    else:
        train_data, test_data, numeric, cat = select_features(
            train_data, test_data, numeric, cat
        )
        train_data, test_data, numeric = add_interactions(
            train_data, test_data, numeric
        )

    return (
        train_data,
        y_col,
        test_data,
        numeric,
        cat,
        task,
        ids,
        tgt,
        id_name
    )