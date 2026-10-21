import numpy as np
import polars as pl
from pathlib import Path
from typing import Any
from itertools import combinations_with_replacement
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.stattools import jarque_bera

def prepare_data(
        directory: str,
        train_file: str = "train.csv",
        test_file: str = "test.csv",
        types_file: str = "data_dictionary.csv",
        alpha: float | int = 0.05
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
    по логике заполняет пропуски в новых признаках нулями или очень большими числами;
    и в конце стандартизирует
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

    train_blocks = []
    test_blocks = []

    if numeric:
        x_train = train_data.select(numeric).to_numpy().copy()
        x_test = test_data.select(numeric).to_numpy().copy()

        for j, c in enumerate(numeric):
            if c in log_columns:
                x_train[:, j] = np.sign(x_train[:, j]) * np.log1p(np.abs(x_train[:, j]))
                x_test[:, j] = np.sign(x_test[:, j]) * np.log1p(np.abs(x_test[:, j]))

        limit = 1e12
        x_train = np.clip(
            np.nan_to_num(x_train, nan=0.0, posinf=limit, neginf=-limit),
            -limit, limit,
        )
        x_test = np.clip(
            np.nan_to_num(x_test, nan=0.0, posinf=limit, neginf=-limit),
            -limit, limit,
        )

        train_features = [x_train]
        test_features = [x_test]

        pairs = list(combinations_with_replacement(range(len(numeric)), 2))
        train_interactions = np.column_stack([
            x_train[:, i] * x_train[:, j] for i, j in pairs
        ])
        test_interactions = np.column_stack([
            x_test[:, i] * x_test[:, j] for i, j in pairs
        ])

        train_interactions = np.clip(
            np.nan_to_num(
                train_interactions, nan=0.0, posinf=limit, neginf=-limit
            ),
            -limit, limit,
        )
        test_interactions = np.clip(
            np.nan_to_num(
                test_interactions, nan=0.0, posinf=limit, neginf=-limit
            ),
            -limit, limit
        )

        train_features.append(train_interactions)
        test_features.append(test_interactions)
        numeric.extend(
            f"interaction__{i}__{j}"
            for i, j in pairs
        )

        x_train = np.column_stack(train_features)
        x_test = np.column_stack(test_features)

        train_blocks.append(pl.DataFrame(x_train, schema=numeric, orient="row"))
        test_blocks.append(pl.DataFrame(x_test, schema=numeric, orient="row"))

    categorical_train = pl.concat(
        [train_data.select(category + binary), missing_train],
        how="horizontal",
    ) if category or binary else missing_train

    categorical_test = pl.concat(
        [test_data.select(category + binary), missing_test],
        how="horizontal",
    ) if category or binary else missing_test

    cat = categorical_train.columns
    train_blocks.append(
        pl.DataFrame(categorical_train, schema=cat, orient="row")
    )
    test_blocks.append(
        pl.DataFrame(categorical_test, schema=cat, orient="row")
    )

    return (
        pl.concat(train_blocks, how="horizontal"),
        y_col,
        pl.concat(test_blocks, how="horizontal"),
        numeric,
        cat,
        task,
        ids,
        tgt,
        id_name
    )