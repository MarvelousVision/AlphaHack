import numpy as np
import polars as pl
from pathlib import Path
from itertools import combinations_with_replacement
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.stattools import jarque_bera

def prepare_data(
    directory: str,
    train_file: str = "train.csv",
    test_file: str = "test.csv",
    types_file: str = "data_dictionary.csv",
    alpha: float | int = 0.05
) -> pl.DataFrame | pl.LazyFrame:
    '''
    Принимает на вход путь к директории с данными, (на всякий случай) названия файлов с ними и
    ещё пороговое p-значение для теста Харке-Бера

    Делит признаки на числовые, бинарные и категориальные (других вроде нет)

    Категориальные признаки заполняет модой и One-Hot энкодит

    Бинарные признаки заполняет модой в зависимости от категориальных и тоже One-Hot энкодит

    Числовые признаки заполняет средним в зависимости от категориальных и бинарных признаков;
    проверяет тестом Харке-Бера и логарифмирует, если они рапределены ненормально;
    добавляет признаки с их взаимодействиями;
    по логике заполняет пропуски в новых признаках нулями или очень большими числами;
    и в конце стандартизирует
    '''
    directory = Path(directory)

    types = pl.read_csv(directory / types_file)
    columns = types["feature"].to_list()
    types = types["dtype"].to_list()

    numeric = [c for c, t in zip(columns, types) if t == "numeric"]
    binary = [c for c, t in zip(columns, types) if t == "binary"]
    category = [c for c, t in zip(columns, types) if t == "category"]

    train = pl.read_csv(
        directory / train_file,
        infer_schema=False,
        null_values=["", "NA", "NaN", "nan", "NULL", "null"],
    ).select(columns)

    test = pl.read_csv(
        directory / test_file,
        infer_schema=False,
        null_values=["", "NA", "NaN", "nan", "NULL", "null"],
    ).select(columns)

    data = pl.concat([train, test])

    for c in numeric:
        value = pl.col(c).cast(pl.Float64, strict=False)
        data = data.with_columns(
            pl.when(value.is_finite()).then(value).otherwise(None).alias(c)
        )

    missing = pl.DataFrame({f"missing__{c}": data[c].is_null().cast(pl.String) for c in columns})

    log_columns = []
    for c in numeric:
        x = data[c].drop_nulls().to_numpy()
        if len(x) >= 8 and np.ptp(x) > 0:
            if jarque_bera(x)[1] < alpha:
                log_columns.append(c)

    for c in category:
        observed = data[c].drop_nulls()
        fill = observed.mode().sort()[0]
        data = data.with_columns(pl.col(c).fill_null(fill))

    for targets, groups, use_mean in [
        (binary, category, False),
        (numeric, category + binary, True),
    ]:
        for c in targets:
            observed = data[c].drop_nulls()

            if use_mean:
                fallback = observed.mean()
                aggregate = pl.col(c).mean()
            else:
                fallback = observed.mode().sort()[0]
                aggregate = pl.col(c).mode().sort().first()

            if groups:
                statistics = (
                    data
                    .filter(pl.col(c).is_not_null())
                    .group_by(groups)
                    .agg(aggregate.alias(c))
                )

                lookup = {
                    tuple(row[:-1]): row[-1]
                    for row in statistics.iter_rows()
                }

                values = [
                    row[-1] if row[-1] is not None
                    else lookup.get(tuple(row[:-1]), fallback)
                    for row in data.select(groups + [c]).iter_rows()
                ]

                data = data.with_columns(
                    pl.Series(c, values, dtype=data.schema[c])
                )
            else:
                data = data.with_columns(pl.col(c).fill_null(fallback))

    blocks = []

    if numeric:
        x = data.select(numeric).to_numpy().copy()

        for j, c in enumerate(numeric):
            if c in log_columns:
                x[:, j] = np.sign(x[:, j]) * np.log1p(np.abs(x[:, j]))

        limit = 1e6
        x = np.clip(
            np.nan_to_num(x, nan=0.0, posinf=limit, neginf=-limit),
            -limit, limit,
        )

        features = [x]
        names = [f"num__{c}" for c in numeric]

        pairs = list(combinations_with_replacement(range(len(numeric)), 2))
        interactions = np.column_stack([
            x[:, i] * x[:, j] for i, j in pairs
        ])

        interactions = np.clip(
            np.nan_to_num(
                interactions, nan=0.0, posinf=limit, neginf=-limit
            ),
            -limit, limit,
        )

        features.append(interactions)
        names.extend(
            f"interaction__{i}__{j}"
            for i, j in pairs
        )

        x = np.column_stack(features)

        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)

        blocks.append(pl.DataFrame(x, schema=names, orient="row"))

    categorical = pl.concat(
        [data.select(category + binary), missing],
        how="horizontal",
    ) if category or binary else missing

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float64,
    )

    encoded_input = categorical.to_numpy()
    encoder.fit(encoded_input)
    encoded = encoder.transform(encoded_input)

    encoded_names = [
        f"ohe__{i}__{name}"
        for i, name in enumerate(
            encoder.get_feature_names_out(categorical.columns)
        )
    ]

    blocks.append(
        pl.DataFrame(encoded, schema=encoded_names, orient="row")
    )

    result = pl.concat(blocks, how="horizontal")

    # Служебный столбец: исходный порядок строк сохраняется.
    return result.with_columns(
        pl.Series("__is_train", [1] * train.height + [0] * test.height)
    )

prepare_data('../data/task1')