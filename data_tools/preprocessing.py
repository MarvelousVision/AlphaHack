import numpy as np
import polars as pl
from pathlib import Path
from itertools import combinations_with_replacement
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.stattools import jarque_bera

def prepare_data(
    directory: str,
    file: str = "train.csv",
    types_file: str = "data_dictionary.csv",
    alpha: float | int = 0.05
) -> tuple[pl.DataFrame, pl.DataFrame | None, list[str]]:
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
    y = types.filter(pl.col("description_ru").str.contains("целевая переменная"))
    columns = types["feature"].to_list()
    if 'id' in columns:
        has_id = True
        columns.remove('id')
    else:
        has_id = False
    columns.remove(y['feature'][0])
    types = [t for t in types["dtype"].to_list() if type not in ('id', y['feature'][0])]

    numeric = [c for c, t in zip(columns, types) if t == "numeric"]
    binary = [c for c, t in zip(columns, types) if t == "binary"]
    category = [c for c, t in zip(columns, types) if t == "category"]

    data = pl.read_csv(
        directory / file,
        infer_schema=False,
        null_values=["", "NA", "NaN", "nan", "NULL", "null"],
        row_index_name = 'id'
    )
    if has_id:
        id = data.select('id')
    if file.startswith('train'):
        data = data.drop_nulls(subset=[y['feature'][0]])
        y_col = data.select(y["feature"][0])
    else:
        y_col = None
    data = data.select(columns)

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

        limit = 1e12
        x = np.clip(
            np.nan_to_num(x, nan=0.0, posinf=limit, neginf=-limit),
            -limit, limit,
        )

        features = [x]

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
        numeric.extend(
            f"interaction__{i}__{j}"
            for i, j in pairs
        )

        x = np.column_stack(features)

        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)

        blocks.append(pl.DataFrame(x, schema=numeric, orient="row"))

    categorical = pl.concat(
        [data.select(category + binary), missing],
        how="horizontal",
    ) if category or binary else missing

    cats = categorical.columns
    blocks.append(
        pl.DataFrame(categorical, schema=cats, orient="row")
    )
    if has_id:
        blocks.append(id)
    return pl.concat(blocks, how="horizontal"), y_col, cats