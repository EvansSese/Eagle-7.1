"""Chronological multi-market model selection and progressive predictions."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import uuid
import warnings
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import BaseCrossValidator

from core.database import connect_existing_database
from core.fixture_dates import fixture_dates_utc
from training.deep_learning.train_sequence_models import _build_sequences, _read_fixture_tables
from training.market_targets import build_market_targets

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSIFICATION_TARGETS = ("result", "btts", "over_0_5", "over_1_5", "over_2_5", "over_3_5", "correct_score")
REGRESSION_TARGETS = ("home_goals", "away_goals", "total_goals")
TARGETS = CLASSIFICATION_TARGETS + REGRESSION_TARGETS
RATIO_MARKERS = (
    "per_match", "_pm", "probability", "probabilty", "consistency", "rating",
    "expected_goals", "_xg", "strength", "_as", "_ds", "odds", "rho",
    "league_avg", "form", "streak_outcome", "expected_goal_margin",
)
HISTORY_LENGTH = 10
MODEL_ROOT = PROJECT_ROOT / "storage" / "trained_market_models" / "v4"
PUBLICATION_LOCK = threading.RLock()


class RareClassCalibrationSplit(BaseCrossValidator):
    """Keep singleton classes in every training fold, never in calibration."""

    def __init__(self, folds=3, random_state=42):
        self.folds = folds
        self.random_state = random_state

    def split(self, X, y, groups=None):
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        holdouts = [[] for _ in range(self.folds)]
        for label in np.unique(y):
            indices = np.flatnonzero(y == label)
            if len(indices) == 1:
                continue
            indices = rng.permutation(indices)
            start = min(range(self.folds), key=lambda index: len(holdouts[index]))
            for position, index in enumerate(indices):
                holdouts[(start + position % min(self.folds, len(indices))) % self.folds].append(index)
        nonempty = [np.asarray(fold, dtype=int) for fold in holdouts if fold]
        if len(nonempty) < 2:
            raise ValueError("SVM calibration requires repeated examples to form at least two held-out folds")
        for test in nonempty:
            train = np.ones(len(y), dtype=bool)
            train[test] = False
            yield np.flatnonzero(train), test

    def get_n_splits(self, X=None, y=None, groups=None):
        if y is None:
            return self.folds
        return sum(1 for _ in self.split(X, y, groups))


def _quote(name):
    return '"' + name.replace('"', '""') + '"'


def load_splits(database, league_id):
    league_id = int(league_id)
    with connect_existing_database(database) as con:
        metadata = con.execute(
            f'SELECT dataset_version FROM {_quote(f"{league_id}-dataset-metadata")}'
        ).fetchone()
        if not metadata or metadata[0] != "chronological-splits-v1":
            raise ValueError("Prepare chronological datasets before training")
        frames = [
            con.execute(f'SELECT * FROM {_quote(f"{league_id}-{name}-dataset")}').df()
            for name in ("training", "validation", "test")
        ]
    for index, frame in enumerate(frames):
        if frame.empty or frame["fixture_id"].isna().any() or frame["fixture_id"].duplicated().any():
            raise ValueError("Each split must contain unique fixture IDs and at least one row")
        frame["date"] = fixture_dates_utc(frame["date"])
        if frame["date"].isna().any():
            raise ValueError("Fixture dates cannot be missing")
        frame = frame.sort_values(["date", "fixture_id"], kind="stable").reset_index(drop=True)
        frames[index] = frame
        build_market_targets(frame)
    if not (frames[0]["date"].max() <= frames[1]["date"].min()
            and frames[1]["date"].max() <= frames[2]["date"].min()):
        raise ValueError("Dataset splits are not chronological")
    if pd.concat([frame["fixture_id"] for frame in frames]).duplicated().any():
        raise ValueError("Fixture IDs overlap across splits")
    if any(set(frame.columns) != set(frames[0].columns) for frame in frames[1:]):
        raise ValueError("Dataset split schemas differ")
    return frames


def feature_bundles(frame):
    """Compare rate/form features with the full set of pre-match aggregates."""
    def usable(name):
        if "__" not in name:
            return False
        _, column = name.split("__", 1)
        column = column.lower()
        return not (column in {"fixture_id", "league_id", "season", "round", "date",
                               "home_team", "away_team", "home_goals", "away_goals",
                               "venue", "status", "result"}
                    or column.endswith(("_id", "_name")))

    features = [name for name in frame if usable(name)]
    if not features:
        raise ValueError("No prepared feature columns found")
    ratio = [name for name in features if any(marker in name.lower() for marker in RATIO_MARKERS)]
    if not ratio:
        raise ValueError("No ratio-based features found")
    # Season is a categorical context feature, not a team or fixture identifier.
    return {"ratios": ratio + ["season"], "aggregates": features + ["season"]}


def add_team_differences(frame):
    """Append numeric pre-match x-team minus y-team metrics for matching scopes."""
    names = set(frame.columns)
    differences = {}
    for name in frame.columns:
        if "__" not in name:
            continue
        prefix, metric = name.split("__", 1)
        if not re.search(r"(^|_)x_team_", metric):
            continue
        if metric.lower().endswith(("_id", "_name")):
            continue
        pairs = [(re.sub(r"(^|_)x_team_", r"\1y_team_", metric, count=1),
                  re.sub(r"(^|_)x_team_", r"\1team_diff_", metric, count=1))]
        if "x_team_home_" in metric:
            pairs.append((metric.replace("x_team_home_", "y_team_away_", 1),
                          metric.replace("x_team_home_", "home_away_diff_", 1)))
        for other_metric, output_metric in pairs:
            counterpart, output = prefix + "__" + other_metric, prefix + "__" + output_metric
            if counterpart not in names or not (pd.api.types.is_numeric_dtype(frame[name])
                                                 and pd.api.types.is_numeric_dtype(frame[counterpart])):
                continue
            if output in names or output in differences:
                raise ValueError(f"Team-difference feature name collision: {output}")
            x = pd.to_numeric(frame[name], errors="raise").replace([np.inf, -np.inf], np.nan)
            y = pd.to_numeric(frame[counterpart], errors="raise").replace([np.inf, -np.inf], np.nan)
            differences[output] = x.astype(float) - y.astype(float)
    return pd.concat([frame, pd.DataFrame(differences, index=frame.index)], axis=1)


def feature_frame(frame, features):
    result = frame.loc[:, features].copy()
    result["season"] = result["season"].astype("string").fillna("__missing__").astype(str)
    for name in features:
        if name == "season":
            continue
        if pd.api.types.is_numeric_dtype(result[name]):
            result[name] = pd.to_numeric(result[name], errors="raise").replace([np.inf, -np.inf], np.nan)
        else:
            result[name] = result[name].astype("string").fillna("__missing__").astype(str)
    return result


def make_preprocessor(frame, features, scale):
    numerical = [c for c in features if c != "season" and pd.api.types.is_numeric_dtype(frame[c])]
    categorical = [c for c in features if c not in numerical]
    numerical_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))]
    if scale:
        numerical_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(numerical_steps), numerical),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ], sparse_threshold=0)


def traditional_grid(classification):
    from catboost import CatBoostClassifier, CatBoostRegressor
    from lightgbm import LGBMClassifier, LGBMRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.svm import SVC, SVR
    from xgboost import XGBClassifier, XGBRegressor

    forest = RandomForestClassifier if classification else RandomForestRegressor
    xgb = XGBClassifier if classification else XGBRegressor
    light = LGBMClassifier if classification else LGBMRegressor
    cat = CatBoostClassifier if classification else CatBoostRegressor
    grid = []
    for depth in (4, 8):
        grid.extend([
            ("random_forest", forest(n_estimators=100, max_depth=depth, random_state=42, n_jobs=1), False),
            ("xgboost", xgb(n_estimators=100, max_depth=depth, learning_rate=0.05, random_state=42, n_jobs=1), False),
            ("lightgbm", light(n_estimators=100, max_depth=depth, learning_rate=0.05, random_state=42, n_jobs=1, verbosity=-1), False),
            ("catboost", cat(iterations=100, depth=depth, learning_rate=0.05, random_seed=42, thread_count=1, verbose=False, allow_writing_files=False), False),
        ])
    for value in (0.1, 1.0):
        if classification:
            grid.extend([
                ("logistic_regression", LogisticRegression(C=value, max_iter=2000, random_state=42), True),
                ("svm", CalibratedClassifierCV(
                    estimator=SVC(C=value, random_state=42),
                    method="sigmoid", cv=RareClassCalibrationSplit(), ensemble=True, n_jobs=1,
                ), True),
            ])
        else:
            grid.extend([
                ("ridge", Ridge(alpha=value), True),
                ("svr", SVR(C=value), True),
            ])
    return grid


def _metrics(y, prediction, classification, classes=None):
    if not classification:
        prediction = np.asarray(prediction).reshape(-1)
        if len(prediction) != len(y) or not np.isfinite(prediction).all():
            raise ValueError("Regressor returned invalid predictions")
        return {"mae": float(mean_absolute_error(y, prediction)),
                "rmse": float(np.sqrt(mean_squared_error(y, prediction)))}
    probabilities = np.asarray(prediction, dtype=float)
    if (probabilities.ndim != 2 or probabilities.shape != (len(y), len(classes))
            or not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)):
        raise ValueError("Classifier returned invalid probabilities or an unexpected shape")
    actual = np.asarray(y).astype(str)
    lookup = {str(label): i for i, label in enumerate(classes)}
    indices = np.array([lookup.get(label, -1) for label in actual])
    correct_probability = np.full(len(actual), 1e-15)
    known = indices >= 0
    correct_probability[known] = probabilities[np.arange(len(actual))[known], indices[known]]
    return {
        "log_loss": float(-np.log(np.clip(correct_probability, 1e-15, 1)).mean()),
        "accuracy": float(accuracy_score(actual, np.asarray(classes)[probabilities.argmax(axis=1)].astype(str))),
        "unseen_target_count": int((~known).sum()),
    }


def sequence_splits(country_database, league_id, frames):
    fixtures = _read_fixture_tables(Path(country_database), league_id)
    sequences, _, _, rows, _, _ = _build_sequences(fixtures, HISTORY_LENGTH, set())
    if rows["fixture_id"].duplicated().any():
        raise ValueError("Raw fixture IDs are not unique")
    raw = rows.set_index("fixture_id")
    result = []
    for frame in frames:
        ids = frame["fixture_id"].tolist()
        if not set(ids).issubset(raw.index):
            raise ValueError("A prepared fixture is missing from raw sequence history")
        matched = raw.loc[ids]
        for column in ("home_team", "away_team", "home_goals", "away_goals", "date"):
            if not np.array_equal(matched[column].to_numpy(), frame[column].to_numpy()):
                raise ValueError(f"Raw and prepared fixtures differ in {column}; re-prepare datasets")
        positions = raw.index.get_indexer(ids)
        result.append({name: values[positions] for name, values in sequences.items()})
    return result


def make_network(architecture, input_size, classes, width):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(42)
    layers = tf.keras.layers
    classification = classes > 0
    if architecture in ("mlp", "dnn"):
        inputs = layers.Input(shape=(input_size,), name="features")
        encoded = layers.Dense(width, activation="relu")(inputs)
        if architecture == "dnn":
            encoded = layers.Dense(width, activation="relu")(encoded)
            encoded = layers.Dense(max(8, width // 2), activation="relu")(encoded)
    else:
        home = layers.Input(shape=(HISTORY_LENGTH, input_size), name="home_history")
        away = layers.Input(shape=(HISTORY_LENGTH, input_size), name="away_history")
        def encode(tensor):
            if architecture in ("lstm", "gru"):
                from training.deep_learning.masked_layers import MaskAwareGRU, MaskAwareLSTM

                masked = layers.Masking(mask_value=0.0)(tensor)
                return (MaskAwareLSTM(width) if architecture == "lstm" else MaskAwareGRU(width))(masked)
            if architecture == "tcn":
                x = layers.Conv1D(width, 3, padding="causal", dilation_rate=1, activation="relu")(tensor)
                x = layers.Conv1D(width, 3, padding="causal", dilation_rate=2, activation="relu")(x)
                return layers.Flatten()(x)
            x = layers.Dense(width)(tensor)
            attention = layers.MultiHeadAttention(num_heads=2, key_dim=max(1, width // 2))(x, x)
            x = layers.LayerNormalization()(layers.Add()([x, attention]))
            ff = layers.Dense(width, activation="relu")(x)
            return layers.Flatten()(layers.LayerNormalization()(layers.Add()([x, ff])))
        encoded = layers.Concatenate()([encode(home), encode(away)])
        encoded = layers.Dense(width, activation="relu")(encoded)
        inputs = {"home_history": home, "away_history": away}
    output = layers.Dense(classes if classification else 1,
                          activation="softmax" if classification else "linear")(encoded)
    model = tf.keras.Model(inputs, output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy" if classification else "mse")
    return model


def _network_fit(model, x, y, epochs, batch_size=64):
    n = len(y)
    rng = np.random.default_rng(42)
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            indices = order[start:start + batch_size]
            batch = {key: value[indices] for key, value in x.items()} if isinstance(x, dict) else x[indices]
            model.train_on_batch(batch, y[indices])


def _network_predict(model, x):
    return np.asarray(model(x, training=False))


def _write_metadata(database, league_id, rows, candidates, predictions, preparation_id):
    table = _quote(f"{int(league_id)}-market-model-runs")
    candidate_table = _quote(f"{int(league_id)}-market-model-candidates")
    prediction_table = _quote(f"{int(league_id)}-market-predictions")
    with duckdb.connect(str(database)) as connection:
        connection.execute("BEGIN")
        try:
            connection.execute(f"""CREATE TABLE IF NOT EXISTS {table} (
                run_id VARCHAR, created_at TIMESTAMP, target VARCHAR, algorithm VARCHAR,
                feature_bundle VARCHAR, features_json VARCHAR, parameters_json VARCHAR,
                validation_metrics_json VARCHAR, test_metrics_json VARCHAR, artifact_path VARCHAR,
                class_labels_json VARCHAR
            )""")
            connection.execute(f"""CREATE TABLE IF NOT EXISTS {candidate_table} (
                run_id VARCHAR, target VARCHAR, algorithm VARCHAR, feature_bundle VARCHAR,
                parameters_json VARCHAR, validation_metrics_json VARCHAR
            )""")
            connection.execute(f"""CREATE TABLE IF NOT EXISTS {prediction_table} (
                run_id VARCHAR, created_at TIMESTAMP, fixture_id BIGINT, target VARCHAR,
                prediction VARCHAR, probabilities_json VARCHAR
            )""")
            connection.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS preparation_id VARCHAR")
            current = connection.execute(
                f'SELECT preparation_id FROM {_quote(f"{int(league_id)}-dataset-metadata")}'
            ).fetchone()
            if not current or current[0] != preparation_id:
                raise ValueError("Datasets changed during training; restart model training")
            connection.executemany(
                f"INSERT INTO {table} VALUES (?, current_timestamp, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(*row, preparation_id) for row in rows],
            )
            connection.executemany(f"INSERT INTO {candidate_table} VALUES (?, ?, ?, ?, ?, ?)", candidates)
            if predictions:
                connection.executemany(
                    f"INSERT INTO {prediction_table} VALUES (?, current_timestamp, ?, ?, ?, ?)",
                    predictions,
                )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise


def _load_prediction_data(database, country_database, league_id):
    with connect_existing_database(database) as con:
        frame = con.execute(
            f'SELECT * FROM {_quote(f"{int(league_id)}-prediction-dataset")}'
        ).df()
    if frame.empty or frame["fixture_id"].isna().any() or frame["fixture_id"].duplicated().any():
        raise ValueError("Prediction dataset must contain unique fixture IDs and at least one row")
    fixtures = _read_fixture_tables(Path(country_database), league_id)
    ids = set(frame["fixture_id"])
    _, _, _, _, rows, sequences = _build_sequences(fixtures, HISTORY_LENGTH, ids)
    if rows.empty or rows["fixture_id"].duplicated().any() or set(rows["fixture_id"]) != ids:
        raise ValueError("Prediction fixtures need valid dates and team names in the raw fixture database")
    raw = rows.set_index("fixture_id")
    matched = raw.loc[frame["fixture_id"].tolist()]
    for column in ("home_team", "away_team", "date"):
        expected = fixture_dates_utc(frame[column]) if column == "date" else frame[column]
        if not np.array_equal(matched[column].to_numpy(), expected.to_numpy()):
            raise ValueError(f"Raw and prediction fixtures differ in {column}; re-prepare datasets")
    positions = raw.index.get_indexer(frame["fixture_id"])
    return add_team_differences(frame), {name: values[positions] for name, values in sequences.items()}


def _prediction_rows(run_id, target, fixture_ids, values, classes):
    result = []
    if classes is not None:
        probabilities = np.asarray(values)
        if target == "result" and not {"H", "D", "A"}.issubset(set(classes)):
            raise ValueError("Result training requires home, draw and away examples")
        # Validate shape and normalization without treating current-season scores as labels.
        _metrics(np.repeat(str(classes[0]), len(fixture_ids)), probabilities, True, classes)
        for fixture_id, row in zip(fixture_ids, probabilities):
            probabilities_by_label = {str(label): float(prob) for label, prob in zip(classes, row)}
            if target == "result" and {"H", "D", "A"}.issubset(probabilities_by_label):
                p = probabilities_by_label
                p.update({"1X": p["H"] + p["D"], "X2": p["D"] + p["A"], "12": p["H"] + p["A"]})
            if target.startswith("over_") and "1" in probabilities_by_label:
                probabilities_by_label[target.replace("over_", "under_", 1)] = 1 - probabilities_by_label["1"]
            result.append((run_id, int(fixture_id), target, str(classes[int(row.argmax())]),
                           json.dumps(probabilities_by_label)))
    else:
        values = np.asarray(values).reshape(-1)
        if len(values) != len(fixture_ids) or not np.isfinite(values).all():
            raise ValueError("Regressor returned invalid upcoming predictions")
        for fixture_id, value in zip(fixture_ids, values):
            result.append((run_id, int(fixture_id), target, str(float(value)), None))
    return result


def _publish_market(path, model, preprocessor, metadata, neural, database, league_id,
                    row, candidates, predictions, preparation_id):
    """Publish a complete artifact and its metadata; restore the old file on write failure."""
    bundle = {"metadata": metadata, "preprocessor": preprocessor, "neural": neural}
    with PUBLICATION_LOCK, tempfile.TemporaryDirectory(dir=path.parent) as temp:
        temp = Path(temp)
        if neural:
            keras_path = temp / "model.keras"
            model.save(keras_path)
            bundle["model"] = keras_path.read_bytes()
        else:
            bundle["model"] = model
        staged = temp / "artifact.joblib"
        joblib.dump(bundle, staged)
        backup = temp / "previous.joblib"
        existed = path.exists()
        if existed:
            shutil.copy2(path, backup)
        os.replace(staged, path)
        try:
            _write_metadata(database, league_id, [row], candidates, predictions, preparation_id)
        except Exception:
            if existed:
                os.replace(backup, path)
            else:
                path.unlink()
            raise


def load_market_artifact(path):
    """Load a published model and its matching preprocessing and feature metadata."""
    with PUBLICATION_LOCK:
        bundle = joblib.load(path)
    if bundle["neural"]:
        import tensorflow as tf
        from training.deep_learning import masked_layers  # Register custom Keras layers.

        with tempfile.TemporaryDirectory() as directory:
            keras_path = Path(directory) / "model.keras"
            keras_path.write_bytes(bundle["model"])
            bundle["model"] = tf.keras.models.load_model(keras_path)
    return bundle


def train_market_models(league_id, dataset_database, country_database, model_root=MODEL_ROOT,
                        progress_callback=None, *, algorithms=None, epochs=20, target_names=TARGETS,
                        market_callback=None):
    """Train candidates on training only, select on validation, evaluate test once."""
    import tensorflow as tf

    if epochs < 1:
        raise ValueError("epochs must be positive")
    target_names = tuple(target_names)
    if not target_names or len(set(target_names)) != len(target_names) or set(target_names) - set(TARGETS):
        raise ValueError("target_names must be a non-empty selection of unique supported targets")
    with connect_existing_database(dataset_database) as connection:
        metadata_table = _quote(f"{int(league_id)}-dataset-metadata")
        connection.execute(f"ALTER TABLE {metadata_table} ADD COLUMN IF NOT EXISTS preparation_id VARCHAR")
        connection.execute(
            f"UPDATE {metadata_table} SET preparation_id = ? WHERE preparation_id IS NULL",
            [uuid.uuid4().hex],
        )
        preparation_id = connection.execute(f"SELECT preparation_id FROM {metadata_table}").fetchone()[0]
    frames = load_splits(dataset_database, league_id)
    frames = [add_team_differences(frame) for frame in frames]
    bundles = feature_bundles(frames[0])
    targets = [build_market_targets(frame) for frame in frames]
    sequence = sequence_splits(country_database, league_id, frames)
    upcoming, upcoming_sequences = _load_prediction_data(dataset_database, country_database, league_id)
    run_id = uuid.uuid4().hex
    run_dir = Path(model_root) / str(int(league_id))
    run_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    enabled = lambda name: algorithms is None or name in algorithms
    counts = [
        len(bundles) * (sum(enabled(name) for name, _, _ in traditional_grid(target in CLASSIFICATION_TARGETS))
                        + 2 * sum(enabled(name) for name in ("mlp", "dnn")))
        + 2 * sum(enabled(name) for name in ("lstm", "gru", "tcn", "transformer"))
        for target in target_names
    ]
    if not all(counts):
        raise ValueError("No candidate algorithms selected for at least one target")
    total = sum(counts) + len(target_names)
    completed = 0

    def report(target, algorithm, skipped=False):
        nonlocal completed
        completed += 1
        if progress_callback:
            message = (f"Skipped non-converged {algorithm} for {target}."
                       if skipped else f"Evaluated {algorithm} for {target}.")
            progress_callback(completed, total, message)

    for target in target_names:
        candidates = []
        classification = target in CLASSIFICATION_TARGETS
        y = [t[target].to_numpy() for t in targets]
        labels = LabelEncoder().fit(y[0].astype(str)) if classification else None
        classes = labels.classes_ if labels is not None else None
        if classification and len(classes) < 2:
            raise ValueError(f"Training target {target} contains fewer than two classes")
        if target == "result" and not {"H", "D", "A"}.issubset(set(classes)):
            raise ValueError("Result training requires home, draw and away examples")
        train_y = labels.transform(y[0].astype(str)) if classification else y[0].astype(float)
        score_key = "log_loss" if classification else "mae"
        best_score = float("inf")
        best = None
        grid = traditional_grid(classification)
        for bundle, features in bundles.items():
            if not any(enabled(name) for name, _, _ in grid) and not any(
                enabled(name) for name in ("mlp", "dnn")
            ):
                continue
            data = [feature_frame(frame, features) for frame in frames[:2]]
            missing = data[0].isna().sum()
            LOGGER.info("%s: %d missing training feature values across %d columns", bundle, int(missing.sum()), int((missing > 0).sum()))
            prepared = {}
            for scale in (False, True):
                if not any(enabled(name) and candidate_scale == scale for name, _, candidate_scale in grid) and not (
                    scale and any(enabled(name) for name in ("mlp", "dnn"))
                ):
                    continue
                preprocessor = make_preprocessor(frames[0], features, scale)
                train_x = preprocessor.fit_transform(data[0]).astype(np.float32)
                val_x = preprocessor.transform(data[1]).astype(np.float32)
                prepared[scale] = (preprocessor, train_x, val_x)
            for algorithm, estimator, scale in grid:
                if not enabled(algorithm):
                    continue
                estimator = clone(estimator)
                preprocessor, train_x, val_x = prepared[scale]
                if algorithm == "svm":
                    _, class_counts = np.unique(train_y, return_counts=True)
                    if (class_counts == 1).any():
                        LOGGER.warning(
                            "%s %s: %d singleton classes cannot be calibrated from held-out examples",
                            target, bundle, int((class_counts == 1).sum()),
                        )
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error", category=ConvergenceWarning)
                        estimator.fit(train_x, train_y)
                except ConvergenceWarning as error:
                    LOGGER.warning(
                        "Skipping non-converged %s candidate for %s (%s): %s",
                        algorithm, target, bundle, error,
                    )
                    candidates.append((run_id, target, algorithm, bundle,
                                       json.dumps(estimator.get_params(), default=str, sort_keys=True),
                                       json.dumps({"status": "not_converged", "reason": str(error)})))
                    report(target, algorithm, skipped=True)
                    continue
                pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
                pred = estimator.predict_proba(val_x) if classification else estimator.predict(val_x)
                metrics = _metrics(y[1], pred, classification, classes)
                candidates.append((run_id, target, algorithm, bundle,
                                   json.dumps(estimator.get_params(), default=str, sort_keys=True),
                                   json.dumps(metrics)))
                if metrics[score_key] < best_score:
                    best_score = metrics[score_key]
                    best = (algorithm, bundle, features, estimator.get_params(), metrics, pipeline, None)
                report(target, algorithm)
            for architecture in ("mlp", "dnn"):
                if not enabled(architecture):
                    continue
                preprocessor, train_x, val_x = prepared[True]
                for width in (32, 64):
                    model = make_network(architecture, train_x.shape[1], len(classes) if classification else 0, width)
                    _network_fit(model, train_x, train_y, epochs)
                    metrics = _metrics(y[1], _network_predict(model, val_x), classification, classes)
                    candidates.append((run_id, target, architecture, bundle,
                                       json.dumps({"width": width, "epochs": epochs}), json.dumps(metrics)))
                    if metrics[score_key] < best_score:
                        best_score = metrics[score_key]
                        best = (architecture, bundle, features, {"width": width, "epochs": epochs}, metrics, model, preprocessor)
                    report(target, architecture)
        for architecture in ("lstm", "gru", "tcn", "transformer"):
            if not enabled(architecture):
                continue
            for width in (32, 64):
                model = make_network(architecture, sequence[0]["home_history"].shape[-1], len(classes) if classification else 0, width)
                _network_fit(model, sequence[0], train_y, epochs)
                metrics = _metrics(y[1], _network_predict(model, sequence[1]), classification, classes)
                candidates.append((run_id, target, architecture, "team_history",
                                   json.dumps({"width": width, "epochs": epochs, "history_length": HISTORY_LENGTH}),
                                   json.dumps(metrics)))
                if metrics[score_key] < best_score:
                    best_score = metrics[score_key]
                    best = (architecture, "team_history", ["goals_for", "goals_against", "was_home", "points", "days_since_last_match", "has_match"],
                            {"width": width, "epochs": epochs, "history_length": HISTORY_LENGTH}, metrics, model, None)
                report(target, architecture)
        if best is None:
            raise ValueError(f"No candidate model converged for {target}")
        algorithm, bundle, features, parameters, validation_metrics, model, preprocessor = best
        if bundle == "team_history":
            test_prediction = _network_predict(model, sequence[2])
        elif preprocessor is not None:
            test_x = preprocessor.transform(feature_frame(frames[2], features)).astype(np.float32)
            test_prediction = _network_predict(model, test_x)
        else:
            test_x = feature_frame(frames[2], features)
            test_prediction = model.predict_proba(test_x) if classification else model.predict(test_x)
        test_metrics = _metrics(y[2], test_prediction, classification, classes)
        neural = algorithm in ("mlp", "dnn", "lstm", "gru", "tcn", "transformer")
        artifact = run_dir / (target + ".joblib")
        if bundle == "team_history":
            upcoming_prediction = _network_predict(model, upcoming_sequences)
        elif preprocessor is not None:
            upcoming_x = preprocessor.transform(feature_frame(upcoming, features)).astype(np.float32)
            upcoming_prediction = _network_predict(model, upcoming_x)
        else:
            upcoming_x = feature_frame(upcoming, features)
            upcoming_prediction = model.predict_proba(upcoming_x) if classification else model.predict(upcoming_x)
        predictions = _prediction_rows(run_id, target, upcoming["fixture_id"], upcoming_prediction, classes)
        params_json = json.dumps(parameters, default=str, sort_keys=True)
        row = (run_id, target, algorithm, bundle, json.dumps(features), params_json,
               json.dumps(validation_metrics), json.dumps(test_metrics), str(artifact.resolve()),
               json.dumps(classes.tolist() if classification else []))
        _publish_market(artifact, model, preprocessor, {
            "target": target, "run_id": run_id, "algorithm": algorithm, "features": features,
            "feature_bundle": bundle, "classes": classes.tolist() if classification else [],
            "parameters": parameters,
        }, neural, dataset_database, league_id, row, candidates, predictions, preparation_id)
        selected.append({"target": target, "algorithm": algorithm, "feature_bundle": bundle,
                         "validation": validation_metrics, "test": test_metrics})
        if market_callback:
            market_callback(selected[-1], len(predictions))
        completed += 1
        if progress_callback:
            progress_callback(completed, total, f"Published {target} model and predictions.")
        tf.keras.backend.clear_session()
    return {"run_id": run_id, "models": selected, "message": f"Saved {len(selected)} selected market models."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league-id", type=int, required=True)
    parser.add_argument("--dataset-database", type=Path, required=True)
    parser.add_argument("--country-database", type=Path, required=True)
    args = parser.parse_args()
    result = train_market_models(args.league_id, args.dataset_database, args.country_database,
                                 progress_callback=lambda done, total, msg: print(f"{done}/{total}: {msg}", flush=True))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
