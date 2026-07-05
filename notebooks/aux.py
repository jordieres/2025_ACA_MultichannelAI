import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.dummy import DummyClassifier

from scipy.stats import wilcoxon

import matplotlib.pyplot as plt


def evaluate_model_cv(model, X, y, groups, cv, model_name, feature_set_name,
                      threshold_grid=np.arange(0.2, 0.7, 0.01)):
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        fitted_model = clone(model)
        fitted_model.fit(X_train, y_train)

        roc_auc = np.nan
        best_threshold = 0.5

        if hasattr(fitted_model, "predict_proba"):
            y_proba_train = fitted_model.predict_proba(X_train)[:, 1]
            y_proba_test = fitted_model.predict_proba(X_test)[:, 1]

            train_f1s = [
                f1_score(y_train, (y_proba_train >= t).astype(int), average="macro", zero_division=0)
                for t in threshold_grid
            ]
            best_threshold = threshold_grid[np.argmax(train_f1s)]

            y_pred = (y_proba_test >= best_threshold).astype(int)
            roc_auc = roc_auc_score(y_test, y_proba_test)
        else:
            y_pred = fitted_model.predict(X_test)

        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        fold_rows.append({
            "model_name": model_name,
            "feature_set": feature_set_name,
            "fold": fold,
            "f1_macro": f1,
            "roc_auc": roc_auc,
            "threshold": best_threshold,
        })

    fold_df = pd.DataFrame(fold_rows)

    summary = {
        "model_name": model_name,
        "feature_set": feature_set_name,
        "f1_macro_mean": fold_df["f1_macro"].mean(),
        "f1_macro_std": fold_df["f1_macro"].std(ddof=0),
        "roc_auc_mean": fold_df["roc_auc"].mean(skipna=True),
        "roc_auc_std": fold_df["roc_auc"].std(ddof=0, skipna=True),
    }

    return fold_df, summary

def evaluate_dummy_majority_cv(y, groups, cv):
    fold_rows = []

    dummy = DummyClassifier(strategy="most_frequent")

    X_dummy = pd.DataFrame(index=y.index)  # X vacío solo para que cv.split tenga longitud consistente

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_dummy, y, groups=groups), start=1):
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        dummy_fold = clone(dummy)
        dummy_fold.fit(np.zeros((len(train_idx), 1)), y_train)

        y_pred = dummy_fold.predict(np.zeros((len(test_idx), 1)))
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        fold_rows.append({
            "model_name": "DummyMajority",
            "feature_set": "BASELINE",
            "fold": fold,
            "f1_macro": f1,
            "roc_auc": np.nan,
            "threshold": np.nan,
        })

    fold_df = pd.DataFrame(fold_rows)

    summary = {
        "model_name": "DummyMajority",
        "feature_set": "BASELINE",
        "f1_macro_mean": fold_df["f1_macro"].mean(),
        "f1_macro_std": fold_df["f1_macro"].std(ddof=0),
        "roc_auc_mean": np.nan,
        "roc_auc_std": np.nan,
    }

    return fold_df, summary


def wilcoxon_comparison(df, model_name, fs_a, fs_b):
    df_model = df[df["model_name"] == model_name]

    a = df_model[df_model["feature_set"] == fs_a].sort_values("fold")
    b = df_model[df_model["feature_set"] == fs_b].sort_values("fold")

    assert all(a["fold"].values == b["fold"].values)

    x = a["f1_macro"].values
    y = b["f1_macro"].values

    stat, p = wilcoxon(y, x)
    diff = y - x

    print(f"{model_name}: {fs_b} vs {fs_a}")
    print(f"Mean diff: {diff.mean():.4f}")
    print(f"Differences: {diff}")
    print(f"Wilcoxon p-value: {p:.4f}")
    print("-" * 40)

    return stat, p, diff


def bootstrap_comparison(
    df,
    model_name,
    fs_a,
    fs_b,
    metric="f1_macro",
    n_boot=10000,
    ci=95,
    random_state=42,
    verbose=True
):
    """
    Bootstrap de la diferencia emparejada entre dos feature sets
    para un mismo modelo y misma métrica.

    Compara fs_b - fs_a.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con resultados fold a fold.
    model_name : str
        Nombre del modelo a filtrar.
    fs_a : str
        Feature set de referencia.
    fs_b : str
        Feature set que se compara contra fs_a.
    metric : str
        Métrica a comparar, por ejemplo 'f1_macro' o 'roc_auc'.
    n_boot : int
        Número de remuestreos bootstrap.
    ci : float
        Nivel del intervalo de confianza, por ejemplo 95.
    random_state : int
        Semilla para reproducibilidad.
    verbose : bool
        Si True, imprime resumen.

    Returns
    -------
    result : dict
        Diccionario con diferencias por fold, media observada,
        distribución bootstrap e intervalo de confianza.
    """

    df_model = df[df["model_name"] == model_name].copy()

    a = (
        df_model[df_model["feature_set"] == fs_a]
        .sort_values("fold")
        .reset_index(drop=True)
    )
    b = (
        df_model[df_model["feature_set"] == fs_b]
        .sort_values("fold")
        .reset_index(drop=True)
    )

    if len(a) == 0 or len(b) == 0:
        raise ValueError("No hay filas para alguna de las combinaciones pedidas.")

    if not np.array_equal(a["fold"].values, b["fold"].values):
        raise ValueError("Los folds no están alineados entre ambos feature sets.")

    x = a[metric].to_numpy()
    y = b[metric].to_numpy()

    if np.isnan(x).any() or np.isnan(y).any():
        raise ValueError(f"La métrica '{metric}' contiene NaN en alguna comparación.")

    diff = y - x
    observed_mean = diff.mean()

    rng = np.random.default_rng(random_state)
    boot_means = np.empty(n_boot)

    n = len(diff)
    for i in range(n_boot):
        sample = rng.choice(diff, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = 100 - ci
    lower = np.percentile(boot_means, alpha / 2)
    upper = np.percentile(boot_means, 100 - alpha / 2)

    result = {
        "model_name": model_name,
        "metric": metric,
        "comparison": f"{fs_b} vs {fs_a}",
        "fs_a": fs_a,
        "fs_b": fs_b,
        "folds": a["fold"].tolist(),
        "diff_per_fold": diff,
        "observed_mean_diff": observed_mean,
        "ci_level": ci,
        "ci_lower": lower,
        "ci_upper": upper,
        "bootstrap_means": boot_means,
        "n_boot": n_boot,
    }

    if verbose:
        print(f"{model_name}: {fs_b} vs {fs_a} | metric={metric}")
        print(f"Mean diff: {observed_mean:.6f}")
        print(f"{ci:.0f}% CI: ({lower:.6f}, {upper:.6f})")
        print(f"Differences: {diff}")
        print("-" * 50)

    return result

def get_oof_predictions(model, X, y, groups, cv):
    oof_rows = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model_fold = clone(model)
        model_fold.fit(X_train, y_train)

        if hasattr(model_fold, "predict_proba"):
            y_proba_train = model_fold.predict_proba(X_train)[:, 1]
            y_proba_test = model_fold.predict_proba(X_test)[:, 1]

            thresholds = np.arange(0.2, 0.7, 0.01)
            f1s = [
                f1_score(y_train, (y_proba_train >= t).astype(int), average="macro", zero_division=0)
                for t in thresholds
            ]
            best_t = thresholds[np.argmax(f1s)]

            y_pred = (y_proba_test >= best_t).astype(int)
        else:
            y_pred = model_fold.predict(X_test)
            y_proba_test = np.full(len(y_test), np.nan)

        for i, idx in enumerate(test_idx):
            oof_rows.append({
                "idx": idx,
                "fold": fold,
                "y_true": y_test.iloc[i],
                "y_pred": y_pred[i],
                "y_proba": y_proba_test[i]
            })

    oof_df = pd.DataFrame(oof_rows).sort_values("idx")
    return oof_df

def evaluate_by_group(df, group_col):
    results = []

    for group, subdf in df.groupby(group_col):
        if len(subdf) < 30:
            continue

        f1 = f1_score(subdf["y_true"], subdf["y_pred"], average="macro")

        try:
            roc = roc_auc_score(subdf["y_true"], subdf["y_proba"])
        except:
            roc = np.nan

        results.append({
            group_col: group,
            "n": len(subdf),
            "f1_macro": f1,
            "roc_auc": roc
        })

    return pd.DataFrame(results).sort_values("f1_macro", ascending=False)


def plot_calls_metric(df, metric="f1_macro", n_col="n"):
    plot_df = df.copy().sort_values("num_calls_total")

    plt.figure(figsize=(10, 5))
    plt.plot(plot_df["num_calls_total"], plot_df[metric], marker="o")
    plt.xlabel("num_calls_total")
    plt.ylabel(metric)
    plt.title(f"{metric} by num_calls_total")
    plt.grid(alpha=0.3)

    for _, row in plot_df.iterrows():
        plt.text(row["num_calls_total"], row[metric] + 0.003, str(int(row[n_col])), ha="center", fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_group_bars(df, group_col, metric="f1_macro", n_col="n", sort_by_metric=True):
    plot_df = df.copy()

    if sort_by_metric:
        plot_df = plot_df.sort_values(metric, ascending=False)
    else:
        plot_df = plot_df.sort_values(group_col)

    plt.figure(figsize=(10, 5))
    plt.bar(plot_df[group_col].astype(str), plot_df[metric])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric)
    plt.xlabel(group_col)
    plt.title(f"{metric} by {group_col}")

    # anotar n
    for i, (_, row) in enumerate(plot_df.iterrows()):
        plt.text(i, row[metric] + 0.002, f"n={row[n_col]}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.show()