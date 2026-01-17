import io
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# XGBoost opsional
try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


# =========================
# Konfigurasi kolom (sesuai notebook Anda)
# =========================
NUMERIC_COLS: List[str] = [
    "IPK",
    "Pendapatan_Orang_Tua",
    "Keikutsertaan_Organisasi",
    "Pengalaman_Sosial",
    "Prestasi_Akademik",
    "Prestasi_Non_Akademik",
]

CATEGORICAL_COLS: List[str] = [
    "Asal_Sekolah",
    "Lokasi_Domisili",
    "Gender",
    "Status_Disabilitas",
]

DEFAULT_TARGET = "Diterima_Beasiswa"


# =========================
# Util
# =========================
def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


@st.cache_data(show_spinner=False)
def _load_sample() -> Optional[pd.DataFrame]:
    """Muat data contoh yang disertakan (beasiswa.csv)."""
    candidates = [
        "beasiswa.csv",
        os.path.join(os.path.dirname(__file__), "beasiswa.csv"),
        os.path.join(os.getcwd(), "beasiswa.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


def _read_uploaded(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file)
    # fallback: coba baca sebagai csv
    return pd.read_csv(file)


def _expected_columns_text() -> str:
    return (
        "Fitur numerik: "
        + ", ".join(NUMERIC_COLS)
        + "\n\nFitur kategorikal: "
        + ", ".join(CATEGORICAL_COLS)
        + f"\n\nLabel/target (untuk training): {DEFAULT_TARGET} (0/1)"
    )


@dataclass
class CleaningLog:
    rows_before: int
    rows_after: int
    dropped_duplicates: int
    coerced_numeric: List[str]
    imputed_numeric: Dict[str, float]
    imputed_categorical: Dict[str, str]


def _style_clean_audit_row(row: pd.Series) -> List[str]:
    status = str(row.get("Status", ""))
    if status == "DIHAPUS":
        bg = "#FEE2E2"  # merah muda
    elif status == "DIUPDATE":
        bg = "#FEF9C3"  # kuning muda
    else:
        bg = ""
    if not bg:
        return ["" for _ in row]
    return [f"background-color: {bg};" for _ in row]

def _reorder_audit_cols(df: pd.DataFrame) -> pd.DataFrame:
    front = ["Status", "Keterangan", "RowID"]
    present_front = [c for c in front if c in df.columns]
    rest = [c for c in df.columns if c not in present_front]
    return df[present_front + rest]


def clean_dataset(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    target_col: Optional[str] = None,
    drop_rows_missing_target: bool = True,
    numeric_impute: str = "median",  # "median" atau "mean"
    categorical_impute: str = "mode",  # hanya "mode" untuk saat ini
) -> Tuple[pd.DataFrame, CleaningLog, pd.DataFrame]:
    """Cleaning yang aman untuk berbagai input file.

    - Trim nama kolom & string kategorikal
    - Drop duplikat
    - Coerce numerik
    - Imputasi missing value (numeric median/mean, kategorikal mode)
    """
    work = df.copy()

    # Strip whitespace nama kolom
    work.columns = [str(c).strip() for c in work.columns]

    removed_parts: List[pd.DataFrame] = []

    # Drop duplikat + simpan baris yang terhapus
    before = len(work)
    dup_mask = work.duplicated()
    if dup_mask.any():
        removed_dups = work.loc[dup_mask].copy()
        removed_dups.insert(0, "Status", "DIHAPUS")
        removed_dups.insert(1, "Keterangan", "Duplikat")
        removed_dups = removed_dups.reset_index().rename(columns={"index": "RowID"})
        removed_dups = _reorder_audit_cols(removed_dups)
        removed_parts.append(removed_dups)
    work = work.loc[~dup_mask].copy()
    after_dedup = len(work)
    dropped_dup = before - after_dedup

    # Trim string kategorikal (tetap jaga NaN agar bisa diimputasi)
    for c in categorical_cols:
        if c in work.columns:
            work[c] = work[c].astype("string").str.strip()
            work[c] = work[c].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    # Coerce numerik
    coerced = []
    for c in numeric_cols:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")
            coerced.append(c)

    # (Opsional) buang baris tanpa target
    if target_col and target_col in work.columns and drop_rows_missing_target:
        miss_mask = work[target_col].isna()
        if miss_mask.any():
            removed_target = work.loc[miss_mask].copy()
            removed_target.insert(0, "Status", "DIHAPUS")
            removed_target.insert(1, "Keterangan", f"Target kosong ({target_col})")
            removed_target = removed_target.reset_index().rename(columns={"index": "RowID"})
            removed_target = _reorder_audit_cols(removed_target)
            removed_parts.append(removed_target)
        work = work.loc[~miss_mask].copy()

    # Catat baris yang berubah karena imputasi
    imputed_cols_by_row: Dict[object, List[str]] = defaultdict(list)

    # Imputasi numeric
    imputed_num: Dict[str, float] = {}
    for c in numeric_cols:
        if c in work.columns:
            if work[c].isna().any():
                miss = work[c].isna()
                if numeric_impute == "mean":
                    val = float(work[c].mean())
                else:
                    val = float(work[c].median())
                work[c] = work[c].fillna(val)
                imputed_num[c] = val
                for rid in work.index[miss].tolist():
                    imputed_cols_by_row[rid].append(c)

    # Imputasi kategorikal (mode)
    imputed_cat: Dict[str, str] = {}
    for c in categorical_cols:
        if c in work.columns:
            if work[c].isna().any():
                miss = work[c].isna()
                mode = work[c].mode(dropna=True)
                fill = str(mode.iloc[0]) if len(mode) else "Tidak Diketahui"
                work[c] = work[c].fillna(fill)
                imputed_cat[c] = fill
                for rid in work.index[miss].tolist():
                    imputed_cols_by_row[rid].append(c)

    # Audit table: gabungan baris DIHAPUS dan DIUPDATE
    updated_df = pd.DataFrame()
    if imputed_cols_by_row:
        updated_ids = list(imputed_cols_by_row.keys())
        updated_df = work.loc[work.index.isin(updated_ids)].copy()
        updated_df.insert(0, "Status", "DIUPDATE")
        updated_df.insert(
            1,
            "Keterangan",
            [
                "Imputasi: " + ", ".join(imputed_cols_by_row.get(idx, []))
                for idx in updated_df.index.tolist()
            ],
        )
        updated_df = updated_df.reset_index().rename(columns={"index": "RowID"})
        updated_df = _reorder_audit_cols(updated_df)

    if removed_parts or (not updated_df.empty):
        audit_df = pd.concat([*removed_parts, updated_df], ignore_index=True, sort=False)
    else:
        audit_df = pd.DataFrame(columns=["Status", "Keterangan", "RowID"]) 

    log = CleaningLog(
        rows_before=len(df),
        rows_after=len(work),
        dropped_duplicates=dropped_dup,
        coerced_numeric=coerced,
        imputed_numeric=imputed_num,
        imputed_categorical=imputed_cat,
    )
    return work, log, audit_df


def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )


def get_model_candidates(random_state: int, class_weight_balanced: bool) -> Dict[str, Pipeline]:
    preprocessor = build_preprocessor(NUMERIC_COLS, CATEGORICAL_COLS)

    lr = LogisticRegression(
        max_iter=2000,
        class_weight=("balanced" if class_weight_balanced else None),
        n_jobs=None,
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        class_weight=("balanced" if class_weight_balanced else None),
    )

    candidates: Dict[str, Pipeline] = {
        "Logistic Regression": Pipeline(steps=[("preprocess", preprocessor), ("model", lr)]),
        "Random Forest": Pipeline(steps=[("preprocess", preprocessor), ("model", rf)]),
    }

    if HAS_XGBOOST:
        xgb = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            eval_metric="logloss",
        )
        candidates["XGBoost"] = Pipeline(steps=[("preprocess", preprocessor), ("model", xgb)])

    return candidates


@dataclass
class EvalResult:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    cm: np.ndarray
    report: str
    report_dict: Dict[str, Dict[str, Any]]
    fitted: Pipeline


def train_and_evaluate(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int,
    selected_models: List[str],
    class_weight_balanced: bool,
) -> List[EvalResult]:
    X = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    y = df[target_col].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    candidates = get_model_candidates(random_state=random_state, class_weight_balanced=class_weight_balanced)
    candidates = {k: v for k, v in candidates.items() if k in selected_models}

    results: List[EvalResult] = []
    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred)

        report = classification_report(y_test, y_pred, digits=4, zero_division=0)
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        results.append(
            EvalResult(
                name=name,
                accuracy=acc,
                precision=prec,
                recall=rec,
                f1=f1,
                cm=cm,
                report=report,
                report_dict=report_dict,
                fitted=pipe,
            )
        )
    return results


def pick_best_model(results: List[EvalResult], metric: str) -> Optional[EvalResult]:
    if not results:
        return None
    key = {
        "Accuracy": lambda r: r.accuracy,
        "Precision": lambda r: r.precision,
        "Recall": lambda r: r.recall,
        "F1-score": lambda r: r.f1,
    }[metric]
    return sorted(results, key=key, reverse=True)[0]


# =========================
# UI Helpers (Plotting)
# =========================
def _mpl_soft_rcparams() -> Dict[str, object]:
    return {
        "figure.facecolor": "none",
        "axes.facecolor": "white",
        "axes.edgecolor": "#E5E7EB",
        "axes.labelcolor": "#374151",
        "xtick.color": "#6B7280",
        "ytick.color": "#6B7280",
        "grid.color": "#E5E7EB",
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
        "axes.grid": True,
    }


def plot_hist(df: pd.DataFrame, col: str, bins: int, container=None):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        st.error(
            "Fitur plot membutuhkan matplotlib. Install dulu dengan `pip install matplotlib` (atau `pip install -r requirements.txt`)."
        )
        return

    values = df[col].dropna().to_numpy()
    with plt.rc_context(_mpl_soft_rcparams()):
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        counts, bin_edges = np.histogram(values, bins=bins)
        widths = np.diff(bin_edges)
        centers = bin_edges[:-1]
        # Gradasi lembut (pastel) per batang
        cmap = plt.cm.Blues
        colors = cmap(np.linspace(0.35, 0.85, max(len(counts), 1)))
        ax.bar(
            centers,
            counts,
            width=widths,
            align="edge",
            color=colors,
            alpha=0.65,
            edgecolor="white",
            linewidth=1.0,
        )
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.7)
        fig.tight_layout()
    if container is None:
        st.pyplot(fig, clear_figure=True)
    else:
        container.pyplot(fig, clear_figure=True)


def plot_bar_colored(df_counts: pd.DataFrame, x_col: str, y_col: str, title: Optional[str] = None, height: int = 260):
    """Bar chart dengan warna berbeda per kategori (fallback ke st.bar_chart jika altair tidak ada)."""
    try:
        import altair as alt
    except ModuleNotFoundError:
        try:
            s = pd.Series(df_counts[y_col].to_numpy(), index=df_counts[x_col].astype(str))
            st.bar_chart(s)
        except Exception:
            st.dataframe(df_counts, use_container_width=True)
        return

    chart = (
        alt.Chart(df_counts)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, opacity=0.88)
        .encode(
            x=alt.X(f"{x_col}:N", title=None, sort=None),
            y=alt.Y(f"{y_col}:Q", title=None),
            color=alt.Color(
                f"{x_col}:N",
                legend=None,
                scale=alt.Scale(scheme="pastel1"),
            ),
            tooltip=[alt.Tooltip(f"{x_col}:N"), alt.Tooltip(f"{y_col}:Q")],
        )
        .properties(height=height)
    )
    if title:
        chart = chart.properties(title=title)
    chart = (
        chart
        .configure_view(strokeOpacity=0)
        .configure_axis(
            gridColor="#E9ECF5",
            domainColor="#E5E7EB",
            tickColor="#E5E7EB",
            labelColor="#6B7280",
            titleColor="#6B7280",
        )
    )
    st.altair_chart(chart, use_container_width=True)


def plot_correlation(df: pd.DataFrame, cols: List[str]):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        st.error(
            "Fitur plot membutuhkan matplotlib. Install dulu dengan `pip install matplotlib` (atau `pip install -r requirements.txt`)."
        )
        return

    corr = df[cols].corr(numeric_only=True)
    with plt.rc_context(_mpl_soft_rcparams()):
        # Dibuat lebih kecil dan lembut
        fig, ax = plt.subplots(figsize=(5.6, 3.9))
        base_font = 8.5
        half_font = max(4.0, base_font * 0.5)
        # Gradasi hijau lembut (kuning→hijau). Negatif akan lebih pucat.
        im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="YlGn", interpolation="nearest")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=half_font)
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(cols, fontsize=half_font)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Nilai korelasi", fontsize=half_font)
        cbar.ax.tick_params(labelsize=half_font)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def _soft_teal_cmap():
    try:
        from matplotlib.colors import LinearSegmentedColormap

        return LinearSegmentedColormap.from_list(
            "soft_teal",
            ["#F5FFFE", "#E8FBF8", "#D3F4EF", "#B8EAE3", "#8ED7CE"],
        )
    except Exception:
        return "PuBuGn"


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str,
    cmap=_soft_teal_cmap(),
    figsize: tuple[float, float] = (4.8, 3.8),
    font_scale: float = 1.0,
):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        st.error(
            "Fitur plot membutuhkan matplotlib. Install dulu dengan `pip install matplotlib` (atau `pip install -r requirements.txt`)."
        )
        return

    with plt.rc_context(_mpl_soft_rcparams()):
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(cm, cmap=cmap, alpha=0.82)
        ax.set_title(title, fontsize=11 * font_scale)
    ax.set_xlabel("Predicted", fontsize=10 * font_scale)
    ax.set_ylabel("Actual", fontsize=10 * font_scale)
    ax.tick_params(axis="both", labelsize=9 * font_scale)

    # Pastikan tick tidak muncul -0.5..1.5 dan label sesuai kelas
    n_rows, n_cols = cm.shape
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels([str(i) for i in range(n_cols)], fontsize=9 * font_scale)
    ax.set_yticklabels([str(i) for i in range(n_rows)], fontsize=9 * font_scale)

    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center", fontsize=10 * font_scale)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def apply_soft_theme():
        st.markdown(
                """
<style>
:root{
    --bg1:#f7f9ff;
    --bg2:#fff7fb;
    --card:rgba(255,255,255,.72);
    --cardBorder:rgba(20,24,40,.08);
    --text:#111827;
    --muted:#6b7280;
    --primary:#5b8cff;
    --primary2:#8a6cff;
    --radius:16px;
}

.stApp{
    background:
        radial-gradient(1200px 600px at 15% 10%, rgba(91,140,255,0.18), transparent 55%),
        radial-gradient(900px 500px at 85% 15%, rgba(138,108,255,0.14), transparent 50%),
        radial-gradient(900px 500px at 60% 90%, rgba(255,140,181,0.10), transparent 55%),
        linear-gradient(135deg, var(--bg1), var(--bg2));
    color: var(--text);
}

[data-testid="stHeader"], [data-testid="stToolbar"]{ background: transparent; }

.block-container{
    padding-top: 2.25rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"]{
    background:
        radial-gradient(420px 320px at 10% 5%, rgba(91,140,255,0.18), transparent 60%),
        radial-gradient(380px 260px at 85% 0%, rgba(138,108,255,0.14), transparent 55%),
        linear-gradient(180deg, rgba(248,251,255,0.95), rgba(244,246,255,0.9));
    border-right: 1px solid rgba(91,140,255,0.18);
    backdrop-filter: blur(16px);
}

[data-testid="stSidebar"] .block-container{
    padding-top: 0.9rem;
    padding-bottom: 0.9rem;
}

[data-testid="stSidebar"]::after{
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    background-image:
        radial-gradient(rgba(91,140,255,0.12) 1px, transparent 1px);
    background-size: 16px 16px;
    opacity: 0.25;
}

.sidebar-brand{
    display: flex;
    align-items: center;
    gap: 0.85rem;
    padding: 0.15rem 0.15rem 0.55rem;
}

.sidebar-spacer{
    height: 0.85rem;
}

.ml-logo{
    width: 52px;
    height: 52px;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(91,140,255,0.28), rgba(138,108,255,0.22));
    border: 1px solid rgba(91,140,255,0.25);
    display: grid;
    place-items: center;
    box-shadow: 0 16px 34px rgba(91,140,255,0.22);
}

.sidebar-appname{
    font-weight: 820;
    letter-spacing: -0.02em;
    font-size: 1.05rem;
    color: var(--text);
    line-height: 1.1;
}

.sidebar-tag{
    font-size: 0.9rem;
    color: rgba(59,66,86,0.75);
    margin-top: 0.1rem;
}

.sidebar-section-title{
    font-size: 1.1rem;
    font-weight: 800;
    color: var(--text);
    margin: 0.15rem 0 0.1rem;
    letter-spacing: -0.01em;
}

.sidebar-section-subtitle{
    font-size: 0.9rem;
    color: var(--muted);
    margin-bottom: 0.35rem;
}

.sidebar-card{
    background: rgba(255,255,255,0.65);
    border: 1px solid var(--cardBorder);
    border-radius: var(--radius);
    padding: 0.7rem 0.75rem;
    box-shadow: 0 10px 28px rgba(17,24,39,0.05);
}

.sidebar-card-title{
    font-weight: 800;
    color: var(--text);
    margin-bottom: 0.25rem;
    letter-spacing: -0.01em;
}

.sidebar-notes{
    margin: 0.25rem 0 0;
    padding-left: 1.1rem;
    color: var(--muted);
}

.sidebar-notes li{
    margin: 0.25rem 0;
}

/* Radio menu styling in sidebar */
[data-testid="stSidebar"] label[data-baseweb="radio"]{
    background: rgba(255,255,255,0.7);
    border: 1px solid rgba(91,140,255,0.14);
    border-radius: 14px;
    padding: 0.42rem 0.6rem;
    margin: 0.12rem 0;
    transition: background .12s ease, border-color .12s ease, transform .08s ease, box-shadow .12s ease;
}

[data-testid="stSidebar"] label[data-baseweb="radio"]:hover{
    background: rgba(255,255,255,0.88);
    border-color: rgba(91,140,255,0.32);
    box-shadow: 0 12px 26px rgba(91,140,255,0.12);
    transform: translateY(-1px);
}

[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked){
    background: linear-gradient(135deg, rgba(91,140,255,0.22), rgba(138,108,255,0.18));
    border-color: rgba(91,140,255,0.45);
    box-shadow: 0 12px 28px rgba(91,140,255,0.18);
}

[data-testid="stSidebar"] label[data-baseweb="radio"] *{
    color: var(--text);
    font-weight: 650;
}

/* Divider in sidebar */
[data-testid="stSidebar"] hr{
    border-color: rgba(17,24,39,0.08);
    margin: 0.65rem 0;
}

/* Compact mode for shorter viewports */
@media (max-height: 820px){
    [data-testid="stSidebar"] .block-container{ padding-top: 0.6rem; padding-bottom: 0.6rem; }
    .ml-logo{ width: 46px; height: 46px; border-radius: 14px; }
    .sidebar-appname{ font-size: 1.0rem; }
    .sidebar-tag{ font-size: 0.85rem; }
    .sidebar-section-title{ font-size: 1.05rem; }
    [data-testid="stSidebar"] label[data-baseweb="radio"]{ padding: 0.38rem 0.55rem; }
    [data-testid="stSidebar"] hr{ margin: 0.55rem 0; }
    .sidebar-card{ padding: 0.62rem 0.7rem; }
    .sidebar-notes li{ margin: 0.22rem 0; }
    .sidebar-spacer{ height: 0.6rem; }
}

h1, h2, h3{
    letter-spacing: -0.02em;
    color: var(--text);
}

h1{
    font-weight: 780;
    /* sedikit lebih kecil, tetap proporsional sebagai judul utama */
    font-size: clamp(1.4rem, 1.15rem + 1.2vw, 2.2rem);
    line-height: 1.06;
}

.stCaption, .stMarkdown p, .stMarkdown li{
    color: var(--muted);
}

.subtitle{
    margin-top: -0.35rem;
    margin-bottom: 2.1rem;
    font-size: 1.05rem;
    font-weight: 600;
    color: rgba(17,24,39,0.72);
}

.name-highlight{
    display: inline-block;
    padding: 0.08rem 0.45rem;
    margin: 0 0.1rem;
    border-radius: 999px;
    background: linear-gradient(135deg, rgba(91,140,255,0.18), rgba(138,108,255,0.14));
    border: 1px solid rgba(91,140,255,0.22);
    color: rgba(17,24,39,0.86);
    font-weight: 750;
}

div[data-testid="stMetric"]{
    background: var(--card);
    border: 1px solid var(--cardBorder);
    border-radius: var(--radius);
    padding: 14px 16px;
    box-shadow: 0 8px 30px rgba(17,24,39,0.06);
}

[data-testid="stForm"], [data-testid="stExpander"], [data-testid="stDataFrame"]{
    background: var(--card);
    border: 1px solid var(--cardBorder);
    border-radius: var(--radius);
    box-shadow: 0 8px 30px rgba(17,24,39,0.06);
}

[data-testid="stDataFrame"]{
    overflow: hidden;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div{
    border-radius: 12px;
    border: 1px solid var(--cardBorder);
    background: rgba(255,255,255,0.72);
}

.stButton > button, .stDownloadButton > button{
    border: 1px solid rgba(91,140,255,0.35);
    background: linear-gradient(135deg, rgba(91,140,255,0.95), rgba(138,108,255,0.95));
    color: white;
    border-radius: 999px;
    padding: 0.55rem 1rem;
    font-weight: 650;
    box-shadow: 0 10px 24px rgba(91,140,255,0.18);
    transition: transform .08s ease, box-shadow .12s ease, border-color .12s ease;
}

.stButton > button:hover, .stDownloadButton > button:hover{
    transform: translateY(-1px);
    box-shadow: 0 14px 30px rgba(91,140,255,0.22);
    border-color: rgba(91,140,255,0.55);
}

.stButton > button:active, .stDownloadButton > button:active{
    transform: translateY(0px);
}

/* Soft alert component (used via soft_alert()) */
.soft-alert{
    padding: 0.8rem 0.95rem;
    border-radius: 14px;
    border: 1px solid rgba(17,24,39,0.08);
    box-shadow: 0 10px 22px rgba(17,24,39,0.06);
    margin: 0.35rem 0 0.75rem;
    color: rgba(17,24,39,0.85);
}

.soft-alert.info{
    background: linear-gradient(135deg, rgba(91,140,255,0.14), rgba(138,108,255,0.12));
    border-color: rgba(91,140,255,0.22);
}

.soft-alert.success{
    background: linear-gradient(135deg, rgba(34,197,94,0.14), rgba(16,185,129,0.12));
    border-color: rgba(16,185,129,0.22);
}

.soft-alert.warning{
    background: linear-gradient(135deg, rgba(245,158,11,0.18), rgba(251,191,36,0.12));
    border-color: rgba(245,158,11,0.26);
}

.soft-alert.error{
    background: linear-gradient(135deg, rgba(244,63,94,0.15), rgba(255,140,181,0.12));
    border-color: rgba(244,63,94,0.24);
}

/* Page/stage title pill (used in all 5 steps) */
.page-title, .stage-title{
    font-size: 1.35rem;
    font-weight: 780;
    color: var(--text);
    margin: 0.45rem 0 0.75rem;
    padding: 0.72rem 1.05rem;
    background: rgba(255,255,255,0.70);
    border: 1px solid rgba(91,140,255,0.14);
    border-radius: 999px;
    display: inline-flex;
    align-items: center;
    gap: 0.65rem;
    max-width: 100%;
    box-shadow: 0 10px 30px rgba(17,24,39,0.06);
    backdrop-filter: blur(10px);
}

.stage-icon{
    width: 40px;
    height: 40px;
    border-radius: 999px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.95), rgba(255,255,255,0.55));
    border: 1px solid rgba(17,24,39,0.08);
    box-shadow: 0 10px 22px rgba(17,24,39,0.06);
    flex: 0 0 auto;
}

.stage-text{
    display: inline-block;
    line-height: 1.15;
    letter-spacing: -0.01em;
    word-break: break-word;
}

.schema-table-container{
  margin-top: 1rem;
  border-radius: var(--radius);
  overflow: hidden;
  box-shadow: 0 10px 30px rgba(17,24,39,0.08);
}

.schema-table-container table{
  width: 100%;
  border-collapse: collapse;
  background: rgba(255,255,255,0.72);
}

.schema-table-container thead{
  background: linear-gradient(90deg, rgba(91,140,255,0.12), rgba(138,108,255,0.1));
  border-bottom: 2px solid rgba(91,140,255,0.15);
}

.schema-table-container th{
  padding: 0.75rem 0.95rem;
  text-align: left;
  font-weight: 750;
  color: var(--text);
  letter-spacing: -0.01em;
  white-space: nowrap;
}

.schema-table-container td{
  padding: 0.65rem 0.95rem;
  color: var(--muted);
  border-bottom: 1px solid rgba(17,24,39,0.05);
}

.schema-table-container tbody tr:hover{
  background: rgba(91,140,255,0.06);
}

.schema-table-container tbody tr:last-child td{
  border-bottom: none;
}

.btn-deskripsi{
  display: inline-flex;
  align-items: center;
  gap: 0.6rem;
  padding: 0.65rem 1.1rem;
  background: linear-gradient(135deg, rgba(255,179,140,0.95), rgba(255,140,140,0.95));
  border: 1px solid rgba(255,140,140,0.3);
  color: white;
  border-radius: 999px;
  font-weight: 700;
  box-shadow: 0 10px 24px rgba(255,140,140,0.2);
  cursor: pointer;
  transition: transform .08s ease, box-shadow .12s ease;
  font-size: 0.95rem;
}

.btn-deskripsi:hover{
  transform: translateY(-2px);
  box-shadow: 0 14px 30px rgba(255,140,140,0.25);
}

.cleaning-cards{ margin-top: 0.4rem; }
.cleaning-card{
    padding: 0.8rem 0.9rem;
    border-radius: 14px;
    border: 1px solid rgba(17,24,39,0.08);
    box-shadow: 0 10px 22px rgba(17,24,39,0.06);
    min-height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.cleaning-card h4{
    margin: 0 0 0.35rem 0;
    font-size: 0.95rem;
    font-weight: 700;
    color: rgba(17,24,39,0.9);
}
.cleaning-card h4.title-small{
    font-size: 0.9rem;
}
.cleaning-card .value{
    font-size: 1.25rem;
    font-weight: 800;
    color: rgba(17,24,39,0.9);
}
.cleaning-card .note{
    margin-top: 0.25rem;
    font-size: 0.8rem;
    color: rgba(17,24,39,0.6);
}

.form-card{
    max-width: 560px;
    margin: 0.6rem auto 1.1rem;
    padding: 0.6rem 0.6rem;
    background: rgba(255,255,255,0.65);
    border: 1px solid var(--cardBorder);
    border-radius: var(--radius);
    box-shadow: 0 10px 28px rgba(17,24,39,0.06);
}

@media (max-width: 900px){
    .form-card{ max-width: 100%; }
}

/* Optional: rapikan footer */
footer{ opacity: 0.65; }
</style>
                """,
                unsafe_allow_html=True,
        )


def stage_title(title: str, icon: str = ""):
    icon_html = f'<span class="stage-icon">{icon}</span>' if icon else ""
    st.markdown(
        f'<div class="stage-title">{icon_html}<span class="stage-text">{title}</span></div>',
        unsafe_allow_html=True,
    )


def soft_alert(text: str, kind: str = "info"):
    kind = (kind or "info").lower().strip()
    if kind not in {"info", "success", "warning", "error"}:
        kind = "info"
    st.markdown(
        f"""
<div class="soft-alert {kind}">
  {text}
</div>
        """,
        unsafe_allow_html=True,
    )


# =========================
# App
# =========================
def init_state():
    st.session_state.setdefault("raw_df", None)
    st.session_state.setdefault("clean_df", None)
    st.session_state.setdefault("clean_log", None)
    st.session_state.setdefault("clean_audit_df", None)
    st.session_state.setdefault("eval_results", None)
    st.session_state.setdefault("best_result", None)
    st.session_state.setdefault("target_col", DEFAULT_TARGET)


def get_schema_data() -> pd.DataFrame:
    """Return schema/metadata table for all columns."""
    return pd.DataFrame([
        {
            "Kolom": "IPK",
            "Tipe": "Numerik",
            "Deskripsi": "Indeks Prestasi Kumulatif",
            "Rentang/Domain": "0.00–4.00",
            "Satuan/Format": "skala 0–4",
        },
        {
            "Kolom": "Pendapatan_Orang_Tua",
            "Tipe": "Numerik",
            "Deskripsi": "Pendapatan orang tua per bulan",
            "Rentang/Domain": "~1–35",
            "Satuan/Format": "juta rupiah/bulan",
        },
        {
            "Kolom": "Asal_Sekolah",
            "Tipe": "Kategori",
            "Deskripsi": "Gabungan status sekolah & lokasi sekolah",
            "Rentang/Domain": "Negeri-Kota / Negeri-Desa / Swasta-Kota / Swasta-Desa",
            "Satuan/Format": "teks",
        },
        {
            "Kolom": "Lokasi_Domisili",
            "Tipe": "Kategori",
            "Deskripsi": "Kabupaten/Kota domisili (simulasi)",
            "Rentang/Domain": "50 kategori Kab/Kota",
            "Satuan/Format": "teks",
        },
        {
            "Kolom": "Keikutsertaan_Organisasi",
            "Tipe": "Numerik (int)",
            "Deskripsi": "Jumlah organisasi aktif",
            "Rentang/Domain": "0–8",
            "Satuan/Format": "jumlah organisasi",
        },
        {
            "Kolom": "Pengalaman_Sosial",
            "Tipe": "Numerik (int)",
            "Deskripsi": "Total jam kegiatan sosial/relawan",
            "Rentang/Domain": "0–400",
            "Satuan/Format": "jam",
        },
        {
            "Kolom": "Gender",
            "Tipe": "Kategori",
            "Deskripsi": "Jenis kelamin",
            "Rentang/Domain": "L / P",
            "Satuan/Format": "teks",
        },
        {
            "Kolom": "Status_Disabilitas",
            "Tipe": "Kategori",
            "Deskripsi": "Status disabilitas",
            "Rentang/Domain": "Ya / Tidak",
            "Satuan/Format": "teks",
        },
        {
            "Kolom": "Prestasi_Akademik",
            "Tipe": "Numerik (int)",
            "Deskripsi": "Jumlah prestasi/penghargaan akademik",
            "Rentang/Domain": "0–8",
            "Satuan/Format": "jumlah prestasi",
        },
        {
            "Kolom": "Prestasi_Non_Akademik",
            "Tipe": "Numerik (int)",
            "Deskripsi": "Jumlah prestasi/penghargaan non-akademik",
            "Rentang/Domain": "0–8",
            "Satuan/Format": "jumlah prestasi",
        },
        {
            "Kolom": "Diterima_Beasiswa",
            "Tipe": "Label (int)",
            "Deskripsi": "Hasil seleksi",
            "Rentang/Domain": "1 = diterima, 0 = tidak",
            "Satuan/Format": "biner",
        },
    ])


def render_schema_table():
    """Render the schema table with styling."""
    schema_df = get_schema_data()
    st.markdown(
        '<div class="schema-table-container">',
        unsafe_allow_html=True,
    )
    st.dataframe(schema_df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


def page_upload_and_clean():
    stage_title("Upload & Cleaning", icon="\U0001F4E5")
    st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload dataset (.csv / .xlsx)",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )
    use_sample = st.toggle("Gunakan data contoh (beasiswa.csv).", value=False)

    df = None
    source = None

    if uploaded is not None:
        try:
            df = _read_uploaded(uploaded)
            source = f"Upload: {uploaded.name}"
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            return
    else:
        if use_sample:
            df = _load_sample()
            source = "Contoh: beasiswa.csv"
        else:
            st.info("Silakan upload dataset atau aktifkan opsi data contoh.")
            return

    if df is None:
        st.error("Data contoh tidak ditemukan. Pastikan file beasiswa.csv ada di repo (selevel app.py).")
        return

    st.success(f"Data aktif: {source}")
    st.subheader("Skema Kolom")
    render_schema_table()

    st.subheader("Preview Data (Raw)")
    st.dataframe(df.head(50), use_container_width=True)
    st.write(f"Rows: {len(df):,} | Columns: {df.shape[1]}")

    st.subheader("Konfigurasi Cleaning")
    c1, c2, c3 = st.columns([1.2, 1.2, 1])
    with c1:
        target_col = st.text_input("Nama Kolom Target", value=st.session_state["target_col"])
    with c2:
        st.markdown('<div style="height:1.9rem"></div>', unsafe_allow_html=True)
        drop_missing_target = st.checkbox("Buang baris yang target-nya kosong", value=True)
    with c3:
        numeric_impute = st.selectbox("Imputasi Numerik", ["median", "mean"], index=0)

    # Jalankan cleaning
    if st.button("Jalankan cleaning", type="primary"):
        clean_df, log, audit_df = clean_dataset(
            df,
            numeric_cols=NUMERIC_COLS,
            categorical_cols=CATEGORICAL_COLS,
            target_col=target_col if target_col in df.columns else None,
            drop_rows_missing_target=drop_missing_target,
            numeric_impute=numeric_impute,
        )
        st.session_state["raw_df"] = df
        st.session_state["clean_df"] = clean_df
        st.session_state["clean_log"] = log
        st.session_state["clean_audit_df"] = audit_df
        st.session_state["target_col"] = target_col

    if st.session_state["clean_df"] is None:
        st.info("Klik **Jalankan cleaning** untuk menyimpan dataset hasil cleaning.")
        return

    clean_df: pd.DataFrame = st.session_state["clean_df"]
    log: CleaningLog = st.session_state["clean_log"]

    st.subheader("Hasil cleaning")
    imputed_num = ", ".join(log.imputed_numeric.keys()) or "Tidak ada"
    imputed_cat = ", ".join(log.imputed_categorical.keys()) or "Tidak ada"
    cards = st.columns([0.9, 0.9, 0.9, 1.1, 1.1])
    card_data = [
        ("Rows Sebelum", f"{log.rows_before:,}", "#EAF3FF"),
        ("Rows Sesudah", f"{log.rows_after:,}", "#ECFDF3"),
        ("Duplikat Dihapus", f"{log.dropped_duplicates:,}", "#FFF6E5"),
        ("Imputasi Numerik", f"{len(log.imputed_numeric)} kolom", "#F3E8FF"),
        ("Imputasi Kategorikal", f"{len(log.imputed_categorical)} kolom", "#FEE2E2"),
    ]
    for col, (title, value, bg) in zip(cards, card_data):
        title_class = "title-small" if title == "Duplikat Dihapus" else ""
        with col:
            st.markdown(
                f"""
<div class="cleaning-card" style="background:{bg};">
    <h4 class="{title_class}">{title}</h4>
  <div class="value">{value}</div>
</div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown('<div style="height:0.55rem"></div>', unsafe_allow_html=True)
    st.markdown(
        f"**Keterangan Imputasi Numerik:** {imputed_num}<br/>**Keterangan Imputasi Kategorikal:** {imputed_cat}",
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:0.65rem"></div>', unsafe_allow_html=True)
    show_audit = st.checkbox("Tampilkan data dihapus dan berubah.", value=False)
    if show_audit:
        audit_df: Optional[pd.DataFrame] = st.session_state.get("clean_audit_df")
        if audit_df is None or audit_df.empty:
            st.info("Tidak ada baris yang dihapus atau berubah karena imputasi.")
        else:
            st.dataframe(
                audit_df.style.apply(_style_clean_audit_row, axis=1),
                use_container_width=True,
            )

    st.markdown('<div style="height:0.9rem"></div>', unsafe_allow_html=True)
    st.dataframe(clean_df.head(50), use_container_width=True)

    st.download_button(
        "Download dataset hasil cleaning (CSV)",
        data=_to_csv_bytes(clean_df),
        file_name="beasiswa_cleaned.csv",
        mime="text/csv",
    )


def page_eda():
    stage_title("EDA (Exploratory Data Analysis)", icon="\U0001F4CA")

    df: Optional[pd.DataFrame] = st.session_state.get("clean_df")
    if df is None:
        st.warning("Belum ada data hasil cleaning. Silakan ke tahap **Upload & Cleaning** dulu.")
        return

    target_col = st.session_state.get("target_col", DEFAULT_TARGET)

    st.subheader("Ringkasan Data")
    miss = int(df.isna().sum().sum())
    c1, c2, c3 = st.columns(3)
    kpis = [
        ("Rows", f"{len(df):,}", "#EAF3FF"),
        ("Columns", f"{df.shape[1]}", "#ECFDF3"),
        ("Total Missing", f"{miss:,}", "#FFF6E5"),
    ]
    for col, (title, value, bg) in zip([c1, c2, c3], kpis):
        with col:
            st.markdown(
                f"""
<div class="cleaning-card" style="background:{bg}; min-height: 88px; padding: 0.65rem 0.75rem;">
    <h4 style="margin:0 0 8px 0; font-size:16px;">{title}</h4>
    <div class="value" style="font-size:26px;">{value}</div>
</div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)

    st.subheader("Tipe Data")
    st.dataframe(df.dtypes.astype(str).reset_index().rename(columns={"index": "column", 0: "dtype"}), use_container_width=True)

    st.subheader("Statistik Deskriptif (Numerik)")
    present_numeric = [c for c in NUMERIC_COLS if c in df.columns]
    if present_numeric:
        st.dataframe(df[present_numeric].describe().T, use_container_width=True)
    else:
        st.info("Tidak ditemukan kolom numerik yang diharapkan.")

    st.subheader("Distribusi Target")
    if target_col in df.columns:
        vc = df[target_col].value_counts(dropna=False)
        vc_df = vc.reset_index()
        vc_df.columns = ["Label", "Jumlah"]
        vc_df["Label"] = vc_df["Label"].astype(str)
        plot_bar_colored(vc_df, x_col="Label", y_col="Jumlah", title=None, height=240)
    else:
        st.info(f"Kolom target **{target_col}** tidak ada di dataset ini (EDA tetap bisa, tapi training butuh target).")

    st.subheader("Plot Data")
    c1, c2 = st.columns(2)
    with c1:
        if present_numeric:
            col = st.selectbox("Histogram Kolom Numerik", present_numeric, index=0)
            hist_placeholder = st.empty()
            bins = st.slider("Bins", 5, 60, 25, 1)
            plot_hist(df, col, bins, container=hist_placeholder)
    with c2:
        present_cat = [c for c in CATEGORICAL_COLS if c in df.columns]
        if present_cat:
            cat = st.selectbox("Frekuensi Kolom Kategorikal", present_cat, index=0)
            vc = df[cat].astype("string").value_counts(dropna=False).head(30)
            vc_df = vc.reset_index()
            vc_df.columns = ["Kategori", "Jumlah"]
            vc_df["Kategori"] = vc_df["Kategori"].fillna("(Kosong)").astype(str)
            plot_bar_colored(vc_df, x_col="Kategori", y_col="Jumlah", title=None, height=260)

    st.subheader("Korelasi Antar Fitur Numerik (Pearson)")
    if len(present_numeric) >= 2:
        plot_correlation(df, present_numeric)
        st.markdown(
            """
**Keterangan:**

- Setiap kotak merepresentasikan hubungan antara 2 kolom numerik.
- Nilai korelasi berada pada rentang $-1$ sampai $+1$:
    - Mendekati $+1$ → hubungan positif kuat (dua fitur cenderung naik bersama).
    - Mendekati $-1$ → hubungan negatif kuat (satu naik saat yang lain turun).
    - Mendekati $0$ → hubungan linear lemah/tidak ada.
- Gradasi hijau: warna makin hijau berarti korelasi makin *positif* dan makin kuat.
            """
        )
    else:
        st.info("Butuh minimal 2 kolom numerik untuk korelasi.")


def page_modeling():
    stage_title("Menjalankan Model", icon="\U0001F9E0")

    df: Optional[pd.DataFrame] = st.session_state.get("clean_df")
    if df is None:
        soft_alert("Belum ada data hasil cleaning. Silakan ke tahap **Upload & Cleaning** dulu.", kind="warning")
        return

    target_col = st.session_state.get("target_col", DEFAULT_TARGET)
    if target_col not in df.columns:
        soft_alert(f"Kolom target **{target_col}** tidak ditemukan. Pastikan dataset memiliki target untuk training.", kind="error")
        return

    missing_cols = [c for c in (NUMERIC_COLS + CATEGORICAL_COLS) if c not in df.columns]
    if missing_cols:
        soft_alert("Dataset Anda tidak memiliki semua kolom fitur yang dibutuhkan:", kind="error")
        st.write(missing_cols)
        return

    st.subheader("Pengaturan Training")
    c1, c2, c3 = st.columns(3)
    with c1:
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
        st.caption("Note : Menentukan porsi data untuk pengujian (test set) saat split data.")
    with c2:
        random_state = st.number_input("Random State", min_value=0, max_value=9999, value=42, step=1)
        st.caption("Note : Menentukan seed acak saat pembagian train-test (dan proses acak lain yang mendukung).")
    with c3:
        st.markdown('<div style="height:0.85rem"></div>', unsafe_allow_html=True)
        balanced = st.checkbox(
            "Gunakan class_weight = balanced\n(untuk data imbalanced)",
            value=True,
        )
        st.caption("Note : Mengaktifkan pembobotan kelas otomatis untuk menangani data imbalanced")

    st.markdown("**Pilih model yang akan dijalankan :**")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        use_lr = st.checkbox("Logistic Regression", value=True)
    with mc2:
        use_rf = st.checkbox("Random Forest", value=True)
    with mc3:
        use_xgb = st.checkbox(
            "XGBoost",
            value=bool(HAS_XGBOOST),
            disabled=not HAS_XGBOOST,
        )

    selected_models: List[str] = []
    if use_lr:
        selected_models.append("Logistic Regression")
    if use_rf:
        selected_models.append("Random Forest")
    if use_xgb and HAS_XGBOOST:
        selected_models.append("XGBoost")

    if not HAS_XGBOOST:
        soft_alert("Catatan: XGBoost tidak tersedia di environment ini. (Aplikasi tetap berjalan.)", kind="info")

    if not selected_models:
        soft_alert("Pilih minimal 1 model untuk dijalankan.", kind="warning")
        return

    if st.button("Train & Evaluate", type="primary"):
        with st.spinner("Training & evaluasi model..."):
            results = train_and_evaluate(
                df=df,
                target_col=target_col,
                test_size=float(test_size),
                random_state=int(random_state),
                selected_models=selected_models,
                class_weight_balanced=bool(balanced),
            )
        st.session_state["eval_results"] = results
        st.session_state["best_result"] = None

    results: Optional[List[EvalResult]] = st.session_state.get("eval_results")
    if not results:
        soft_alert("Klik **Train & Evaluate** untuk menjalankan model.", kind="info")
        return

    soft_alert(f"Selesai. Model yang dievaluasi: {len(results)}", kind="success")

    # Urutkan tampilan berdasarkan performa terbaik (F1-score)
    results_sorted = sorted(results, key=lambda r: r.f1, reverse=True)

    st.subheader("Ringkasan Hasil")
    soft_cmap = _soft_teal_cmap()
    metrics_df = pd.DataFrame(
        [
            {
                "Model": r.name,
                "Accuracy": r.accuracy,
                "Precision": r.precision,
                "Recall": r.recall,
                "F1-score": r.f1,
            }
            for r in results_sorted
        ]
    ).sort_values("F1-score", ascending=False)

    st.dataframe(
        metrics_df.style.format({"Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}"}).background_gradient(
            cmap=soft_cmap, subset=["Accuracy", "Precision", "Recall", "F1-score"]
        ),
        use_container_width=True,
    )

    st.subheader("Detail Tiap Model")
    for r in results_sorted:
        with st.expander(f"{r.name}"):
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown("**Ringkasan Metrik**")
                r1c1, r1c2 = st.columns(2)
                for col, title, value, bg in [
                    (r1c1, "Accuracy", f"{r.accuracy:.3f}", "#EAF3FF"),
                    (r1c2, "Precision", f"{r.precision:.3f}", "#ECFDF3"),
                ]:
                    with col:
                        st.markdown(
                            f"""
<div class="cleaning-card" style="background:{bg}; min-height: 86px; padding: 0.6rem 0.75rem; margin-bottom: 0.65rem;">
  <h4 style="margin:0 0 6px 0; font-size:14px;">{title}</h4>
  <div class="value" style="font-size:24px;">{value}</div>
</div>
                            """,
                            unsafe_allow_html=True,
                        )

                r2c1, r2c2 = st.columns(2)
                for col, title, value, bg in [
                    (r2c1, "Recall", f"{r.recall:.3f}", "#FFF6E5"),
                    (r2c2, "F1-score", f"{r.f1:.3f}", "#F3E8FF"),
                ]:
                    with col:
                        st.markdown(
                            f"""
<div class="cleaning-card" style="background:{bg}; min-height: 86px; padding: 0.6rem 0.75rem; margin-bottom: 0.65rem;">
  <h4 style="margin:0 0 6px 0; font-size:14px;">{title}</h4>
  <div class="value" style="font-size:24px;">{value}</div>
</div>
                            """,
                            unsafe_allow_html=True,
                        )

                st.markdown(
                    f"""
- **Accuracy** {r.accuracy:.3f}: proporsi prediksi benar dari seluruh data uji.
- **Precision** {r.precision:.3f}: dari semua yang diprediksi *diterima (1)*, berapa yang benar-benar diterima.
- **Recall** {r.recall:.3f}: dari semua yang benar-benar *diterima (1)*, berapa yang berhasil terdeteksi.
- **F1-score** {r.f1:.3f}: ringkasan keseimbangan Precision dan Recall (berguna saat data tidak seimbang).
                    """
                )
            with c2:
                plot_confusion_matrix(r.cm, title=f"Confusion Matrix: {r.name}", cmap=soft_cmap)

            st.markdown("**Classification Report**")
            try:
                rep = pd.DataFrame(r.report_dict).T
                rep = rep.rename(
                    columns={
                        "precision": "Precision",
                        "recall": "Recall",
                        "f1-score": "F1-score",
                        "support": "Support",
                    }
                )
                # rapikan urutan baris jika ada
                preferred = ["0", "1", "accuracy", "macro avg", "weighted avg"]
                order = [i for i in preferred if i in rep.index] + [i for i in rep.index if i not in preferred]
                rep = rep.loc[order]
                st.dataframe(
                    rep.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}", "Support": "{:.0f}"}).background_gradient(
                        cmap=soft_cmap, subset=[c for c in ["Precision", "Recall", "F1-score"] if c in rep.columns]
                    ),
                    use_container_width=True,
                )
            except Exception:
                # fallback bila struktur report berubah
                st.code(r.report, language="text")

            st.markdown(
                """
**Catatan Confusion Matrix**

- Baris menunjukkan label asli (Actual), kolom menunjukkan prediksi (Predicted).
- Nilai diagonal (kiri-atas & kanan-bawah) adalah prediksi benar; di luar diagonal adalah kesalahan prediksi.
                """
            )


def page_compare_and_best():
    stage_title("Membandingkan Metrik & Memilih Model Terbaik", icon="\u2696\ufe0f")

    results: Optional[List[EvalResult]] = st.session_state.get("eval_results")
    if not results:
        st.warning("Belum ada hasil training. Jalankan tahap **Menjalankan Model** dulu.")
        return

    metrics_df = pd.DataFrame(
        [
            {
                "Model": r.name,
                "Accuracy": r.accuracy,
                "Precision": r.precision,
                "Recall": r.recall,
                "F1-score": r.f1,
            }
            for r in results
        ]
    )

    st.subheader("Tabel perbandingan metrik")
    st.caption("(Accuracy, Precision, Recall, F1-score)")
    sorted_df = metrics_df.sort_values("F1-score", ascending=False)

    def _highlight_best_f1(col: pd.Series):
        try:
            max_val = float(pd.to_numeric(col, errors="coerce").max())
        except Exception:
            max_val = None
        styles: List[str] = []
        for v in col:
            try:
                is_best = (max_val is not None) and (float(v) == max_val)
            except Exception:
                is_best = False
            styles.append(
                "background-color: rgba(91,140,255,0.22); color: rgba(17,24,39,0.92); font-weight: 800;"
                if is_best
                else ""
            )
        return styles

    st.dataframe(
        sorted_df.style.format({"Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "F1-score": "{:.4f}"})
        .apply(_highlight_best_f1, subset=["F1-score"]),
        use_container_width=True,
    )

    top_model = sorted_df.iloc[0]["Model"]
    top_f1 = float(sorted_df.iloc[0]["F1-score"])
    st.markdown(
        """
**Deskripsi metrik:**
- **Accuracy**: proporsi prediksi yang benar dari seluruh data.
- **Precision**: ketepatan prediksi kelas positif (mengurangi false positive).
- **Recall**: kemampuan menangkap data kelas positif (mengurangi false negative).
- **F1-score**: keseimbangan antara precision dan recall.
"""
    )
    st.info(
        f"Berdasarkan tabel, model dengan **F1-score** tertinggi adalah **{top_model}** (F1 = {top_f1:.4f}). "
        "Ini menunjukkan keseimbangan terbaik antara precision dan recall, sehingga secara umum paling stabil untuk data yang berpotensi tidak seimbang."
    )

    st.subheader("Visual Perbandingan")
    try:
        import altair as alt

        melted = metrics_df.melt(
            id_vars=["Model"],
            value_vars=["Accuracy", "Precision", "Recall", "F1-score"],
            var_name="Metrik",
            value_name="Nilai",
        )

        color_scale = alt.Scale(
            domain=["Accuracy", "Precision", "Recall", "F1-score"],
            range=["#93C5FD", "#A7F3D0", "#FDE68A", "#E9D5FF"],
        )

        chart = (
            alt.Chart(melted)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, opacity=0.92)
            .encode(
                x=alt.X("Model:N", title=None, axis=alt.Axis(labelAngle=0, labelColor="#6B7280")),
                xOffset=alt.XOffset("Metrik:N"),
                y=alt.Y("Nilai:Q", title=None, scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Metrik:N", scale=color_scale, legend=alt.Legend(title=None, orient="top")),
                tooltip=[
                    alt.Tooltip("Model:N"),
                    alt.Tooltip("Metrik:N"),
                    alt.Tooltip("Nilai:Q", format=".4f"),
                ],
            )
            .properties(height=320)
            .configure_view(strokeOpacity=0)
            .configure_axis(
                gridColor="#E9ECF5",
                domainColor="#E5E7EB",
                tickColor="#E5E7EB",
                labelColor="#6B7280",
                titleColor="#6B7280",
            )
            .configure_legend(labelColor="#6B7280")
        )
        st.altair_chart(chart, use_container_width=True)
    except ModuleNotFoundError:
        # fallback sederhana jika altair tidak tersedia
        st.bar_chart(metrics_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]])

    st.subheader("Memilih Model Terbaik (Kuantitatif)")
    st.markdown(
        '<div style="font-size:1.02rem; color: rgba(107,114,128,0.95); margin-top:-0.35rem; margin-bottom:0.65rem;">'
        'Secara default, banyak studi memilih <b>F1-score</b> sebagai metrik utama ketika data target tidak seimbang '
        '(karena mempertimbangkan precision &amp; recall sekaligus). Namun Anda bebas memilih metrik di bawah.'
        "</div>",
        unsafe_allow_html=True,
    )

    best_metric = st.selectbox("Metrik penentu model terbaik", ["Accuracy", "Precision", "Recall", "F1-score"], index=3)
    best = pick_best_model(results, best_metric)
    st.session_state["best_result"] = best

    if best is None:
        st.error("Tidak bisa menentukan model terbaik (hasil kosong).")
        return

    st.success(f"Model terbaik berdasarkan **{best_metric}** adalah: **{best.name}**")
    best_metrics_table = pd.DataFrame(
        {
            "Metrik": ["Accuracy", "Precision", "Recall", "F1-score"],
            "Nilai": [best.accuracy, best.precision, best.recall, best.f1],
        }
    )
    st.dataframe(
        best_metrics_table.style.format({"Nilai": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )
    best_value = {
        "Accuracy": best.accuracy,
        "Precision": best.precision,
        "Recall": best.recall,
        "F1-score": best.f1,
    }[best_metric]
    st.info(
        f"Model **{best.name}** dipilih karena memiliki nilai **{best_metric}** tertinggi dibanding model lain "
        f"(nilai = {best_value:.4f}). Dengan kriteria ini, model tersebut paling optimal untuk tujuan evaluasi yang dipilih."
    )
    soft_cmap = _soft_teal_cmap()
    plot_confusion_matrix(
        best.cm,
        title=f"Confusion Matrix (Best): {best.name}",
        cmap=soft_cmap,
        # 25% lebih besar dari sebelumnya
        figsize=(2.45, 1.95),
        # 50% lebih kecil dari ukuran font saat ini
        font_scale=0.425,
    )

    st.markdown("**Classification Report**")
    try:
        rep = pd.DataFrame(best.report_dict).T
        rep = rep.rename(
            columns={
                "precision": "Precision",
                "recall": "Recall",
                "f1-score": "F1-score",
                "support": "Support",
            }
        )
        preferred = ["0", "1", "accuracy", "macro avg", "weighted avg"]
        order = [i for i in preferred if i in rep.index] + [i for i in rep.index if i not in preferred]
        rep = rep.loc[order]
        st.dataframe(
            rep.style.format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1-score": "{:.4f}", "Support": "{:.0f}"}).background_gradient(
                cmap=soft_cmap, subset=[c for c in ["Precision", "Recall", "F1-score"] if c in rep.columns]
            ),
            use_container_width=True,
        )

        st.markdown(
            """
**Deskripsi output Classification Report**

- **Baris kelas (0/1)** menunjukkan performa model untuk masing-masing label.
- **Precision**: dari semua prediksi ke kelas tersebut, berapa yang benar.
- **Recall**: dari semua data yang benar-benar kelas tersebut, berapa yang berhasil terdeteksi.
- **F1-score**: ringkasan keseimbangan precision & recall (semakin tinggi semakin baik).
- **Support**: jumlah data uji pada kelas tersebut.
- **accuracy**: akurasi total pada seluruh data uji.
- **macro avg**: rata-rata sederhana antar kelas (setiap kelas bobotnya sama).
- **weighted avg**: rata-rata berbobot sesuai jumlah data (support), sehingga kelas mayoritas lebih berpengaruh.
            """
        )
    except Exception:
        st.code(best.report, language="text")


def page_prediction():
    stage_title("Prediksi", icon="\U0001F52E")

    st.caption("Gunakan model terbaik untuk prediksi 1 data atau batch.")

    best: Optional[EvalResult] = st.session_state.get("best_result")
    df: Optional[pd.DataFrame] = st.session_state.get("clean_df")
    if df is None:
        st.warning("Belum ada data. Mulai dari tahap **Upload & Cleaning**.")
        return
    if best is None:
        st.warning("Belum ada model terbaik. Tentukan dulu di tahap **Membandingkan Metrik & Memilih Model Terbaik**.")
        return

    has_proba_model = hasattr(best.fitted, "predict_proba")
    if has_proba_model:
        threshold = st.slider(
            "Ambang Probabilitas Diterima (label 1)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
        )
        st.caption(
            "Jika probabilitas diterima ≥ ambang, maka prediksi label = 1 (diterima). "
            "Jika < ambang, maka label = 0 (tidak)."
        )
    else:
        threshold = 0.5
        st.info("Model ini tidak menyediakan probabilitas (`predict_proba`), sehingga prediksi menggunakan label langsung dari model.")

    st.subheader("Prediksi 1 Data (Form)")
    with st.form("single_pred"):
        label_map = {
            "IPK": ("IPK", "Skala 0 - 4"),
            "Pendapatan_Orang_Tua": ("Pendapatan Orang Tua", "Juta/Bulan"),
            "Keikutsertaan_Organisasi": ("Keikutsertaan Organisasi", "Jumlah, Skala 0-10"),
            "Pengalaman_Sosial": ("Pengalaman Sosial", "Jam, Skala 0-400"),
            "Prestasi_Akademik": ("Prestasi Akademik", "Jumlah, Skala 0-10"),
            "Prestasi_Non_Akademik": ("Prestasi Non Akademik", "Jumlah, Skala 0-10"),
            "Asal_Sekolah": "Asal Sekolah",
            "Lokasi_Domisili": "Lokasi Domisili",
            "Status_Disabilitas": "Status Disabilitas",
        }
        vals_num = {}
        for c in NUMERIC_COLS:
            label = label_map.get(c, c)
            if isinstance(label, tuple):
                main, detail = label
                st.markdown(f"**{main}** ({detail})")
                display_label = ""
            else:
                st.markdown(f"**{label}**")
                display_label = ""
            vals_num[c] = st.number_input(
                display_label,
                value=float(df[c].median()) if c in df.columns else 0.0,
                label_visibility="collapsed",
                key=f"num_{c}",
            )
        vals_cat = {}
        for c in CATEGORICAL_COLS:
            opts = sorted(df[c].astype(str).unique().tolist()) if c in df.columns else []
            default = opts[0] if opts else ""
            label = label_map.get(c, c)
            if isinstance(label, tuple):
                main, detail = label
                st.markdown(f"**{main}** ({detail})")
                display_label = ""
            else:
                st.markdown(f"**{label}**")
                display_label = ""
            if opts:
                vals_cat[c] = st.selectbox(
                    display_label,
                    options=opts,
                    index=0,
                    label_visibility="collapsed",
                    key=f"cat_{c}",
                )
            else:
                vals_cat[c] = st.text_input(
                    display_label,
                    value=default,
                    label_visibility="collapsed",
                    key=f"cat_{c}",
                )
        submitted = st.form_submit_button("Prediksi")

    if submitted:
        one = {**vals_num, **vals_cat}
        X_one = pd.DataFrame([one])
        proba = None
        if has_proba_model:
            try:
                proba = float(best.fitted.predict_proba(X_one)[0, 1])
            except Exception:
                proba = None

        # Pastikan label konsisten dengan probabilitas jika tersedia
        if proba is not None:
            pred = int(proba >= float(threshold))
        else:
            pred = int(best.fitted.predict(X_one)[0])

        if proba is not None:
            st.success(
                f"Prediksi label: **{pred}** (1 = diterima, 0 = tidak) dengan probabilitas **{proba:.6f}**"
            )
            proba_tolak = 1.0 - float(proba)
            if pred == 1:
                st.caption(
                    f"Probabilitas di atas adalah peluang *diterima (label 1)* = {proba:.6f} (≥ {float(threshold):.2f}) → prediksi **diterima**. "
                    f"Probabilitas *ditolak (label 0)* ≈ {proba_tolak:.6f}."
                )
            else:
                st.caption(
                    f"Probabilitas di atas adalah peluang *diterima (label 1)* = {proba:.6f} (< {float(threshold):.2f}) → prediksi **tidak diterima**. "
                    f"Probabilitas *ditolak (label 0)* ≈ {proba_tolak:.6f}."
                )
        else:
            st.success(f"Prediksi label: **{pred}** (1 = diterima, 0 = tidak)")

    st.subheader("Prediksi Batch (Upload CSV)")
    st.caption("Upload CSV berisi kolom fitur yang sama (tanpa target) untuk membuat hasil prediksi massal.")
    file_pred = st.file_uploader("Upload data pendaftar (CSV)", type=["csv"], key="pred_uploader")
    if file_pred is not None:
        try:
            dnew = pd.read_csv(file_pred)
        except Exception as e:
            st.error(f"Gagal membaca CSV: {e}")
            return

        dnew.columns = [str(c).strip() for c in dnew.columns]
        missing = [c for c in (NUMERIC_COLS + CATEGORICAL_COLS) if c not in dnew.columns]
        if missing:
            st.error("Kolom berikut wajib ada di file prediksi:")
            st.write(missing)
            return

        # cleaning ringan pada data prediksi
        dnew2, _, _ = clean_dataset(
            dnew,
            numeric_cols=NUMERIC_COLS,
            categorical_cols=CATEGORICAL_COLS,
            target_col=None,
            drop_rows_missing_target=False,
        )

        out = dnew2.copy()
        X_batch = dnew2[NUMERIC_COLS + CATEGORICAL_COLS]
        yhat = None

        if has_proba_model:
            try:
                out["Prob_Diterima"] = best.fitted.predict_proba(X_batch)[:, 1]
                out["Prob_Ditolak"] = 1.0 - out["Prob_Diterima"]
                out["Prediksi_Label"] = (out["Prob_Diterima"] >= float(threshold)).astype(int)
            except Exception:
                pass
        if "Prediksi_Label" not in out.columns:
            yhat = best.fitted.predict(X_batch)
            out["Prediksi_Label"] = yhat

        # Resume (cards) di atas tabel
        total_n = int(len(out))
        diterima_n = int((out["Prediksi_Label"] == 1).sum())
        ditolak_n = int((out["Prediksi_Label"] == 0).sum())
        k1, k2, k3 = st.columns(3)
        for col, (title, value, bg) in zip(
            [k1, k2, k3],
            [
                ("Total Data Pendaftar", f"{total_n:,}", "#EAF3FF"),
                ("Beasiswa Diterima", f"{diterima_n:,}", "#ECFDF3"),
                ("Beasiswa Ditolak", f"{ditolak_n:,}", "#FEE2E2"),
            ],
        ):
            with col:
                st.markdown(
                    f"""
<div class="cleaning-card" style="background:{bg}; min-height: 88px; padding: 0.65rem 0.75rem;">
  <h4 style="margin:0 0 8px 0; font-size:16px;">{title}</h4>
  <div class="value" style="font-size:26px;">{value}</div>
</div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)

        if ("Prob_Diterima" in out.columns) and has_proba_model:
            st.caption(f"Ambang prediksi diterima (label 1) saat ini: {float(threshold):.2f}")

        feature_cols = [c for c in (NUMERIC_COLS + CATEGORICAL_COLS) if c in out.columns]
        has_proba = ("Prob_Diterima" in out.columns) and ("Prob_Ditolak" in out.columns)

        acc = out[out["Prediksi_Label"] == 1].copy()
        rej = out[out["Prediksi_Label"] == 0].copy()

        acc["Kategori_Prediksi"] = "Diterima"
        rej["Kategori_Prediksi"] = "Ditolak"

        if has_proba:
            acc = acc.sort_values("Prob_Diterima", ascending=False)
            rej = rej.sort_values("Prob_Ditolak", ascending=False)

        st.markdown("**Tabel Probabilitas Diterima (Prediksi = 1)**" if has_proba else "**Tabel Prediksi Diterima (Prediksi = 1)**")
        acc_cols = ["Kategori_Prediksi", "Prediksi_Label"]
        if has_proba:
            acc_cols.append("Prob_Diterima")
        acc_cols += feature_cols
        acc_view_df = acc[[c for c in acc_cols if c in acc.columns]].head(50)
        if has_proba:
            st.dataframe(acc_view_df.style.format({"Prob_Diterima": "{:.6f}"}), use_container_width=True)
        else:
            st.dataframe(acc_view_df, use_container_width=True)

        st.markdown("**Tabel Probabilitas Ditolak (Prediksi = 0)**" if has_proba else "**Tabel Prediksi Ditolak (Prediksi = 0)**")
        rej_cols = ["Kategori_Prediksi", "Prediksi_Label"]
        if has_proba:
            rej_cols.append("Prob_Ditolak")
        rej_cols += feature_cols
        rej_view_df = rej[[c for c in rej_cols if c in rej.columns]].head(50)
        if has_proba:
            st.dataframe(rej_view_df.style.format({"Prob_Ditolak": "{:.6f}"}), use_container_width=True)
        else:
            st.dataframe(rej_view_df, use_container_width=True)

        # Gabungan untuk download (diterima dulu, lalu ditolak)
        out_download = pd.concat([acc, rej], axis=0, ignore_index=True)

        st.download_button(
            "Download hasil prediksi (CSV)",
            data=_to_csv_bytes(out_download),
            file_name="hasil_prediksi_beasiswa.csv",
            mime="text/csv",
        )


def main():
    st.set_page_config(page_title="Seleksi Beasiswa - Machine Learning", layout="wide")
    apply_soft_theme()
    init_state()

    st.title("Sistem Seleksi Beasiswa dengan Machine Learning")
    st.markdown(
        '<div class="subtitle">Tugas Pembelajaran Mesin oleh <span class="name-highlight">Indri Puspita Sari</span> dan <span class="name-highlight">Landi Ruslandi</span></div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown(
            """
<div class="sidebar-brand">
    <div class="ml-logo" aria-hidden="true">
        <svg width="34" height="34" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="g" x1="10" y1="10" x2="54" y2="54" gradientUnits="userSpaceOnUse">
                    <stop stop-color="#5b8cff" stop-opacity="0.95"/>
                    <stop offset="1" stop-color="#8a6cff" stop-opacity="0.95"/>
                </linearGradient>
            </defs>
            <circle cx="20" cy="20" r="6" fill="url(#g)"/>
            <circle cx="44" cy="18" r="6" fill="url(#g)"/>
            <circle cx="46" cy="44" r="6" fill="url(#g)"/>
            <circle cx="18" cy="46" r="6" fill="url(#g)"/>
            <circle cx="32" cy="32" r="6" fill="url(#g)"/>
            <path d="M24 20L38 18" stroke="#111827" stroke-opacity="0.35" stroke-width="3" stroke-linecap="round"/>
            <path d="M22 24L30 28" stroke="#111827" stroke-opacity="0.35" stroke-width="3" stroke-linecap="round"/>
            <path d="M34 32L40 40" stroke="#111827" stroke-opacity="0.35" stroke-width="3" stroke-linecap="round"/>
            <path d="M30 34L22 42" stroke="#111827" stroke-opacity="0.35" stroke-width="3" stroke-linecap="round"/>
            <path d="M36 28L42 22" stroke="#111827" stroke-opacity="0.35" stroke-width="3" stroke-linecap="round"/>
        </svg>
    </div>
    <div>
        <div class="sidebar-appname">Seleksi Beasiswa</div>
        <div class="sidebar-tag">Machine Learning</div>
    </div>
</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-spacer"></div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section-title">Menu Tahapan</div>', unsafe_allow_html=True)
        st.markdown('<div style="height:0.45rem"></div>', unsafe_allow_html=True)
        pages = [
            "1) Upload & Cleaning",
            "2) EDA",
            "3) Menjalankan Model",
            "4) Bandingkan Metrik & Model Terbaik",
            "5) Prediksi",
        ]
        choice = st.radio("Pilih tahapan", pages, index=0, label_visibility="collapsed")
        st.divider()

    if choice.startswith("1"):
        page_upload_and_clean()
    elif choice.startswith("2"):
        page_eda()
    elif choice.startswith("3"):
        page_modeling()
    elif choice.startswith("4"):
        page_compare_and_best()
    else:
        page_prediction()


if __name__ == "__main__":
    main()
