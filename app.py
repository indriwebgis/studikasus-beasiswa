import io
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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


def clean_dataset(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    target_col: Optional[str] = None,
    drop_rows_missing_target: bool = True,
    numeric_impute: str = "median",  # "median" atau "mean"
    categorical_impute: str = "mode",  # hanya "mode" untuk saat ini
) -> Tuple[pd.DataFrame, CleaningLog]:
    """Cleaning yang aman untuk berbagai input file.

    - Trim nama kolom & string kategorikal
    - Drop duplikat
    - Coerce numerik
    - Imputasi missing value (numeric median/mean, kategorikal mode)
    """
    work = df.copy()

    # Strip whitespace nama kolom
    work.columns = [str(c).strip() for c in work.columns]

    # Drop duplikat
    before = len(work)
    work = work.drop_duplicates()
    after_dedup = len(work)
    dropped_dup = before - after_dedup

    # Trim string kategorikal
    for c in categorical_cols:
        if c in work.columns:
            work[c] = work[c].astype(str).str.strip()

    # Coerce numerik
    coerced = []
    for c in numeric_cols:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")
            coerced.append(c)

    # (Opsional) buang baris tanpa target
    if target_col and target_col in work.columns and drop_rows_missing_target:
        work = work[work[target_col].notna()].copy()

    # Imputasi numeric
    imputed_num: Dict[str, float] = {}
    for c in numeric_cols:
        if c in work.columns:
            if work[c].isna().any():
                if numeric_impute == "mean":
                    val = float(work[c].mean())
                else:
                    val = float(work[c].median())
                work[c] = work[c].fillna(val)
                imputed_num[c] = val

    # Imputasi kategorikal (mode)
    imputed_cat: Dict[str, str] = {}
    for c in categorical_cols:
        if c in work.columns:
            if work[c].isna().any():
                mode = work[c].mode(dropna=True)
                fill = str(mode.iloc[0]) if len(mode) else "Tidak Diketahui"
                work[c] = work[c].fillna(fill)
                imputed_cat[c] = fill

    log = CleaningLog(
        rows_before=len(df),
        rows_after=len(work),
        dropped_duplicates=dropped_dup,
        coerced_numeric=coerced,
        imputed_numeric=imputed_num,
        imputed_categorical=imputed_cat,
    )
    return work, log


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
        results.append(
            EvalResult(
                name=name,
                accuracy=acc,
                precision=prec,
                recall=rec,
                f1=f1,
                cm=cm,
                report=report,
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
def plot_hist(df: pd.DataFrame, col: str, bins: int):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.hist(df[col].dropna().to_numpy(), bins=bins)
    ax.set_title(f"Histogram: {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    st.pyplot(fig, clear_figure=True)


def plot_correlation(df: pd.DataFrame, cols: List[str]):
    import matplotlib.pyplot as plt

    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots()
    im = ax.imshow(corr.values)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    ax.set_title("Correlation (Pearson)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig, clear_figure=True)


def plot_confusion_matrix(cm: np.ndarray, title: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center")
    st.pyplot(fig, clear_figure=True)


# =========================
# App
# =========================
def init_state():
    st.session_state.setdefault("raw_df", None)
    st.session_state.setdefault("clean_df", None)
    st.session_state.setdefault("clean_log", None)
    st.session_state.setdefault("eval_results", None)
    st.session_state.setdefault("best_result", None)
    st.session_state.setdefault("target_col", DEFAULT_TARGET)


def page_upload_and_clean():
    st.header("1) Upload & Cleaning")

    st.caption("Jika Anda tidak mengunggah data baru, Anda bisa menggunakan data contoh yang sudah dilampirkan (beasiswa.csv).")

    uploaded = st.file_uploader("Upload dataset (.csv / .xlsx)", type=["csv", "xlsx", "xls"])
    use_sample = st.toggle("Gunakan data contoh (beasiswa.csv) jika tidak upload", value=True)

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
    st.write("**Skema kolom:**")
    st.code(_expected_columns_text(), language="text")

    st.subheader("Preview data (raw)")
    st.dataframe(df.head(50), use_container_width=True)
    st.write(f"Rows: {len(df):,} | Columns: {df.shape[1]}")

    st.subheader("Konfigurasi cleaning")
    c1, c2, c3 = st.columns(3)
    with c1:
        target_col = st.text_input("Nama kolom target", value=st.session_state["target_col"])
    with c2:
        drop_missing_target = st.checkbox("Buang baris yang target-nya kosong", value=True)
    with c3:
        numeric_impute = st.selectbox("Imputasi numerik", ["median", "mean"], index=0)

    # Jalankan cleaning
    if st.button("Jalankan cleaning", type="primary"):
        clean_df, log = clean_dataset(
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
        st.session_state["target_col"] = target_col

    if st.session_state["clean_df"] is None:
        st.info("Klik **Jalankan cleaning** untuk menyimpan dataset hasil cleaning.")
        return

    clean_df: pd.DataFrame = st.session_state["clean_df"]
    log: CleaningLog = st.session_state["clean_log"]

    st.subheader("Hasil cleaning")
    st.write(
        {
            "rows_before": log.rows_before,
            "rows_after": log.rows_after,
            "dropped_duplicates": log.dropped_duplicates,
            "imputed_numeric_cols": list(log.imputed_numeric.keys()),
            "imputed_categorical_cols": list(log.imputed_categorical.keys()),
        }
    )
    st.dataframe(clean_df.head(50), use_container_width=True)

    st.download_button(
        "Download dataset hasil cleaning (CSV)",
        data=_to_csv_bytes(clean_df),
        file_name="beasiswa_cleaned.csv",
        mime="text/csv",
    )


def page_eda():
    st.header("2) EDA (Exploratory Data Analysis)")

    df: Optional[pd.DataFrame] = st.session_state.get("clean_df")
    if df is None:
        st.warning("Belum ada data hasil cleaning. Silakan ke tahap **Upload & Cleaning** dulu.")
        return

    target_col = st.session_state.get("target_col", DEFAULT_TARGET)

    st.subheader("Ringkasan data")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rows", f"{len(df):,}")
    with c2:
        st.metric("Columns", f"{df.shape[1]}")
    with c3:
        miss = int(df.isna().sum().sum())
        st.metric("Total missing", f"{miss:,}")

    st.subheader("Tipe data")
    st.dataframe(df.dtypes.astype(str).reset_index().rename(columns={"index": "column", 0: "dtype"}), use_container_width=True)

    st.subheader("Statistik deskriptif (numerik)")
    present_numeric = [c for c in NUMERIC_COLS if c in df.columns]
    if present_numeric:
        st.dataframe(df[present_numeric].describe().T, use_container_width=True)
    else:
        st.info("Tidak ditemukan kolom numerik yang diharapkan.")

    st.subheader("Distribusi target")
    if target_col in df.columns:
        vc = df[target_col].value_counts(dropna=False)
        st.bar_chart(vc)
        st.write(vc)
    else:
        st.info(f"Kolom target **{target_col}** tidak ada di dataset ini (EDA tetap bisa, tapi training butuh target).")

    st.subheader("Plot cepat")
    c1, c2 = st.columns(2)
    with c1:
        if present_numeric:
            col = st.selectbox("Histogram kolom numerik", present_numeric, index=0)
            bins = st.slider("Bins", 5, 60, 25, 1)
            plot_hist(df, col, bins)
    with c2:
        present_cat = [c for c in CATEGORICAL_COLS if c in df.columns]
        if present_cat:
            cat = st.selectbox("Frekuensi kolom kategorikal", present_cat, index=0)
            st.bar_chart(df[cat].value_counts().head(30))

    st.subheader("Korelasi antar fitur numerik")
    if len(present_numeric) >= 2:
        plot_correlation(df, present_numeric)
    else:
        st.info("Butuh minimal 2 kolom numerik untuk korelasi.")


def page_modeling():
    st.header("3) Menjalankan Model")

    df: Optional[pd.DataFrame] = st.session_state.get("clean_df")
    if df is None:
        st.warning("Belum ada data hasil cleaning. Silakan ke tahap **Upload & Cleaning** dulu.")
        return

    target_col = st.session_state.get("target_col", DEFAULT_TARGET)
    if target_col not in df.columns:
        st.error(f"Kolom target **{target_col}** tidak ditemukan. Pastikan dataset memiliki target untuk training.")
        return

    missing_cols = [c for c in (NUMERIC_COLS + CATEGORICAL_COLS) if c not in df.columns]
    if missing_cols:
        st.error("Dataset Anda tidak memiliki semua kolom fitur yang dibutuhkan:")
        st.write(missing_cols)
        return

    st.subheader("Pengaturan training")
    c1, c2, c3 = st.columns(3)
    with c1:
        test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    with c2:
        random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)
    with c3:
        balanced = st.checkbox("Gunakan class_weight=balanced (untuk data imbalanced)", value=True)

    default_models = ["Logistic Regression", "Random Forest"] + (["XGBoost"] if HAS_XGBOOST else [])
    selected_models = st.multiselect(
        "Pilih model yang akan dijalankan",
        options=default_models,
        default=default_models,
    )

    if not HAS_XGBOOST:
        st.info("Catatan: XGBoost tidak tersedia di environment ini. (Aplikasi tetap berjalan.)")

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
        st.info("Klik **Train & Evaluate** untuk menjalankan model.")
        return

    st.success(f"Selesai. Model yang dievaluasi: {len(results)}")

    st.subheader("Ringkasan hasil (sementara)")
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
    ).sort_values("F1-score", ascending=False)

    st.dataframe(metrics_df, use_container_width=True)

    st.subheader("Detail tiap model")
    for r in results:
        with st.expander(f"{r.name} — detail"):
            c1, c2 = st.columns([1, 1])
            with c1:
                st.write(
                    {
                        "Accuracy": r.accuracy,
                        "Precision": r.precision,
                        "Recall": r.recall,
                        "F1-score": r.f1,
                    }
                )
            with c2:
                plot_confusion_matrix(r.cm, title=f"Confusion Matrix: {r.name}")

            st.text("Classification report:")
            st.code(r.report, language="text")


def page_compare_and_best():
    st.header("4) Membandingkan Metrik & Memilih Model Terbaik")

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

    st.subheader("Tabel perbandingan metrik (Accuracy, Precision, Recall, F1-score)")
    st.dataframe(metrics_df.sort_values("F1-score", ascending=False), use_container_width=True)

    st.subheader("Visual perbandingan")
    metric = st.selectbox("Pilih metrik untuk dibandingkan", ["Accuracy", "Precision", "Recall", "F1-score"], index=3)
    chart_df = metrics_df.set_index("Model")[[metric]].sort_values(metric, ascending=False)
    st.bar_chart(chart_df)

    st.subheader("Memilih model terbaik (kuantitatif)")
    st.caption(
        "Secara default, banyak studi memilih **F1-score** sebagai metrik utama ketika data target tidak seimbang "
        "(karena mempertimbangkan precision & recall sekaligus). Namun Anda bebas memilih metrik di bawah."
    )

    best_metric = st.selectbox("Metrik penentu model terbaik", ["Accuracy", "Precision", "Recall", "F1-score"], index=3)
    best = pick_best_model(results, best_metric)
    st.session_state["best_result"] = best

    if best is None:
        st.error("Tidak bisa menentukan model terbaik (hasil kosong).")
        return

    st.success(f"Model terbaik berdasarkan **{best_metric}** adalah: **{best.name}**")
    st.write(
        {
            "Accuracy": best.accuracy,
            "Precision": best.precision,
            "Recall": best.recall,
            "F1-score": best.f1,
        }
    )
    plot_confusion_matrix(best.cm, title=f"Confusion Matrix (Best): {best.name}")
    st.code(best.report, language="text")


def page_prediction():
    st.markdown('<div class="page-title">\U0001F52E 5) Prediksi</div>', unsafe_allow_html=True)

    best: Optional[EvalResult] = st.session_state.get("best_result")
    df: Optional[pd.DataFrame] = st.session_state.get("clean_df")
    if df is None:
        st.warning("Belum ada data. Mulai dari tahap **Upload & Cleaning**.")
        return
    if best is None:
        st.warning("Belum ada model terbaik. Tentukan dulu di tahap **Membandingkan Metrik & Memilih Model Terbaik**.")
        return

    st.subheader("Prediksi 1 data (form)")
    with st.form("single_pred"):
        vals_num = {}
        for c in NUMERIC_COLS:
            vals_num[c] = st.number_input(c, value=float(df[c].median()) if c in df.columns else 0.0)
        vals_cat = {}
        for c in CATEGORICAL_COLS:
            opts = sorted(df[c].astype(str).unique().tolist()) if c in df.columns else []
            default = opts[0] if opts else ""
            vals_cat[c] = st.selectbox(c, options=opts, index=0) if opts else st.text_input(c, value=default)
        submitted = st.form_submit_button("Prediksi")

    if submitted:
        one = {**vals_num, **vals_cat}
        X_one = pd.DataFrame([one])
        pred = int(best.fitted.predict(X_one)[0])
        proba = None
        if hasattr(best.fitted, "predict_proba"):
            try:
                proba = float(best.fitted.predict_proba(X_one)[0, 1])
            except Exception:
                proba = None

        st.success(f"Prediksi label: **{pred}** (1 = diterima, 0 = tidak)")
        if proba is not None:
            st.write({"probabilitas_diterima": proba})

    st.subheader("Prediksi batch (upload CSV)")
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
        dnew2, _ = clean_dataset(
            dnew,
            numeric_cols=NUMERIC_COLS,
            categorical_cols=CATEGORICAL_COLS,
            target_col=None,
            drop_rows_missing_target=False,
        )

        yhat = best.fitted.predict(dnew2[NUMERIC_COLS + CATEGORICAL_COLS])
        out = dnew2.copy()
        out["Prediksi_Label"] = yhat

        if hasattr(best.fitted, "predict_proba"):
            try:
                out["Prob_Diterima"] = best.fitted.predict_proba(dnew2[NUMERIC_COLS + CATEGORICAL_COLS])[:, 1]
                out = out.sort_values("Prob_Diterima", ascending=False)
            except Exception:
                pass

        st.dataframe(out.head(50), use_container_width=True)

        st.download_button(
            "Download hasil prediksi (CSV)",
            data=_to_csv_bytes(out),
            file_name="hasil_prediksi_beasiswa.csv",
            mime="text/csv",
        )


def main():
    st.set_page_config(page_title="Seleksi Beasiswa - Machine Learning", layout="wide")
    init_state()

    st.title("Sistem Seleksi Beasiswa dengan Machine Learning")
    st.caption("Tahapan: Upload & cleaning → EDA → Menjalankan model → Bandingkan metrik → Pilih model terbaik (kuantitatif).")

    with st.sidebar:
        st.header("Menu Tahapan")
        pages = [
            "1) Upload & Cleaning",
            "2) EDA",
            "3) Menjalankan Model",
            "4) Bandingkan Metrik & Model Terbaik",
            "5) Prediksi",
        ]
        choice = st.radio("Pilih tahapan", pages, index=0)
        st.divider()
        st.subheader("Catatan")
        st.write(
            "- Jika tidak upload data, aktifkan **Gunakan data contoh**.\n"
            "- Pastikan kolom fitur sesuai dengan skema yang telah ditentukan.\n"
        )

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
