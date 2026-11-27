#!/usr/bin/env python3
"""
Streamlit web UI for single-patient schizophrenia prediction.

Improvements:
 - Shows Gene x Brain mapping matrix only after a successful prediction run
 - Compute non-zero weights by combining model fused-feature importances
   with per-original 'loading_abs' values from the mapping CSV (if available)
 - Provide top-K filtering, normalization and CSV download
 - Delegates transformation/prediction to scripts/process_and_predict.py

Minimal patch: persist the computed pivot + mapping + explanation to st.session_state
so widget interactions (heatmap options) do not cause the matrix to disappear.
Also remove duplicate inline rendering so the matrix appears only once (in persistent viewer).
"""
import streamlit as st
from pathlib import Path
import numpy as np
import pandas as pd
import subprocess
import json
import sys
import os
import io
import time
import matplotlib.pyplot as plt
from PIL import Image

# ---- Config ----
PROJECT_ROOT = Path.cwd()
UPLOADS_DIR = PROJECT_ROOT / "uploads"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
PROCESS_SCRIPT = PROJECT_ROOT / "scripts" / "process_and_predict.py"

DEFAULT_MODEL = MODELS_DIR / "fusion_rf.pkl"
DEFAULT_MAP = RESULTS_DIR / "fusion_feature_component_mapping.csv"
DEFAULT_CACHE_DIR = RESULTS_DIR

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Helpers ----
def save_streamlit_upload(uploaded_file, dest_path: Path):
    """Save uploaded BytesIO to disk. Returns path."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    data = uploaded_file.getvalue()
    with open(dest_path, "wb") as fh:
        fh.write(data)
    return dest_path

def run_process_predict(cmd, timeout=300):
    """Run helper script and return (rc, stdout, stderr)."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        return -1, "", f"Timeout after {timeout}s: {e}"
    except Exception as e:
        return -2, "", str(e)

def last_json_from_stdout(stdout_text):
    """Extract last JSON object from helper stdout (if any)."""
    if not stdout_text:
        return None
    lines = [ln.strip() for ln in stdout_text.strip().splitlines() if ln.strip()]
    for ln in reversed(lines):
        try:
            obj = json.loads(ln)
            return obj
        except Exception:
            continue
    return None

def load_mapping_df(map_path: Path):
    """Load mapping CSV and normalize expected column names; return DataFrame or None."""
    if not map_path.exists():
        return None
    try:
        df = pd.read_csv(map_path)
    except Exception:
        return None
    # normalize column names to a known set
    col_map = {}
    lc = {c.lower(): c for c in df.columns}
    if 'feature_component' in lc:
        col_map[lc['feature_component']] = 'feature_component'
    elif 'feature' in lc:
        col_map[lc['feature']] = 'feature_component'
    # original modality
    if 'orig_mod' in lc:
        col_map[lc['orig_mod']] = 'orig_mod'
    elif 'orig_modality' in lc:
        col_map[lc['orig_modality']] = 'orig_mod'
    # orig_index
    if 'orig_index' in lc:
        col_map[lc['orig_index']] = 'orig_index'
    # gene symbol
    if 'gene_symbol' in lc:
        col_map[lc['gene_symbol']] = 'gene_symbol'
    # possible loading columns
    if 'loading' in lc:
        col_map[lc['loading']] = 'loading'
    if 'loading_abs' in lc:
        col_map[lc['loading_abs']] = 'loading_abs'
    if 'rank' in lc:
        col_map[lc['rank']] = 'rank'
    df = df.rename(columns=col_map)
    # ensure columns exist
    for c in ['feature_component', 'orig_mod', 'orig_index', 'gene_symbol', 'loading', 'loading_abs', 'rank']:
        if c not in df.columns:
            df[c] = pd.NA
    return df

def compute_gene_brain_matrix(map_df: pd.DataFrame, expl_df: pd.DataFrame, top_genes=None, top_brains=None):
    """
    Build gene x brain matrix using:
      - expl_df: fused_feature, importance
      - map_df: feature_component -> original rows (orig_mod, orig_index, gene_symbol, loading_abs)
    Implementation details:
      - robust numeric cast for loading_abs/loading
      - per-component softmax weighting for genes and brains
      - optional deterministic jitter to break ties when weights are flat
    """
    if map_df is None or map_df.empty or expl_df is None or expl_df.empty:
        return pd.DataFrame()

    # ---- helpers ----
    def to_float_series(s):
        if s is None:
            return pd.Series(dtype=float)
        # accept strings like "0.12", "1e-3", and coerce others to NaN
        try:
            return pd.to_numeric(s, errors="coerce")
        except Exception:
            return pd.Series([np.nan]*len(s))

    def softmax(x):
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return x
        # guard: if all NaN or all equal -> uniform
        if np.all(~np.isfinite(x)) or np.nanstd(x) == 0:
            return np.ones_like(x, dtype=float) / max(1, x.size)
        x = np.where(np.isfinite(x), x, np.nanmin(x[np.isfinite(x)]))
        x = x - np.nanmax(x)  # stabilize
        e = np.exp(x)
        s = e.sum()
        return e / s if s > 0 else np.ones_like(x) / x.size

    # ---- normalize inputs ----
    map_df = map_df.copy()
    map_df["feature_component"] = map_df["feature_component"].astype(str)
    map_df["orig_mod"] = map_df["orig_mod"].astype(str).str.lower().fillna("")
    # prefer loading_abs, else abs(loading), else NaN
    la = to_float_series(map_df.get("loading_abs"))
    lb = to_float_series(map_df.get("loading")).abs()
    map_df["loading_abs_calc"] = la.where(la.notna(), lb)

    # importances
    expl_df = expl_df.copy()
    if "fused_feature" not in expl_df.columns:
        expl_df = expl_df.rename(columns={expl_df.columns[0]: "fused_feature"})
    if "importance" not in expl_df.columns:
        if expl_df.shape[1] > 1:
            expl_df = expl_df.rename(columns={expl_df.columns[1]: "importance"})
        else:
            expl_df["importance"] = 0.0
    expl_map = {str(r["fused_feature"]): float(r["importance"]) for _, r in expl_df.iterrows()
                if pd.notna(r.get("fused_feature"))}

    links = []  # (gene, brain, weight)

    # ---- iterate per component ----
    for comp, sub in map_df.groupby("feature_component"):
        imp = float(expl_map.get(comp, 0.0))
        if imp == 0.0:
            continue

        # classify rows
        is_gene = sub["orig_mod"].str.contains("gene") | sub["gene_symbol"].notna()
        is_brain = sub["orig_mod"].str.contains("brain") | sub["orig_mod"].str.contains("fmri") | sub["orig_mod"].str.contains("voxel")
        gene_rows = sub[is_gene].copy()
        brain_rows = sub[is_brain].copy()
        if gene_rows.empty:
            gene_rows = sub[sub["gene_symbol"].notna()].copy()
        if brain_rows.empty:
            brain_rows = sub[sub["orig_index"].notna()].copy()
        if gene_rows.empty or brain_rows.empty:
            continue

        # compute softmax weights (more contrast than simple normalization)
        gw = softmax(to_float_series(gene_rows["loading_abs_calc"]).fillna(0.0).values)
        bw = softmax(to_float_series(brain_rows["loading_abs_calc"]).fillna(0.0).values)
        gene_rows = gene_rows.assign(_w=gw)
        brain_rows = brain_rows.assign(_w=bw)

        # build keys
        gene_keys = []
        for _, gr in gene_rows.iterrows():
            if pd.notna(gr.get("gene_symbol")) and str(gr.get("gene_symbol")).strip().lower() not in ("", "nan", "none"):
                gene_keys.append(str(gr.get("gene_symbol")))
            else:
                idx = gr.get("orig_index")
                gene_keys.append(f"genes:{int(idx)}" if pd.notna(idx) else "genes:unknown")

        brain_keys = []
        for _, br in brain_rows.iterrows():
            bidx = br.get("orig_index")
            brain_keys.append(str(int(bidx)) if pd.notna(bidx) and str(bidx).replace(".0", "").isdigit() else str(bidx))

        # optional deterministic jitter to break ties across components
        # (kept tiny so it doesn't dominate)
        JITTER_EPS = 1e-6

        for (gkey, gw_val) in zip(gene_keys, gene_rows["_w"].values):
            for (bkey, bw_val) in zip(brain_keys, brain_rows["_w"].values):
                base = imp * float(gw_val) * float(bw_val)
                # stable cell-specific jitter: hash(gkey,bkey,comp)
                h = hash((gkey, bkey, comp)) % 997  # prime modulus
                jitter = (h / 997.0) * JITTER_EPS
                links.append((gkey, bkey, base + jitter))

    if not links:
        return pd.DataFrame()

    df_links = pd.DataFrame(links, columns=["gene", "brain", "weight"])
    pivot = df_links.pivot_table(index="gene", columns="brain", values="weight", aggfunc="sum", fill_value=0.0)

    # top filters
    if top_genes is not None:
        topg = pivot.sum(axis=1).sort_values(ascending=False).head(top_genes).index
        pivot = pivot.loc[topg]
    if top_brains is not None:
        topb = pivot.sum(axis=0).sort_values(ascending=False).head(top_brains).index
        pivot = pivot[topb]

    # sort
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index,
                      pivot.sum(axis=0).sort_values(ascending=False).index]
    return pivot

# ---- UI ----
st.set_page_config(page_title="Schizophrenia Predictor", layout="wide")

# --- Add top-level navigation (Home / Predict) ---
# Programmatic navigation helper: handle a one-shot flag before the radio widget is created
if st.session_state.get("goto_predict", False):
    # set the radio-backed session key before the widget is instantiated
    st.session_state["nav"] = "Predict"
    # clear the one-shot flag
    st.session_state["goto_predict"] = False

# Use a session-backed radio so we can programmatically switch to Predict.
nav = st.sidebar.radio("Navigate", options=["Home", "Predict"], index=0, key="nav")

if nav == "Home":
    # Dark medical-themed hero + features + team + newsletter
    st.markdown(
        """
        <style>
        .hero {
            background: linear-gradient(120deg, #0b1220 0%, #1f1b2e 40%, #2b0f2f 100%);
            color: #ffffff;
            padding: 48px;
            border-radius: 12px;
        }
        .hero h1 { font-size:38px; margin: 6px 0 6px 0; }
        .hero p { color: #cbd5e1; font-size:16px; line-height:1.5; }
        .feature {
            background: linear-gradient(90deg, rgba(124,58,237,0.12), rgba(236,72,153,0.06));
            padding:18px; border-radius:10px; text-align:center;
        }
        .cta {
            background: linear-gradient(90deg,#7b3cff,#ff6aa2);
            color: white; padding:10px 20px; border-radius:8px; font-weight:600;
            text-decoration:none;
        }
        .section-title { color:#fff; font-weight:700; font-size:28px; margin-top:22px; }
        .muted { color: #9aa7bf; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Hero
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("<h1>Schizophrenia Project — Patient prediction & explainability</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p class='muted'>Run single-patient predictions, inspect fused feature explanations and explore a Gene × Brain mapping heatmap for interpretable insights into model decisions. Secure, local processing — results saved to results/ and uploads/.</p>",
            unsafe_allow_html=True,
        )
        # Get Started sets the navigator to Predict (no explicit rerun call)
        if st.button("Get Started", key="home_get_started"):
            # set a one-shot flag that will be read before the radio widget is created
            st.session_state["goto_predict"] = True
            # no st.experimental_rerun() — Streamlit will rerun automatically after the button interaction
    with col2:
        st.markdown(
            "<div style='background:#071028; padding:20px; border-radius:10px; color:#9fb8d9;'>"
            "<h3 style='color:#fff;'>Quick Stats</h3>"
            "<p class='muted'>Model: ensemble fusion<br>Explainability: fused-feature → gene×brain</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Features
    st.markdown("<div class='section-title'>Core features</div>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("<div class='feature'><h4>Fast single-patient prediction</h4><p class='muted'>upload gene & brain vectors or select a sample index from the server to begin analysis.</p></div>", unsafe_allow_html=True)
    with f2:
        st.markdown("<div class='feature'><h4>Interpretable mapping</h4><p class='muted'>Fused importances mapped to genes and brain indices with heatmap viewer.</p></div>", unsafe_allow_html=True)
    with f3:
        st.markdown("<div class='feature'><h4>Downloadable outputs</h4><p class='muted'>CSV downloads for predictions, explanations and gene×brain matrices.</p></div>", unsafe_allow_html=True)

    st.markdown("---")

    # Who we are / mission (medical themed)
    st.markdown("<div class='section-title'>Who we are</div>", unsafe_allow_html=True)
    st.markdown(
        "We have developed this platform to provide an interpretable software demo for schizophrenia risk prediction and explainability. "
        "This project is a platform implementation and does not represent original clinical research or validated clinical findings.",
    )

    # Team placeholders (replace with actual photos if available)
    assets_dir = PROJECT_ROOT / "assets" / "team"
    assets_dir.mkdir(parents=True, exist_ok=True)  # creates folder if missing (no harm)

    def _team_image(path: Path, fallback_brightness=200, target_width=240):
        """
        Return a PIL.Image (or numpy array fallback) resized with high-quality resampling.
        Resizing on the server side with LANCZOS avoids blurry browser scaling.
        """
        if path.exists():
            try:
                im = Image.open(str(path)).convert("RGB")
                w, h = im.size
                if w != target_width:
                    nh = max(1, int(h * (target_width / float(w))))
                    im = im.resize((target_width, nh), Image.LANCZOS)
                return im
            except Exception:
                pass
        # fallback: solid-color numpy image
        return np.full((target_width, target_width, 3), fallback_brightness, dtype=np.uint8)

    t1, t2, t3 = st.columns(3)
    with t1:
        img1 = _team_image(assets_dir / "researcher1.jpg", 200, target_width=240)
        st.image(img1, caption="Developer 1  \n ADITYA PUJARI", width=240)
        st.caption("Platform developer")
    with t2:
        img2 = _team_image(assets_dir / "researcher2.jpg", 180, target_width=240)
        st.image(img2, caption="Developer 2  \n ADITYA JAGTAP", width=240)
        st.caption("Frontend & UI")
    with t3:
        img3 = _team_image(assets_dir / "researcher3.jpg", 160, target_width=240)
        st.image(img3, caption="Developer 3  \n ADITI DIXIT", width=240)
        st.caption("ML & Explainability")

    st.markdown("---")

    # Newsletter / contact
    st.markdown("<div class='section-title'>Stay updated</div>", unsafe_allow_html=True)
    news_col1, news_col2 = st.columns([3,2])
    with news_col1:
        email = st.text_input("Email address to subscribe", key="home_news_email", placeholder="name@example.com")
    with news_col2:
        if st.button("Subscribe", key="home_news_sub"):
            st.success("Subscribed (placeholder). Add your Mailchimp/API integration to store the address.")
    st.markdown("---")
    st.info("Tip: click Get Started or pick 'Predict' from the sidebar to run the prediction workflow.")
    st.stop()

# If not Home, continue rendering the existing Predict UI below
if nav == "Predict":
    # --- UI CONTROLS ---
    st.title("Schizophrenia — Patient Prediction & Explanation")
    st.markdown(
        "Upload a single patient's gene expression vector and brain features (or pick a sample index from the dataset). "
        "Prefer PCA-transformed brain vectors for very large uploads (>200MB) or use server-side sample index."
    )

    # Sidebar inputs
    st.sidebar.header("Settings")
    model_path = st.sidebar.text_input("Model path", str(DEFAULT_MODEL))
    map_path = st.sidebar.text_input("Feature mapping CSV", str(DEFAULT_MAP))
    cache_dir = st.sidebar.text_input("PCA cache directory", str(DEFAULT_CACHE_DIR))
    pca_override = st.sidebar.text_input("Override PCA counts (JSON) — optional", "")
    top_n = st.sidebar.number_input("Top features to show", value=20, min_value=1)
    st.sidebar.markdown("---")
    st.sidebar.markdown("Upload files (single-sample). Accepted: .npy or .csv")
    uploaded_gene = st.sidebar.file_uploader("Gene vector (.npy or .csv)", type=["npy", "csv"])
    uploaded_brain = st.sidebar.file_uploader("Brain vector (.npy or .csv)", type=["npy", "csv"])
    uploaded_snp = st.sidebar.file_uploader("SNP vector (.npy or .csv) (optional)", type=["npy", "csv"])
    st.sidebar.markdown("---")
    use_sample_index = st.sidebar.checkbox("Use sample index from server dataset instead of uploading files")
    sample_index = None
    if use_sample_index:
        sample_index = st.sidebar.number_input("Sample index (0-based)", min_value=0, value=0)

    st.sidebar.markdown("---")
    st.sidebar.write(
        "Tips:\n"
        "- If brain raw is huge (>200MB), precompute PCA (server-side) and upload PCA vector instead.\n"
        "- For raw uploads, gene and brain raw must match training feature counts (server dataset shapes).\n"
        "- Make sure `results/pca_gene.npz`, `results/pca_brain.npz`, `results/pca_snp.npz` exist if you plan to upload raw."
    )

    # Initialize session_state slots for persistence
    if "last_pivot" not in st.session_state:
        st.session_state["last_pivot"] = None
    if "last_map_df" not in st.session_state:
        st.session_state["last_map_df"] = None
    if "last_expl_df" not in st.session_state:
        st.session_state["last_expl_df"] = None
    if "last_summary" not in st.session_state:
        st.session_state["last_summary"] = None

    # --- Run prediction ---
    if st.button("Run prediction"):
        # Basic checks
        if not PROCESS_SCRIPT.exists():
            st.error(f"Helper script not found: {PROCESS_SCRIPT}. Add scripts/process_and_predict.py")
            st.stop()
        if not Path(model_path).exists():
            st.error(f"Model not found: {model_path}")
            st.stop()

        # Save uploads and ensure single-row shape (if user uploaded CSV column or 1D npy)
        def write_and_fix(uploaded_file, dest_name):
            p = UPLOADS_DIR / dest_name
            save_streamlit_upload(uploaded_file, p)
            # try to load and ensure shape (1, n_features)
            try:
                if p.suffix.lower() == ".npy":
                    arr = np.load(str(p))
                else:
                    df = pd.read_csv(str(p), header=None)
                    arr = df.values
                # reshape 1D -> (1,n)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                # if multiple rows: keep first row only (warn)
                if arr.shape[0] > 1:
                    st.warning(f"{p.name} had {arr.shape[0]} rows; keeping first row only for prediction.")
                    arr = arr[:1, :]
                # save standardized .npy
                np.save(str(p.with_suffix(".npy")), arr)
                return p.with_suffix(".npy"), arr
            except Exception as e:
                st.error(f"Failed processing upload {p.name}: {e}")
                st.stop()

        saved_gene = saved_brain = saved_snp = None
        try:
            if uploaded_gene is not None:
                saved_gene, arr_g = write_and_fix(uploaded_gene, "patient_gene_uploaded" + Path(uploaded_gene.name).suffix)
                st.success(f"Saved gene upload ({arr_g.shape}) → {saved_gene}")
            elif use_sample_index:
                saved_gene = None
            else:
                st.error("Please upload gene vector or select a sample index.")
                st.stop()

            if uploaded_brain is not None:
                saved_brain, arr_b = write_and_fix(uploaded_brain, "patient_brain_uploaded" + Path(uploaded_brain.name).suffix)
                st.success(f"Saved brain upload ({arr_b.shape}) → {saved_brain}")
            elif use_sample_index:
                saved_brain = None
            else:
                st.error("Please upload brain vector or select a sample index.")
                st.stop()

            if uploaded_snp is not None:
                saved_snp, arr_s = write_and_fix(uploaded_snp, "patient_snp_uploaded" + Path(uploaded_snp.name).suffix)
                st.success(f"Saved SNP upload ({arr_s.shape}) → {saved_snp}")
            else:
                saved_snp = None

        except Exception as e:
            st.error(f"Failed to save uploads: {e}")
            st.stop()

        # If sample index used, write single-sample npys from server dataset
        if use_sample_index:
            data_dir = PROJECT_ROOT / "data"
            Xg_path = data_dir / "X_genes.npy"
            Xb_path = data_dir / "X_brain.npy"
            Xs_path = data_dir / "X_snps.npy"
            if not Xg_path.exists() or not Xb_path.exists():
                st.error("Server dataset files not found (X_genes.npy, X_brain.npy). Cannot use sample index.")
                st.stop()
            try:
                Xg = np.load(str(Xg_path))
                Xb = np.load(str(Xb_path))
                idx = int(sample_index)
                if idx < 0 or idx >= Xg.shape[0]:
                    st.error("Sample index out of range.")
                    st.stop()
                tmp_gene = UPLOADS_DIR / "patient_gene_from_index.npy"
                tmp_brain = UPLOADS_DIR / "patient_brain_from_index.npy"
                np.save(tmp_gene, Xg[[idx], :])
                np.save(tmp_brain, Xb[[idx], :])
                saved_gene = tmp_gene
                saved_brain = tmp_brain
                # optional snp
                if Xs_path.exists():
                    Xs = np.load(str(Xs_path))
                    if idx < Xs.shape[0]:
                        tmp_snp = UPLOADS_DIR / "patient_snp_from_index.npy"
                        np.save(tmp_snp, Xs[[idx], :])
                        saved_snp = tmp_snp
            except Exception as e:
                st.error(f"Failed to prepare sample-index files: {e}")
                st.stop()

        # Build command to call process_and_predict.py
        cmd = [sys.executable, str(PROCESS_SCRIPT),
               "--model", str(model_path),
               "--map", str(map_path),
               "--cache-dir", str(cache_dir),
               "--out", str(RESULTS_DIR / "patient_pred_uploaded.csv"),
               "--explain-out", str(RESULTS_DIR / "patient_pred_explain.csv"),
               "--uploads-outdir", str(UPLOADS_DIR),
               "--top-n", str(int(top_n)),
               "--verbose"
        ]
        if pca_override:
            cmd += ["--force-pca-counts", pca_override]
        if saved_gene:
            cmd += ["--gene", str(saved_gene)]
        if saved_brain:
            cmd += ["--brain", str(saved_brain)]
        if saved_snp:
            cmd += ["--snp", str(saved_snp)]

        st.info("Running prediction helper (this may take a few seconds)...")
        with st.spinner("Processing..."):
            rc, out, err = run_process_predict(cmd, timeout=300)

        if out:
            st.text_area("Subprocess stdout", value=out, height=180)
        if err:
            st.text_area("Subprocess stderr", value=err, height=180)

        if rc != 0:
            st.error(f"Prediction helper failed (rc={rc}). Check logs above.")
            st.stop()

        # parse JSON summary printed last by helper
        summary = last_json_from_stdout(out)
        if summary is None:
            st.warning("Could not parse helper JSON output. Check helper stdout above.")
            summary = {}

        # Show prediction (CSV)
        pred_csv = RESULTS_DIR / "patient_pred_uploaded.csv"
        if pred_csv.exists():
            try:
                df_pred = pd.read_csv(pred_csv)
                st.subheader("Prediction")
                st.dataframe(df_pred)
                # show metric if probability present
                prob = None
                if "predicted_prob" in summary and summary["predicted_prob"] is not None:
                    prob = float(summary["predicted_prob"])
                elif "probability_schizophrenia" in df_pred.columns:
                    prob = float(df_pred.iloc[0]["probability_schizophrenia"])
                if prob is not None:
                    st.metric("Schizophrenia probability (%)", f"{100*prob:.2f}%", delta=f"P(SZ)={prob:.3f}")
                else:
                    st.info("No probability value found in outputs to display as percentage.")
            except Exception as e:
                st.warning(f"Could not read prediction CSV: {e}")
        else:
            st.warning("Prediction CSV not found.")

        # Explanation CSV
        expl_csv = RESULTS_DIR / "patient_pred_explain.csv"
        if expl_csv.exists():
            st.subheader("Explanation (top fused features)")
            try:
                df_expl = pd.read_csv(expl_csv)
                st.table(df_expl.head(int(top_n)))
                # plot horizontal bar
                if "importance" in df_expl.columns and "fused_feature" in df_expl.columns:
                    fig, ax = plt.subplots(figsize=(6, min(6, 0.3 * len(df_expl))))
                    ax.barh(df_expl['fused_feature'][::-1], df_expl['importance'][::-1])
                    ax.set_xlabel("Approx importance")
                    st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not load explanation CSV: {e}")
        else:
            st.info("No explanation CSV produced (model may not expose feature_importances_).")

        # Fused vector preview
        fused_npy = UPLOADS_DIR / "patient_fused.npy"
        if fused_npy.exists():
            st.subheader("Fused vector preview")
            try:
                fused = np.load(str(fused_npy))
                st.write("Fused shape:", fused.shape)
                preview = fused.flatten()[:50].tolist()
                st.write(preview)
            except Exception as e:
                st.warning(f"Could not load fused vector: {e}")

        st.success("Done. Results saved under `results/` and `uploads/`.")

        # ---- Gene x Brain matrix (compute & persist only; rendering moved to persistent viewer) ----
        map_df = load_mapping_df(Path(map_path))
        if map_df is None or map_df.empty:
            st.info("Mapping CSV not found or empty. Run the explainability step to produce mapping file.")
            # clear any previous pivot
            st.session_state["last_pivot"] = pd.DataFrame()
            st.session_state["last_map_df"] = map_df
            st.session_state["last_expl_df"] = None
            st.session_state["last_summary"] = summary
        elif not expl_csv.exists():
            st.info("No explanation CSV found. Run prediction first to create explanation CSV.")
            st.session_state["last_pivot"] = pd.DataFrame()
            st.session_state["last_map_df"] = map_df
            st.session_state["last_expl_df"] = None
            st.session_state["last_summary"] = summary
        else:
            try:
                expl_df = pd.read_csv(expl_csv)
                # --- prefer per-sample importance if available, else fallback ---
                if 'fused_feature' not in expl_df.columns:
                    expl_df = expl_df.rename(columns={expl_df.columns[0]: 'fused_feature'})
                if 'importance_sample' in expl_df.columns:
                    expl_df['importance'] = expl_df['importance_sample']
                elif 'importance' in expl_df.columns:
                    pass  # already present
                elif 'importance_global' in expl_df.columns:
                    expl_df['importance'] = expl_df['importance_global']
                else:
                    if expl_df.shape[1] > 1:
                        expl_df = expl_df.rename(columns={expl_df.columns[1]: 'importance'})
                    else:
                        expl_df['importance'] = 0.0
                expl_df = expl_df[['fused_feature', 'importance']].dropna()

                # choose defaults for top filtering (persist viewer will allow changes)
                default_max_genes = 50
                default_max_brains = 50

                pivot = compute_gene_brain_matrix(map_df, expl_df, top_genes=default_max_genes, top_brains=default_max_brains)

                # Persist pivot + map_df + expl_df to session_state for the persistent viewer
                st.session_state["last_pivot"] = pivot.copy() if not pivot.empty else pd.DataFrame()
                st.session_state["last_map_df"] = map_df.copy()
                st.session_state["last_expl_df"] = expl_df.copy()
                st.session_state["last_summary"] = summary

                st.info("Gene×Brain mapping computed. Use the persistent viewer below to explore the matrix and heatmap.")
            except Exception as e:
                st.error(f"Failed to compute Gene×Brain mapping: {e}")
                st.session_state["last_pivot"] = pd.DataFrame()
                st.session_state["last_map_df"] = map_df
                st.session_state["last_expl_df"] = None
                st.session_state["last_summary"] = summary

# ------------------ Persistent viewer (reads from st.session_state) ------------------
# Single authoritative view for the matrix & heatmap (no duplicates).
st.markdown("---")
st.header("Gene × Brain Matrix (viewer)")

if st.session_state.get("last_pivot") is None or st.session_state.get("last_pivot").empty:
    st.info("No Gene×Brain matrix computed yet. Run prediction first to compute the matrix; it will appear here and remain when you change heatmap options.")
else:
    pivot = st.session_state["last_pivot"].copy()
    map_df_saved = st.session_state.get("last_map_df")
    expl_df_saved = st.session_state.get("last_expl_df")

    st.write(f"Matrix computed from mapping ({'present' if map_df_saved is not None else 'missing'}) and explanation ({'present' if expl_df_saved is not None else 'missing'}).")

    # persistent controls with explicit keys so changing them doesn't clear the pivot
    col1, col2 = st.columns(2)
    with col1:
        max_genes_p = st.number_input("Max genes to show (top by total weight) — persistent", min_value=5, max_value=500, value=50, key="persist_max_genes")
    with col2:
        max_brains_p = st.number_input("Max brain indices to show (top by total weight) — persistent", min_value=5, max_value=500, value=50, key="persist_max_brains")

    # clip pivot according to persistent controls
    gene_sums = pivot.sum(axis=1).sort_values(ascending=False)
    brain_sums = pivot.sum(axis=0).sort_values(ascending=False)
    genes_top = gene_sums.head(max_genes_p).index
    brains_top = brain_sums.head(max_brains_p).index
    mat = pivot.loc[genes_top, brains_top]

    st.subheader("Weighted Gene × Brain Matrix")
    st.dataframe(mat.style.format("{:.4f}"))

    with st.expander("Heatmap options (persistent)"):
        norm_p = st.checkbox("Normalize rows (per gene) — persistent", value=False, key="persist_norm")
        cmap_p = st.selectbox("Colormap — persistent", options=["viridis","magma","plasma","coolwarm"], index=0, key="persist_cmap")
        show_numbers = st.checkbox("Show numbers on heatmap (persistent)", value=False, key="persist_show_numbers")

    arr = mat.values.astype(float)
    if norm_p:
        row_max = arr.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0
        arr = arr / row_max

    fig, ax = plt.subplots(figsize=(min(16, 0.18*mat.shape[1] + 2), min(9, 0.18*mat.shape[0] + 2)))
    im = ax.imshow(arr, aspect="auto", interpolation="nearest", cmap=cmap_p)
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index.tolist(), fontsize=8)
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns.tolist(), rotation=90, fontsize=8)
    ax.set_title("Gene × Brain Weighted Mapping (Importance-based)")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    if show_numbers and mat.size <= 2000:
        for (i, j), val in np.ndenumerate(arr):
            ax.text(j, i, f"{val:.4f}", ha="center", va="center", color="white", fontsize=6)
    st.pyplot(fig)

    csv = mat.to_csv(index=True).encode("utf-8")
    st.download_button("Download Matrix CSV", csv, "gene_brain_matrix_weighted.csv", "text/csv")