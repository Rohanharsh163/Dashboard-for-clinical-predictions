# =============================================================================
# BITS F464 Machine Learning
# Automated ML Pipeline and Dashboard for Clinical Prediction under Temporal Shift
#
# Team 30
#   1. Sailesh Nichenametla    — 2023A7PS0147H
#   2. Rohan Harshith Amarthaluri — 2023A7PS0015H
#   3. Dhanush Thirunagari     — 2023A2PS1280H
#   4. Aniket Shukla           — 2023A7PS0134H
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os, io, warnings
from pathlib import Path
from sklearn.multioutput import MultiOutputClassifier

warnings.filterwarnings("ignore")

# ── sklearn ──────────────────────────────────────────────────────────────────
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, multilabel_confusion_matrix,
    roc_curve, auc, classification_report,
    ConfusionMatrixDisplay, make_scorer
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.inspection import permutation_importance 

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Team 30 — Clinical ML Pipeline",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .team-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1.5rem;
        color: white;
    }
    .metric-card {
        background: #f8f9fa; border-left: 4px solid #0f3460;
        padding: 0.8rem 1rem; border-radius: 6px; margin-bottom: 0.5rem;
    }
    .section-header {
        background: #0f3460; color: white;
        padding: 0.5rem 1rem; border-radius: 6px; margin: 1rem 0 0.5rem 0;
        font-weight: 600;
    }
    .stTab > div { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Clinical ML Pipeline")
    st.markdown("**Team 30 · BITS F464**")
    st.markdown("---")

    # st.markdown("### 📁 Upload CSV Files")
    # st.caption("Upload the 4 EHR CSV files from the assignment dataset folder.")
    # patients_file    = st.file_uploader("patients.csv",     type="csv", key="pat")
    # conditions_file  = st.file_uploader("conditions.csv",   type="csv", key="con")
    # observations_file= st.file_uploader("observations.csv", type="csv", key="obs")
    # encounters_file  = st.file_uploader("encounters.csv",   type="csv", key="enc")
    # medications_file   = st.file_uploader("medications.csv", type="csv", key = "med")
    # procedures_file    = st.file_uploader("procedures.csv", type="csv", key = "proc")
    # allergies_file     = st.file_uploader("allergies.csv", type="csv", key = "all")
    # immunizations_file = st.file_uploader("immunizations.csv", type="csv", key = "imm")
    # devices_file       = st.file_uploader("devices.csv", type="csv", key = "dev")
    # st.markdown("---")
    
    st.markdown("### ⚙️ Pipeline Settings")

    split_date = st.date_input(
        "Temporal Split Date"
        "\nRecommended Split: 2025-10-22 (approximate 60:40 split)",
        value=pd.Timestamp("2025-10-22").date(),
        help="Records before this date → Dataset 1 (Historical). On/after → Dataset 2 (Current). "
             "2025-10-22 is the recommended split — it gives a roughly 60/40 historical/current "
             "divide across this Synthea dataset and captures meaningful temporal drift."
    )
    test_size = st.slider("Test Set Size (%)", 10, 40, 20, 5) / 100

    st.markdown("---")
    st.markdown("### 🎯 Prediction Mode")
    pred_mode = st.radio(
        "Target representation",
        ["Multilabel (top-10 conditions)", "Binary (any condition present)"],
        index=0,
        help=(
            "**Multilabel**: predict a separate 0/1 flag for each of the 10 most common "
            "conditions. More expressive, but harder to learn — precision/recall are averaged "
            "across all 10 labels.\n\n"
            "**Binary**: predict whether the patient has *any* of the top-10 conditions "
            "(TARGET column). One output, better class balance, higher F1 — useful for "
            "directly comparing feature representations (Assignment Task 3f)."
        ),
    )
    is_binary = pred_mode.startswith("Binary")

    st.markdown("---")
    st.markdown("### 🌳 Decision Tree")
    dt_max_depth  = st.slider("Max Depth",    1, 20, 5)
    dt_min_split  = st.slider("Min Samples Split", 2, 50, 10)

    st.markdown("### 🔲 SVM")
    svm_C     = st.selectbox("C (Regularization)", [0.01, 0.1, 1, 10, 100], index=2)
    svm_kernel= st.selectbox("Kernel", ["rbf", "linear", "poly"], index=0)

    st.markdown("### 🧠 Neural Network (MLP)")
    mlp_layers  = st.text_input("Hidden Layers (e.g. 64,32)", value="64,32")
    mlp_lr      = st.selectbox("Learning Rate", [0.001, 0.01, 0.1], index=0)
    mlp_epochs  = st.slider("Max Iterations", 100, 500, 200, 50)

    st.markdown("---")
    run_pipeline = st.button("▶ Run Full Pipeline", type="primary", width='stretch')


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="team-header">
    <h2 style="margin:0">🏥 Automated ML Pipeline — Clinical Prediction under Temporal Shift</h2>
    <p style="margin:0.3rem 0 0 0; opacity:0.85">
        BITS F464 Machine Learning · Assignment 2 · Team 30<br>
        Sailesh Nichenametla · Rohan Harshith Amarthaluri · Dhanush Thirunagari · Aniket Shukla
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

# @st.cache_data(show_spinner=False)
# def load_and_merge(pat_bytes, con_bytes, obs_bytes, enc_bytes,
#                        med_bytes=None, proc_bytes=None,
#                        allergy_bytes=None, imm_bytes=None, dev_bytes=None):
#     """Load and merge EHR tables into a single patient-level DataFrame."""
#     patients = pd.read_csv(io.BytesIO(pat_bytes), on_bad_lines='skip', header = 0)
#     conditions = pd.read_csv(io.BytesIO(con_bytes), on_bad_lines='skip')
#     observations = pd.read_csv(io.BytesIO(obs_bytes), on_bad_lines='skip')
#     encounters = pd.read_csv(io.BytesIO(enc_bytes), on_bad_lines='skip')

#     # ── Standardise column names to lowercase
#     for df in [patients, conditions, observations, encounters]:
#         df.columns = df.columns.str.lower().str.strip()

#     medications = pd.read_csv(io.BytesIO(med_bytes), on_bad_lines='skip') if med_bytes else None
#     procedures  = pd.read_csv(io.BytesIO(proc_bytes), on_bad_lines='skip') if proc_bytes else None
#     allergies   = pd.read_csv(io.BytesIO(allergy_bytes), on_bad_lines='skip') if allergy_bytes else None
#     immunizations = pd.read_csv(io.BytesIO(imm_bytes), on_bad_lines='skip') if imm_bytes else None
#     devices     = pd.read_csv(io.BytesIO(dev_bytes), on_bad_lines='skip') if dev_bytes else None

#     for df in [medications, procedures, allergies, immunizations, devices]:
#         if df is not None:
#            df.columns = df.columns.str.lower().str.strip()

#     return patients, conditions, observations, encounters, \
#        medications, procedures, allergies, immunizations, devices

@st.cache_data(show_spinner=False)
def load_and_merge():
    allergies_file = pd.read_csv("synthea-mimic/csv/allergies.csv", on_bad_lines='skip')
    careplans_file = pd.read_csv("synthea-mimic/csv/careplans.csv", on_bad_lines='skip')
    claims_transactions_file = pd.read_csv("synthea-mimic/csv/claims_transactions.csv", on_bad_lines='skip')
    claims_file = pd.read_csv("synthea-mimic/csv/claims.csv", on_bad_lines='skip')
    conditions_file = pd.read_csv("synthea-mimic/csv/conditions.csv", on_bad_lines='skip')
    devices_file = pd.read_csv("synthea-mimic/csv/devices.csv", on_bad_lines='skip')
    encounters_file = pd.read_csv("synthea-mimic/csv/encounters.csv", on_bad_lines='skip')
    imaging_studies_file = pd.read_csv("synthea-mimic/csv/imaging_studies.csv", on_bad_lines='skip')
    immunizations_file = pd.read_csv("synthea-mimic/csv/immunizations.csv", on_bad_lines='skip')
    medications_file = pd.read_csv("synthea-mimic/csv/medications.csv", on_bad_lines='skip')
    observations_file = pd.read_csv("synthea-mimic/csv/observations.csv", on_bad_lines='skip')
    organizations_file = pd.read_csv("synthea-mimic/csv/organizations.csv", on_bad_lines='skip')
    patients_file = pd.read_csv("synthea-mimic/csv/patients.csv", on_bad_lines='skip', header=0)
    payer_transitions_file = pd.read_csv("synthea-mimic/csv/payer_transitions.csv", on_bad_lines='skip')
    procedures_file = pd.read_csv("synthea-mimic/csv/procedures.csv", on_bad_lines='skip')
    providers_file = pd.read_csv("synthea-mimic/csv/providers.csv", on_bad_lines='skip')
    supplies_file = pd.read_csv("synthea-mimic/csv/supplies.csv", on_bad_lines='skip')

    dfs = [allergies_file, careplans_file, claims_transactions_file, claims_file, conditions_file, devices_file, encounters_file, imaging_studies_file, immunizations_file, 
             medications_file, observations_file, organizations_file, patients_file, payer_transitions_file, procedures_file, providers_file, supplies_file]
    
    for df in dfs:
        df.columns = df.columns.str.lower().str.strip()

    return allergies_file, careplans_file, claims_transactions_file, claims_file, conditions_file, devices_file, encounters_file, imaging_studies_file, immunizations_file, \
             medications_file, observations_file, organizations_file, patients_file, payer_transitions_file, procedures_file, providers_file, supplies_file


def detect_patient_col(df, candidates=("patient", "patient_id", "patientid", "id")):
    for c in candidates:
        if c in df.columns:
            return c
    # fall back to first column
    return df.columns[0]


def detect_date_col(df, candidates=("start", "date", "start_date", "encounterdate",
                                     "observation_date", "onset", "stop")):
    for c in candidates:
        if c in df.columns:
            return c
    # try any column with 'date' in name
    date_cols = [c for c in df.columns if "date" in c.lower()]
    return date_cols[0] if date_cols else None


def engineer_features(patients, conditions, observations, encounters, split_date_str, 
                      medications=None, procedures=None, allergies=None, immunizations=None, devices=None):
    """
    Full feature engineering pipeline:
      - Aggregate observations per patient (mean, std, count per type)
      - Merge demographics
      - Derive classification target from conditions
      - Attach encounter dates for temporal split
    """
    split_ts = pd.Timestamp(split_date_str)

    # ── Patient demographics ─────────────────────────────────────────────────
    pat_id_col = detect_patient_col(patients)
    patients   = patients.rename(columns={pat_id_col: "PATIENT"})

    birth_col  = "birthdate"
    gen_col  = "gender"
    race_col = "race"
    eth_col  = "ethnicity"
    age_col = "age"

    demo_cols = ["PATIENT"]
    demo_cols.append(birth_col)
    demo_cols.append(gen_col)
    demo_cols.append(race_col)
    demo_cols.append(eth_col)

    pat_demo = patients[demo_cols].copy()

    # Compute age from birthdate if needed
    pat_demo[birth_col] = pd.to_datetime(pat_demo[birth_col], errors="coerce")
    pat_demo[age_col] = (pd.Timestamp.now() - pat_demo[birth_col]).dt.days / 365.25
    pat_demo = pat_demo.drop(columns=[birth_col])

    # Reset index so that pat_demo and the new OHE columns share the same
    # 0-based integer index. Without this, pd.concat(..., axis=1) would
    # silently introduce NaN columns wherever the indices don't match.
    pat_demo = pat_demo.reset_index(drop=True)

    ohe = OneHotEncoder(sparse_output=False)
    pat_demo_data = ohe.fit_transform(pat_demo[[gen_col]].fillna("unknown").astype(str))
    encoded_data = pd.DataFrame(pat_demo_data, columns=ohe.get_feature_names_out())
    pat_demo = pd.concat([pat_demo, encoded_data], axis=1)   # axis=1 → add columns, not rows
    pat_demo = pat_demo.drop(columns=[gen_col])

    ohe2 = OneHotEncoder(sparse_output=False)
    pat_demo_data = ohe2.fit_transform(pat_demo[[race_col]].fillna("unknown").astype(str))
    encoded_data = pd.DataFrame(pat_demo_data, columns=ohe2.get_feature_names_out())
    pat_demo = pd.concat([pat_demo, encoded_data], axis=1)   # axis=1 → add columns, not rows
    pat_demo = pat_demo.drop(columns=[race_col])

    ohe3 = OneHotEncoder(sparse_output=False)
    pat_demo_data = ohe3.fit_transform(pat_demo[[eth_col]].fillna("unknown").astype(str))
    encoded_data = pd.DataFrame(pat_demo_data, columns=ohe3.get_feature_names_out())
    pat_demo = pd.concat([pat_demo, encoded_data], axis=1)   # axis=1 → add columns, not rows
    pat_demo = pat_demo.drop(columns=[eth_col])

    # ── Observations: aggregate numeric values per patient ───────────────────
    obs_id_col = detect_patient_col(observations)
    observations = observations.rename(columns={obs_id_col: "PATIENT"})

    # Identify numeric value column
    val_col = "value"
    type_col = "type"
    code_col = "code"

    obs_numeric = observations.copy()
    obs_numeric[val_col] = pd.to_numeric(obs_numeric[val_col], errors="coerce")
    obs_numeric = obs_numeric.dropna(subset=[val_col])


    # ── Filter to clinically meaningful observation categories ──────────────
    # The observations table contains survey responses (e.g. "What was your best
    # estimate of pain") that dominate frequency counts but are not useful clinical
    # features.  Keep only vital-signs and laboratory measurements.
    CLINICAL_CATEGORIES = {"vital-signs", "laboratory"}
    cat_col = "category"
    if cat_col in obs_numeric.columns:
        obs_clinical = obs_numeric[
            obs_numeric[cat_col].str.lower().isin(CLINICAL_CATEGORIES)
        ].copy()
        # Fall back to full set if the filter leaves nothing (edge case)
        if len(obs_clinical) == 0:
            obs_clinical = obs_numeric
    else:
        obs_clinical = obs_numeric

    # Pivot: mean & std of top observation types per patient
    # Using head(10) keeps memory usage manageable on the 1.5M-row observations file
    # while still covering the most clinically informative measurements.
    top_codes = obs_clinical[code_col].value_counts().head(10).index

    # Build a code→description lookup so pivot columns get readable names
    # (avoids ugly names like "2356890.3_mean" in the dashboard tables and plots)
    obs_desc_lookup = (
        obs_clinical.drop_duplicates(subset=[code_col])
        .set_index(code_col)["description"]
        .to_dict()
    )

    def _clean_obs_name(code):
        """Convert a raw observation code to a clean, readable column-name token."""
        desc = obs_desc_lookup.get(code, str(code))
        cleaned = "".join(c if c.isalnum() or c == " " else "_" for c in str(desc))
        return cleaned.replace(" ", "_")[:25].strip("_")

    obs_top   = obs_clinical[obs_clinical[code_col].isin(top_codes)]
    obs_pivot_mean = obs_top.pivot_table(
        index="PATIENT", columns=code_col, values=val_col, aggfunc="mean"
    )
    obs_pivot_std  = obs_top.pivot_table(
        index="PATIENT", columns=code_col, values=val_col, aggfunc="std"
    )
    # NOTE: capture the observation value column name BEFORE code_col is re-assigned
    # further down in this function. Using val_col here is correct ("value"), but
    # obs_count must be computed NOW before val_col is shadowed.
    obs_val_col_snapshot = val_col   # = "value"
    obs_numeric_for_count = observations.copy()
    obs_numeric_for_count[obs_val_col_snapshot] = pd.to_numeric(
        obs_numeric_for_count[obs_val_col_snapshot], errors="coerce"
    )
    obs_count = obs_numeric_for_count.groupby("PATIENT")[obs_val_col_snapshot].count()
    obs_count.name = "num_observations"   # Series.name is correct; .columns would crash

    # Use NaN (not -1) for missing obs pivots so the later median imputation step
    # fills them with clinically meaningful values instead of a spurious -1 signal.
    # The -1 sentinel was misleading: it mixed "not measured" with actual negative values.
    numeric_cols = obs_pivot_mean.select_dtypes(include=[np.number]).columns
    # obs_pivot_mean already has NaN for missing cells; leave them as-is for median fill later.
    # obs_pivot_std likewise — NaN for patients with only 1 observation (std undefined).
    pass
        

    # Build human-readable column names; deduplicate if two codes share the same description
    def _make_unique_names(codes, suffix):
        seen = {}
        result = []
        for c in codes:
            base = f"{_clean_obs_name(c)}{suffix}"
            if base in seen:
                seen[base] += 1
                result.append(f"{base}_{seen[base]}")
            else:
                seen[base] = 0
                result.append(base)
        return result

    obs_pivot_mean.columns = _make_unique_names(obs_pivot_mean.columns, "_mean")
    obs_pivot_std.columns  = _make_unique_names(obs_pivot_std.columns,  "_std")
    # obs_count is a Series indexed by PATIENT — join it directly onto the DataFrame
    obs_features = obs_pivot_mean.join(obs_pivot_std, how="outer").join(obs_count, how="outer").reset_index()


    # ── Encounters: get latest encounter date per patient ────────────────────
    enc_id_col   = detect_patient_col(encounters)
    encounters   = encounters.rename(columns={enc_id_col: "PATIENT"})
    enc_date_col = detect_date_col(encounters)

    encounters[enc_date_col] = pd.to_datetime(encounters[enc_date_col], errors="coerce")
    enc_latest = encounters.groupby("PATIENT")[enc_date_col].max().reset_index()
    enc_latest.columns = ["PATIENT", "LAST_ENCOUNTER"]

    # ── Conditions: define classification target ─────────────────────────────
    cond_id_col = detect_patient_col(conditions)
    conditions  = conditions.rename(columns={cond_id_col: "PATIENT"})

    code_col = "code"
    desc_col = "description"

    # Using head(10) keeps multilabel output dimension practical (10-label classification
    # is already non-trivial) and speeds up SVM and MLP training.
    top_conds = conditions[code_col].value_counts().head(10).index
    cond_filtered = conditions[conditions[code_col].isin(top_conds)]

    cond_matrix = pd.crosstab(cond_filtered['PATIENT'], cond_filtered[code_col])

    # Prefix every condition-code column with "HAS_" so that:
    #   (a) the existing exclude set (startswith("HAS_")) removes them from features,
    #       preventing data leakage into the model, AND
    #   (b) individual condition flags are still visible in the merged DataFrame for EDA.
    cond_matrix.columns = [f"HAS_{c}" for c in cond_matrix.columns]

    cond_target = (cond_matrix > 0).astype(int).reset_index()

    # Binary TARGET: 1 if the patient has ANY of the top conditions, else 0.
    # Derived AFTER building individual flags so models only see TARGET, not the flags.
    has_cols = [c for c in cond_target.columns if c.startswith("HAS_")]
    cond_target["TARGET"] = (cond_target[has_cols].sum(axis=1) > 0).astype(int)

    code_to_desc = dict(zip(conditions[code_col], conditions[desc_col]))

    # Human-readable label for the most common condition (used in the dashboard display)
    primary_code = conditions[code_col].value_counts().index[0]
    primary_desc = str(code_to_desc.get(primary_code, f"Condition {primary_code}"))
    target_label = f"Multilabel: top-10 conditions (primary: {primary_desc})"

    # label_names maps each HAS_ column to the human-readable condition description.
    # Used in classification reports and confusion matrix titles throughout the dashboard.
    label_names = {
        f"HAS_{c}": str(code_to_desc.get(c, f"Condition {c}"))
        for c in top_conds
    }

    # Top conditions by frequency → binary target: has any of top-3 conditions
    # top_conds = conditions[code_col].value_counts().head(10).index.tolist()
    # cond_flags = {}
    # for cond in top_conds:
    #      if isinstance(cond, str):
    #         clean_cond = "".join(c for c in cond if c.isalnum())[:20].upper()
    #      else:
    #         clean_cond = "UNKNOWN"
    #
    #      flag_name = "HAS_" + clean_cond
    #
    # # Primary target: has the most common condition
    # primary_cond_name = "TOP3_CONDITIONS"
    #
    # cond_target = pd.DataFrame({
    #     "PATIENT": conditions["PATIENT"].unique()
    # })
    #
    # cond_target["TARGET"] = cond_target["PATIENT"].isin(
    #     conditions[conditions[code_col].isin(top_conds)]["PATIENT"]
    # ).astype(int)
    #
    # # Also add presence flags for top-3 conditions
    # for flag_name, pats in cond_flags.items():
    #     cond_target[flag_name] = cond_target["PATIENT"].isin(pats).astype(int)
    #
    # cond_target = pd.DataFrame({
    #     "PATIENT": conditions["PATIENT"].unique(),
    #     "TARGET":  1
    # })
    # primary_cond_name = "Medical Condition"

    # ── Merge all ────────────────────────────────────────────────────────────
    merged = (
        pat_demo
        .merge(obs_features, on="PATIENT", how="left")
        .merge(enc_latest,   on="PATIENT", how="left")
        .merge(cond_target,  on="PATIENT", how="left")
    )
    # ===== OPTIONAL FEATURE COUNTS =====
    def count_feature(df, name):
        if df is None:
            return None
        col = detect_patient_col(df)
        df = df.rename(columns={col: "PATIENT"})
        return df.groupby("PATIENT").size().reset_index(name=name)

    med_feat  = count_feature(medications, "med_count")
    proc_feat = count_feature(procedures, "proc_count")
    all_feat  = count_feature(allergies, "allergy_count")
    imm_feat  = count_feature(immunizations, "immunization_count")
    dev_feat  = count_feature(devices, "device_count")

    # Merge optional features
    for feat in [med_feat, proc_feat, all_feat, imm_feat, dev_feat]:
        if feat is not None:
            merged = merged.merge(feat, on="PATIENT", how="left")

    # Smarter imputation strategy:
    #   - Clinical observation aggregates (_mean, _std): fill with column median of
    #     observed values.  Filling with 0 or -1 injects a false "no measurement = zero
    #     vital-sign" signal that hurts all models, especially the MLP.
    #   - Count features (med_count, proc_count, etc.) and HAS_ flags: fill with 0
    #     because a missing count genuinely means "none recorded".
    clinical_agg_cols = [c for c in merged.columns
                         if c.endswith("_mean") or c.endswith("_std")]
    non_clinical_cols = [c for c in merged.columns if c not in clinical_agg_cols]

    for col in clinical_agg_cols:
        col_median = merged[col].median()
        if pd.isna(col_median):
            col_median = 0.0
        merged[col] = merged[col].fillna(col_median)

    merged[non_clinical_cols] = merged[non_clinical_cols].fillna(0)

    return merged, target_label, code_to_desc, label_names


def temporal_split(df, split_ts = None):
    df = df.copy()

    # Convert date — use tz_convert(None) to strip timezone from tz-aware datetimes.
    # tz_localize(None) raises TypeError when the series is already timezone-aware.
    df["LAST_ENCOUNTER"] = pd.to_datetime(df["LAST_ENCOUNTER"], errors="coerce", utc=True).dt.tz_convert(None)

    # SORT by time
    df = df.sort_values("LAST_ENCOUNTER")

    if not split_ts:
        # FORCE 50-50 split
        mid = len(df) // 2

        ds1 = df.iloc[:mid].copy()
        ds2 = df.iloc[mid:].copy()
    else:
        split_ts = pd.to_datetime(split_ts)
        ds1 = df.loc[df['LAST_ENCOUNTER'] < split_ts, :].copy()
        ds2 = df.loc[df['LAST_ENCOUNTER'] >= split_ts, :].copy()

    return ds1, ds2


def get_feature_matrix(df, feature_cols):
    """Return imputed, numeric feature matrix."""
    X = df[feature_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    return X


def build_models(dt_depth=10, dt_min_s=20, svm_c=1.0, svm_k='rbf',
                 mlp_hidden="64,32", mlp_lr_val=0.001, mlp_iter=300,
                 binary=False):
    hidden = tuple(int(x.strip()) for x in mlp_hidden.split(",") if x.strip().isdigit())
    if not hidden:
        hidden = (64, 32)

    if binary:
        # Plain (non-wrapped) classifiers — standard binary classification.
        # class_weight="balanced" corrects for the TARGET class imbalance directly.
        models = {
            "Decision Tree": DecisionTreeClassifier(
                max_depth=dt_depth, min_samples_split=dt_min_s, random_state=42,
                class_weight="balanced"
            ),
            "SVM": SVC(
                C=svm_c, kernel=svm_k, probability=True, random_state=42,
                class_weight="balanced"
            ),
            "Neural Network": MLPClassifier(
                hidden_layer_sizes=hidden,
                learning_rate_init=mlp_lr_val,
                max_iter=mlp_iter,
                validation_fraction=0.0,
                early_stopping=False
            ),
        }
    else:
        models = {
            # class_weight="balanced" makes each estimator penalise misclassified minority-class
            # samples more heavily, preventing the model from predicting all-zero for rare conditions.
            "Decision Tree": MultiOutputClassifier(DecisionTreeClassifier(
                max_depth=dt_depth, min_samples_split=dt_min_s, random_state=42,
                class_weight="balanced"
            )),
            "SVM": MultiOutputClassifier(SVC(
                C=svm_c, kernel=svm_k, probability=True, random_state=42,
                class_weight="balanced"
            )),
            "Neural Network": MultiOutputClassifier(MLPClassifier(
                hidden_layer_sizes=hidden,
                learning_rate_init=mlp_lr_val,
                max_iter=mlp_iter,
                validation_fraction=0.0,
                early_stopping=False
            )),
        }
    return models

def evaluate_model(model, X_test, y_test, label="", label_names=None):
    """Return dict of metrics + confusion matrix.

    Works for both binary (y_test is 1-D Series/array) and multilabel
    (y_test is a 2-D DataFrame/array with one column per condition).
    """
    y_pred = model.predict(X_test)
    y_test_arr = y_test.values if hasattr(y_test, "values") else np.array(y_test)

    binary_mode = y_test_arr.ndim == 1

    if binary_mode:
        avg = "binary"
        cm  = confusion_matrix(y_test_arr, y_pred)
        mcm = multilabel_confusion_matrix(y_test_arr, y_pred)
        report_names = ["No condition", "Has condition"]
    else:
        # "weighted" averages per-label scores by support — more informative than
        # "macro" when class distributions are skewed across labels.
        avg = "weighted"
        mcm = multilabel_confusion_matrix(y_test_arr, y_pred)
        cm  = mcm[0]   # primary label's 2×2 for display
        if label_names is not None:
            report_names = list(label_names.values()) if isinstance(label_names, dict) else label_names
        else:
            report_names = None

    return {
        "label":     label,
        "accuracy":  accuracy_score(y_test_arr, y_pred),
        "precision": precision_score(y_test_arr, y_pred, average=avg, zero_division=0),
        "recall":    recall_score(y_test_arr, y_pred, average=avg, zero_division=0),
        "f1":        f1_score(y_test_arr, y_pred, average=avg, zero_division=0),
        "cm":        cm,
        "cm_all":    mcm,
        "y_pred":    y_pred,
        "y_test":    y_test_arr,
        "binary_mode": binary_mode,
        "report":    classification_report(
                         y_test_arr, y_pred,
                         target_names=report_names, zero_division=0
                     ),
    }


def plot_roc(model, X_test, y_test, label, ax, color):
    """Plot ROC curve — handles both binary (1-D) and multilabel (2-D) targets.

    Key design decisions:
    - Per-label errors (e.g. only one class present in a small DS2 test split) are
      caught individually so the remaining valid labels still contribute to the
      macro-average curve.  The old bare `except: pass` swallowed errors for the
      WHOLE model, causing the entire DS2 curve to disappear.
    - AUC=nan appeared because np.mean([]) == nan when every label was skipped.
      Now we only plot if at least one label produced a valid curve.
    """
    y_test_arr = y_test.values if hasattr(y_test, "values") else np.array(y_test)
    if not hasattr(model, "predict_proba"):
        return

    try:
        proba = model.predict_proba(X_test)
    except Exception:
        return

    base_fpr = np.linspace(0, 1, 101)

    if y_test_arr.ndim == 1:
        # ── Binary mode: proba is a plain (n, 2) array ───────────────────────
        if isinstance(proba, list):
            proba = proba[0]
        if proba.ndim < 2 or proba.shape[1] < 2:
            return
        # Guard: roc_curve raises ValueError if only one class is present
        if len(np.unique(y_test_arr)) < 2:
            return
        try:
            fpr, tpr, _ = roc_curve(y_test_arr, proba[:, 1], pos_label=1)
            roc_auc = auc(fpr, tpr)
            # Interpolate to a uniform 201-point FPR grid so the curve looks
            # smooth rather than step-wise. Decision Trees and SVMs produce only
            # a small number of distinct probability thresholds, causing a boxy
            # staircase appearance when the raw (fpr, tpr) points are connected.
            # Interpolation preserves the exact AUC while improving readability.
            base_fpr = np.linspace(0, 1, 201)
            tpr_smooth = np.interp(base_fpr, fpr, tpr)
            ax.plot(base_fpr, tpr_smooth, color=color, lw=2,
                    label=f"{label} (AUC={roc_auc:.3f})")
        except Exception:
            return

    elif isinstance(proba, list):
        # ── Multilabel mode: proba is a list of per-label (n, 2) arrays ──────
        tprs, aucs_list = [], []
        for i, p in enumerate(proba):
            # Skip if this label's probability array is malformed
            if not isinstance(p, np.ndarray) or p.ndim < 2 or p.shape[1] < 2:
                continue
            col = y_test_arr[:, i] if y_test_arr.ndim == 2 else y_test_arr
            # Skip labels where only one class is present in this split —
            # this is the root cause of AUC=nan on DS2 test sets
            if len(np.unique(col)) < 2:
                continue
            try:
                fpr_i, tpr_i, _ = roc_curve(col, p[:, 1])
                aucs_list.append(auc(fpr_i, tpr_i))
                tprs.append(np.interp(base_fpr, fpr_i, tpr_i))
            except Exception:
                continue   # skip this label only, not the whole model

        if not tprs:
            # All labels had only one class — draw a greyed-out placeholder so
            # the legend entry still appears and the user knows it was attempted
            ax.plot(base_fpr, base_fpr, color=color, lw=1.5, linestyle=":",
                    label=f"{label} (no valid labels in split)")
            return

        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = float(np.mean(aucs_list))
        ax.plot(base_fpr, mean_tpr, color=color, lw=2,
                label=f"{label} macro-avg (AUC={mean_auc:.3f})")

    else:
        # Fallback for a plain 2-D array from a non-wrapped model
        if proba.ndim < 2 or proba.shape[1] < 2:
            return
        if len(np.unique(y_test_arr)) < 2:
            return
        try:
            fpr, tpr, _ = roc_curve(y_test_arr, proba[:, 1], pos_label=1)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"{label} (AUC={roc_auc:.3f})")
        except Exception:
            return


def plot_confusion(cm, title, ax):
    """Plot a 2×2 confusion matrix.

    For multilabel models, 'cm' is the primary condition's 2×2 sub-matrix
    extracted from multilabel_confusion_matrix(...)[0] in evaluate_model().
    """
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    ax.set_title(title + "\n(primary condition)", fontweight="bold", fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")


def metrics_table(results_list):
    rows = []
    for r in results_list:
        rows.append({
            "Model + Dataset": r["label"],
            "Accuracy":  f"{r['accuracy']:.4f}",
            "Precision": f"{r['precision']:.4f}",
            "Recall":    f"{r['recall']:.4f}",
            "F1-Score":  f"{r['f1']:.4f}",
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📋 Overview",
    "📊 Data & EDA",
    "🔧 Feature Engineering",
    "🤖 Model Training",
    "📈 Evaluation & Comparison",
    "🔄 Continual Learning",
    "🔍 Model Interpretation",
])

tab_overview, tab_eda, tab_feat, tab_train, tab_eval, tab_cl, tab_interp = tabs

# ── TAB 0: OVERVIEW ──────────────────────────────────────────────────────────
with tab_overview:
    # c1, c2, c3 = st.columns(3)
    # c1.info("**Step 1** · Upload 4 CSV files in the sidebar")

    # c2, c3 = st.columns(2)
    # c2.info("**Step 1** · Adjust pipeline settings (split date, hyperparams)")
    # c3.info("**Step 2** · Click **Run Full Pipeline**")

    st.markdown("### 📌 Assignment Overview")
    st.markdown("""
| Task | Description |
|------|-------------|
| **Data Processing** | Merge multi-table EHR data, engineer features via aggregation, encode categorical variables |
| **Temporal Split** | Divide merged data into Dataset 1 (Historical) and Dataset 2 (Current) |
| **EDA** | Descriptive statistics, distribution plots, class balance, data drift analysis |
| **Model Training** | Train Decision Tree, SVM, Neural Network on Dataset 1 train set |
| **Cross-Dataset Eval** | Evaluate Dataset 1 models on both DS1 test and DS2 test sets |
| **Continual Learning** | Fine-tune models on Dataset 2 train set; evaluate on DS2 test |
| **Interpretation** | Feature importance, bias-variance analysis, complexity study |
    """)

    st.markdown("### Target Variable ###")
    st.markdown("- **Multilable**: The target is a 10-tuple of the 10 most common conditions seen in patients.\n"
                "- **Binary**: The target is the existence of any of the 10 most common conditions in patients.\n")

#     st.markdown("### 🏗️ Pipeline Architecture")
#     st.code("""
# EHR Tables (patients, conditions, observations, encounters)
#         │
#         ▼
#   Merge on PATIENT ID
#         │
#         ▼
#   Feature Engineering  ← aggregation, encoding, normalisation
#         │
#         ▼
#   Temporal Split ──────────────────────────────┐
#         │                                       │
#   Dataset 1 (Historical)              Dataset 2 (Current)
#   train │ test                         train │ test
#         │                                       │
#         ▼                                       │
#   Train DT / SVM / MLP                         │
#         │                                       │
#         ├──── Eval on DS1 test ◄────────────────┤
#         └──── Eval on DS2 test ◄────────────────┘
#                                                 │
#                                                 ▼
#                                       Continual Learning
#                                     (fine-tune on DS2 train)
#                                                 │
#                                                 ▼
#                                       Eval on DS2 test
#     """, language="text")


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

if run_pipeline:
    # if not all([patients_file, conditions_file, observations_file, encounters_file]):
    #     st.error("⚠️ Please upload all 4 CSV files before running the pipeline.")
    #     st.stop()

    # ── Load & merge ──────────────────────────────────────────────────────────
    # with st.spinner("Loading and merging EHR tables…"):
    #     patients, conditions, observations, encounters, \
    #     medications, procedures, allergies, immunizations, devices = load_and_merge(
    #         patients_file.read(),
    #         conditions_file.read(),
    #         observations_file.read(),
    #         encounters_file.read(),
    #         medications_file.read() if medications_file else None,
    #         procedures_file.read() if procedures_file else None,
    #         allergies_file.read() if allergies_file else None,
    #         immunizations_file.read() if immunizations_file else None,
    #         devices_file.read() if devices_file else None
    #     )

    with st.spinner("Loading and Merging EHR tables..."):
        allergies, careplans, claims_transactions, claims, conditions, devices, encounters, imaging_studies, immunizations, \
             medications, observations, organizations, patients, payer_transitions, procedures, providers, supplies = load_and_merge()

    # ── Feature engineering ────────────────────────────────────────────────────
    with st.spinner("Engineering features…"):
        merged, target_condition, code_to_desc, label_names = engineer_features(
            patients, conditions, observations, encounters,
            str(split_date),
            medications=medications, procedures=procedures,
            allergies=allergies, immunizations=immunizations, devices=devices
        )

    # ── Temporal split ─────────────────────────────────────────────────────────
    ds1, ds2 = temporal_split(merged, str(split_date))

    # Safety check
    min_samples = 30
    if len(ds1) < min_samples or len(ds2) < min_samples:
        st.warning(
            f"Small dataset after split (DS1={len(ds1)}, DS2={len(ds2)}). "
            "Try adjusting the temporal split date."
        )

    # ── Define feature columns ─────────────────────────────────────────────────
    exclude = {"PATIENT", "TARGET", "LAST_ENCOUNTER"} | \
              {c for c in merged.columns if c.startswith("HAS_")}
    feature_cols = [c for c in merged.columns if c not in exclude]
    target_cols  = [c for c in merged.columns if c.startswith("HAS_")]

    # EDA columns: only interpretable features — age, clinical obs aggregates, and count features.
    # OHE-expanded columns (e.g. gender_F, race_white, ethnicity_nonhispanic) are excluded
    # from EDA because their names are confusing and they add no analytical insight in plots.
    eda_cols = [
        c for c in feature_cols
        if c == "age"
        or c.endswith("_mean")
        or c.endswith("_std")
        or c in ("num_observations", "med_count", "proc_count",
                 "allergy_count", "immunization_count", "device_count")
    ]

    # ── Store in session state ─────────────────────────────────────────────────
    st.session_state["merged"]           = merged
    st.session_state["ds1"]              = ds1
    st.session_state["ds2"]              = ds2
    st.session_state["feature_cols"]     = feature_cols
    st.session_state["eda_cols"]         = eda_cols        # clean interpretable cols for EDA display
    st.session_state["target_cols"]      = target_cols    # list of HAS_ columns (multilabel targets)
    st.session_state["label_names"]      = label_names    # {HAS_col: readable condition name}
    st.session_state["target_condition"] = target_condition
    st.session_state["is_binary"]        = is_binary

    # ── Resolve active target based on prediction mode ─────────────────────────
    # Binary mode uses the pre-built TARGET column (1 = patient has any top-10 condition).
    # Multilabel mode uses all HAS_ columns as a 2-D label matrix.
    if is_binary:
        active_target_cols = ["TARGET"]
    else:
        active_target_cols = target_cols

    st.session_state["active_target_cols"] = active_target_cols

    # ── Prepare train/test splits ─────────────────────────────────────────────
    def make_split(ds, feat_cols, tsize):
        # active_target_cols is either ["TARGET"] (binary) or all HAS_ cols (multilabel).
        # Binary mode supports stratify= because the target is a single 1-D series.
        y = ds[active_target_cols] if not is_binary else ds["TARGET"]
        X = get_feature_matrix(ds, feat_cols)
        if is_binary:
            return train_test_split(X, y, test_size=tsize, random_state=42,
                                    stratify=y)
        return train_test_split(X, y, test_size=tsize, random_state=42)

    X1_tr, X1_te, y1_tr, y1_te = make_split(ds1, feature_cols, test_size)
    X2_tr, X2_te, y2_tr, y2_te = make_split(ds2, feature_cols, test_size)

    # NOTE: The old imbalance check (y1_tr.value_counts().min() < 2) is removed —
    # value_counts() on a multilabel DataFrame raises AttributeError.

    scaler = StandardScaler()
    X1_tr_sc = scaler.fit_transform(X1_tr)
    X1_te_sc = scaler.transform(X1_te)
    X2_tr_sc = scaler.transform(X2_tr)
    X2_te_sc = scaler.transform(X2_te)

    st.session_state.update({
        "X1_tr": X1_tr, "X1_te": X1_te, "y1_tr": y1_tr, "y1_te": y1_te,
        "X2_tr": X2_tr, "X2_te": X2_te, "y2_tr": y2_tr, "y2_te": y2_te,
        "X1_tr_sc": X1_tr_sc, "X1_te_sc": X1_te_sc,
        "X2_tr_sc": X2_tr_sc, "X2_te_sc": X2_te_sc,
        "scaler": scaler,
    })

    # ── Train models on DS1 ───────────────────────────────────────────────────
    mlp_hidden_str = mlp_layers
    models = build_models(dt_max_depth, dt_min_split, svm_C, svm_kernel,
                          mlp_hidden_str, mlp_lr, mlp_epochs,
                          binary=is_binary)

    trained = {}
    with st.spinner("Training models on Dataset 1…"):
        for name, model in models.items():
            if name == "Decision Tree":
                model.fit(X1_tr, y1_tr)
            elif name == "Neural Network":
                # MLPClassifier has no class_weight parameter.
                # sample_weight in fit() requires sklearn >= 1.2 and is often
                # insufficient on its own for highly skewed binary datasets —
                # the optimizer can still converge to "predict all-negative".
                #
                # Robust fix: oversample the minority (positive) class before
                # training.  This is version-independent, requires no extra
                # libraries, and directly presents the MLP with a balanced view
                # of the data on every gradient-descent step.
                if is_binary:
                    y_arr = y1_tr.values if hasattr(y1_tr, "values") else np.array(y1_tr)
                    pos_idx = np.where(y_arr == 1)[0]
                    neg_idx = np.where(y_arr == 0)[0]
                    if len(pos_idx) > 0 and len(neg_idx) > len(pos_idx):
                        rng = np.random.default_rng(42)
                        extra = rng.choice(pos_idx,
                                           size=len(neg_idx) - len(pos_idx),
                                           replace=True)
                        bal_idx = np.concatenate([np.arange(len(y_arr)), extra])
                        rng.shuffle(bal_idx)
                        X_bal = X1_tr_sc[bal_idx]
                        y_bal = y_arr[bal_idx]
                    else:
                        X_bal, y_bal = X1_tr_sc, (y1_tr.values if hasattr(y1_tr, "values")
                                                   else np.array(y1_tr))
                    model.fit(X_bal, y_bal)
                else:
                    # Multilabel: aggregate label to build per-sample weights.
                    # Try sample_weight first; fall back gracefully for older sklearn.
                    agg_y1 = (y1_tr.values.sum(axis=1) > 0).astype(int)
                    sw = compute_sample_weight("balanced", agg_y1)
                    try:
                        model.fit(X1_tr_sc, y1_tr, sample_weight=sw)
                    except TypeError:
                        model.fit(X1_tr_sc, y1_tr)
            else:
                model.fit(X1_tr_sc, y1_tr)
            trained[name] = model

    st.session_state["trained"] = trained

    # ── Evaluate DS1 models ───────────────────────────────────────────────────
    results = {}
    for name, model in trained.items():
        if name == "Decision Tree":
            r1 = evaluate_model(model, X1_te,    y1_te, f"{name} → DS1 test", label_names)
            r2 = evaluate_model(model, X2_te,    y2_te, f"{name} → DS2 test", label_names)
        else:
            r1 = evaluate_model(model, X1_te_sc, y1_te, f"{name} → DS1 test", label_names)
            r2 = evaluate_model(model, X2_te_sc, y2_te, f"{name} → DS2 test", label_names)
        results[name] = {"ds1": r1, "ds2": r2}

    st.session_state["results"] = results

    # ── Continual learning (fine-tune on DS2 train) ───────────────────────────
    cl_models = {}
    with st.spinner("Applying continual learning on Dataset 2…"):
        for name, model in trained.items():
            X_combined = 0
            if name == "Decision Tree":
                X_combined = np.vstack([X1_tr, X2_tr])
                if is_binary:
                    y_combined = pd.concat([y1_tr, y2_tr]).reset_index(drop=True)
                    cl_model = DecisionTreeClassifier(
                        max_depth=dt_max_depth, min_samples_split=dt_min_split,
                        random_state=42, class_weight="balanced"
                    )
                else:
                    # Re-train tree on combined DS1 train + DS2 train.
                    y_combined = pd.concat([y1_tr, y2_tr]).reset_index(drop=True)
                    cl_model = MultiOutputClassifier(DecisionTreeClassifier(
                        max_depth=dt_max_depth, min_samples_split=dt_min_split, random_state=42, class_weight="balanced"
                    ))
                cl_model.fit(X_combined, y_combined)
            elif name == "SVM":
                X_combined_sc = np.vstack([X1_tr_sc, X2_tr_sc])
                y_combined = pd.concat([y1_tr, y2_tr]).reset_index(drop=True)
                if is_binary:
                    cl_model = SVC(
                        C=svm_C, kernel=svm_kernel, probability=True,
                        random_state=42, class_weight="balanced"
                    )
                else:
                    # SVM doesn't support partial_fit; retrain on combined.
                    cl_model = MultiOutputClassifier(SVC(
                        C=svm_C, kernel=svm_kernel, probability=True, random_state=42, class_weight="balanced"
                    ))
                cl_model.fit(X_combined_sc, y_combined)
            else:
                # MLP: warm_start fine-tuning on DS2 train.
                import copy
                cl_model = copy.deepcopy(model)
                if is_binary:
                    # Oversample minority class for balanced fine-tuning (version-safe).
                    y2_arr = y2_tr.values if hasattr(y2_tr, "values") else np.array(y2_tr)
                    pos_idx2 = np.where(y2_arr == 1)[0]
                    neg_idx2 = np.where(y2_arr == 0)[0]
                    if len(pos_idx2) > 0 and len(neg_idx2) > len(pos_idx2):
                        rng2 = np.random.default_rng(42)
                        extra2 = rng2.choice(pos_idx2,
                                             size=len(neg_idx2) - len(pos_idx2),
                                             replace=True)
                        bal_idx2 = np.concatenate([np.arange(len(y2_arr)), extra2])
                        rng2.shuffle(bal_idx2)
                        X_bal2 = X2_tr_sc[bal_idx2]
                        y_bal2 = y2_arr[bal_idx2]
                    else:
                        X_bal2, y_bal2 = X2_tr_sc, y2_arr
                    cl_model.warm_start = True
                    cl_model.max_iter = 100
                    cl_model.fit(X_bal2, y_bal2)
                else:
                    agg_y2 = (y2_tr.values.sum(axis=1) > 0).astype(int)
                    sw2 = compute_sample_weight("balanced", agg_y2)
                    for i in range(len(cl_model.estimators_)):
                        cl_model.estimators_[i].warm_start = True
                        cl_model.estimators_[i].max_iter = 100
                        cl_model.estimators_[i].fit(X2_tr_sc, y2_tr.iloc[:, i], sample_weight=sw2)
            cl_models[name] = cl_model

    cl_results = {}
    for name, cl_model in cl_models.items():
        if name == "Decision Tree":
            r = evaluate_model(cl_model, X2_te, y2_te, f"{name} (CL) → DS2 test", label_names)
        else:
            r = evaluate_model(cl_model, X2_te_sc, y2_te, f"{name} (CL) → DS2 test", label_names)
        cl_results[name] = r

    st.session_state["cl_models"]  = cl_models
    st.session_state["cl_results"] = cl_results

    st.success("✅ Pipeline complete! Navigate the tabs to explore results.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: DATA & EDA
# ─────────────────────────────────────────────────────────────────────────────
with tab_eda:
    if "merged" not in st.session_state:
        st.info("Run the pipeline to see EDA results.")
    else:
        merged   = st.session_state["merged"]
        ds1      = st.session_state["ds1"]
        ds2      = st.session_state["ds2"]
        feat_col = st.session_state["feature_cols"]
        # eda_cols: clinical obs + count features only — OHE indicator columns excluded
        eda_cols = st.session_state["eda_cols"]
        lnames   = st.session_state.get("label_names", {})

        st.markdown('<div class="section-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Patients",      len(merged))
        c2.metric("Dataset 1 (Hist.)",   len(ds1))
        c3.metric("Dataset 2 (Current)", len(ds2))
        c4.metric("Clinical Features",   len(eda_cols))

        # Raw data preview — show only interpretable columns
        with st.expander("📄 Merged Dataset Preview (clinical columns only)"):
            preview_cols = ["PATIENT"] + eda_cols + list(lnames.keys())[:5]
            preview_cols = [c for c in preview_cols if c in merged.columns]
            st.dataframe(merged[preview_cols].head(20), width="stretch")

        st.markdown('<div class="section-header">📈 Descriptive Statistics</div>', unsafe_allow_html=True)
        ds_choice = st.radio("Select dataset:", ["Dataset 1 (Historical)", "Dataset 2 (Current)"],
                             horizontal=True, key="eda_ds")
        cur_ds = ds1 if "1" in ds_choice else ds2
        # Only numeric columns from eda_cols (excludes OHE indicator columns)
        num_cols = [c for c in eda_cols if c in cur_ds.columns and
                    pd.api.types.is_numeric_dtype(cur_ds[c])]

        st.dataframe(cur_ds[num_cols].describe().T.round(3), width="stretch")

        st.markdown('<div class="section-header">📊 Feature Distributions</div>', unsafe_allow_html=True)
        plot_feat = st.selectbox("Select feature to plot:", num_cols, key="eda_feat")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"Distribution Analysis: {plot_feat}", fontweight="bold")

        # Histogram comparing DS1 vs DS2
        axes[0].hist(ds1[plot_feat].dropna(), bins=30, color="#0f3460", alpha=0.7, label="DS1 (Historical)")
        axes[0].hist(ds2[plot_feat].dropna(), bins=30, color="#e94560", alpha=0.5, label="DS2 (Current)")
        axes[0].set_title("Histogram: DS1 vs DS2"); axes[0].legend()
        axes[0].set_xlabel(plot_feat); axes[0].set_ylabel("Count")

        # Boxplot
        box_data = [ds1[plot_feat].dropna(), ds2[plot_feat].dropna()]
        axes[1].boxplot(box_data, labels=["DS1", "DS2"], patch_artist=True,
                        boxprops=dict(facecolor="#0f3460", alpha=0.6))
        axes[1].set_title("Boxplot: DS1 vs DS2"); axes[1].set_ylabel(plot_feat)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # # ── Condition Prevalence (replaces the confusing binary TARGET bar chart) ─────
        # st.markdown('<div class="section-header">📊 Condition Prevalence per Dataset</div>',
        #             unsafe_allow_html=True)
        # st.caption("Percentage of patients in each dataset who have each of the top-10 conditions.")
        # has_cols_present = [c for c in st.session_state.get("target_cols", []) if c in ds1.columns]
        # if has_cols_present:
        #     cond_labels = [lnames.get(c, c)[:30] for c in has_cols_present]
        #     ds1_pct = [ds1[c].mean() * 100 for c in has_cols_present]
        #     ds2_pct = [ds2[c].mean() * 100 for c in has_cols_present]

        #     fig_prev, ax_prev = plt.subplots(figsize=(12, 5))
        #     xp = np.arange(len(cond_labels))
        #     wp = 0.35
        #     ax_prev.bar(xp - wp/2, ds1_pct, wp, label="DS1 (Historical)",
        #                 color="#0f3460", edgecolor="black", alpha=0.85)
        #     ax_prev.bar(xp + wp/2, ds2_pct, wp, label="DS2 (Current)",
        #                 color="#e94560", edgecolor="black", alpha=0.85)
        #     ax_prev.set_xticks(xp)
        #     ax_prev.set_xticklabels(cond_labels, rotation=30, ha="right", fontsize=8)
        #     ax_prev.set_ylabel("% of patients with condition")
        #     ax_prev.set_title("Condition Prevalence — DS1 vs DS2", fontweight="bold")
        #     ax_prev.legend()
        #     plt.tight_layout()
        #     st.pyplot(fig_prev)
        #     plt.close()

        st.markdown('<div class="section-header">🌡️ Correlation Heatmap</div>', unsafe_allow_html=True)
        # Use at most 12 eda_cols + TARGET for the heatmap (keep it readable)
        heatmap_cols = num_cols[:min(12, len(num_cols))]
        has_target   = "TARGET" in cur_ds.columns
        corr_cols    = heatmap_cols + (["TARGET"] if has_target else [])
        corr = cur_ds[corr_cols].corr()
        fig2, ax2 = plt.subplots(figsize=(11, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2,
                    linewidths=0.5, vmin=-1, vmax=1)
        ax2.set_title(f"Correlation Heatmap — {ds_choice}", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # st.markdown('<div class="section-header">🔄 Data Drift Analysis (DS1 vs DS2)</div>',
        #             unsafe_allow_html=True)
        # st.markdown("Mean shift of clinical features between Dataset 1 (Historical) and Dataset 2 (Current):")
        # drift_cols = [c for c in num_cols if c in ds1.columns and c in ds2.columns][:12]
        # drift_df = pd.DataFrame({
        #     "Feature":  drift_cols,
        #     "DS1 Mean": [ds1[c].mean() for c in drift_cols],
        #     "DS2 Mean": [ds2[c].mean() for c in drift_cols],
        # })
        # drift_df["Drift (Δ mean)"] = drift_df["DS2 Mean"] - drift_df["DS1 Mean"]
        # drift_df["Drift %"] = (
        #     drift_df["Drift (Δ mean)"] / (drift_df["DS1 Mean"].abs() + 1e-9) * 100
        # ).round(2)

        # fig3, ax3 = plt.subplots(figsize=(10, 4))
        # colors_d = ["#e94560" if v > 0 else "#0f3460" for v in drift_df["Drift (Δ mean)"]]
        # ax3.barh(drift_df["Feature"], drift_df["Drift (Δ mean)"],
        #          color=colors_d, edgecolor="black", alpha=0.85)
        # ax3.axvline(0, color="black", linewidth=1)
        # ax3.set_title("Feature Mean Drift: DS2 − DS1", fontweight="bold")
        # ax3.set_xlabel("Δ Mean")
        # plt.tight_layout()
        # st.pyplot(fig3)
        # plt.close()

        # st.dataframe(drift_df.round(4), width="stretch")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
with tab_feat:
    if "merged" not in st.session_state:
        st.info("Run the pipeline to see feature engineering results.")
    else:
        merged   = st.session_state["merged"]
        feat_col = st.session_state["feature_cols"]
        tc       = st.session_state["target_condition"]
        lnames   = st.session_state.get("label_names", {})

        st.markdown('<div class="section-header">🔧 Feature Engineering Summary</div>',
                    unsafe_allow_html=True)

        st.markdown(f"""
**Target Variables:** Multi-label — one binary flag per top-10 condition.  
*{tc}*

**Feature Engineering Steps Applied:**

| Step | Method | Output Columns |
|------|--------|----------------|
| Observation Aggregation | Mean & Std of top-10 obs codes per patient | `<obs_name>_mean`, `<obs_name>_std` |
| Observation Count | Total observations per patient | `num_observations` |
| Age Extraction | Days since birthdate ÷ 365.25 | `age` |
| Gender Encoding | One-Hot (M/F/unknown) | `gender_*` |
| Race Encoding | One-Hot | `race_*` |
| Ethnicity Encoding | One-Hot | `ethnicity_*` |
| Medication Count | Prescriptions per patient | `med_count` |
| Procedure Count | Procedures per patient | `proc_count` |
| Allergy Count | Allergies per patient | `allergy_count` |
| Condition Flags | Binary presence flag per top-10 condition | `HAS_<code>` (targets, not features) |
        """)

        st.markdown('<div class="section-header">📋 Feature Columns Used for Modeling</div>',
                    unsafe_allow_html=True)
        st.caption("Long observation names are truncated to 35 characters for readability. "
                   "Full names are used internally.")
        feat_df = pd.DataFrame({
            "Feature": [
                (c[:35] + "…") if len(c) > 35 else c
                for c in feat_col
            ],
            # "Full Column Name": feat_col,
            "Type": [str(merged[c].dtype) for c in feat_col],
            # "Missing %": [(merged[c].isna().mean() * 100).round(1) for c in feat_col],
            "DS1 Mean": [
                round(float(st.session_state["ds1"][c].mean()), 3)
                if c in st.session_state["ds1"].columns else np.nan
                for c in feat_col
            ],
            "DS2 Mean": [
                round(float(st.session_state["ds2"][c].mean()), 3)
                if c in st.session_state["ds2"].columns else np.nan
                for c in feat_col
            ],
        })
        st.dataframe(feat_df, width="stretch")

        # st.markdown('<div class="section-header">📊 Missing Value Analysis</div>', unsafe_allow_html=True)
        # miss = feat_df[feat_df["Missing %"] > 0].sort_values("Missing %", ascending=False)
        # if len(miss):
        #     fig, ax = plt.subplots(figsize=(10, max(3, len(miss) * 0.4)))
        #     ax.barh(miss["Feature (display)"], miss["Missing %"],
        #             color="#0f3460", edgecolor="black", alpha=0.85)
        #     ax.set_title("Features with Missing Values (% missing before imputation)",
        #                  fontweight="bold")
        #     ax.set_xlabel("Missing %")
        #     plt.tight_layout()
        #     st.pyplot(fig); plt.close()
        # else:
        #     st.success("✅ No missing values after median imputation.")

        # ── Condition Prevalence (replaces the binary class-balance pie charts) ──────
        # The PDF asks for "class distribution analysis". For multilabel classification,
        # the meaningful equivalent is the per-condition positive rate in each dataset.
        st.markdown('<div class="section-header">📊 Condition Distribution across Datasets</div>',
                    unsafe_allow_html=True)
        st.caption(
            "Shows what percentage of patients in DS1 vs DS2 have each of the 10 target conditions. "
            "This is the multilabel equivalent of class distribution analysis (as required by the assignment)."
        )
        target_cols_present = [c for c in st.session_state.get("target_cols", [])
                                if c in merged.columns]
        if target_cols_present:
            cond_display = [lnames.get(c, c)[:35] for c in target_cols_present]
            ds1_rate = [st.session_state["ds1"][c].mean() * 100 for c in target_cols_present]
            ds2_rate = [st.session_state["ds2"][c].mean() * 100 for c in target_cols_present]

            prev_df = pd.DataFrame({
                "Condition":      cond_display,
                "DS1 Prevalence %": [round(v, 1) for v in ds1_rate],
                "DS2 Prevalence %": [round(v, 1) for v in ds2_rate],
            })
            prev_df["Δ (DS2−DS1) %"] = (
                prev_df["DS2 Prevalence %"] - prev_df["DS1 Prevalence %"]
            ).round(1)
            st.dataframe(prev_df, width="stretch")

            fig_p, ax_p = plt.subplots(figsize=(12, 5))
            xp = np.arange(len(cond_display))
            wp = 0.35
            ax_p.bar(xp - wp/2, ds1_rate, wp, label="DS1 (Historical)",
                     color="#0f3460", edgecolor="black", alpha=0.85)
            ax_p.bar(xp + wp/2, ds2_rate, wp, label="DS2 (Current)",
                     color="#e94560", edgecolor="black", alpha=0.85)
            ax_p.set_xticks(xp)
            ax_p.set_xticklabels(cond_display, rotation=35, ha="right", fontsize=8)
            ax_p.set_ylabel("% of patients")
            ax_p.set_title("Condition Prevalence: DS1 vs DS2", fontweight="bold")
            ax_p.legend()
            plt.tight_layout()
            st.pyplot(fig_p); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
with tab_train:
    if "trained" not in st.session_state:
        st.info("Run the pipeline to see training results.")
    else:
        trained  = st.session_state["trained"]
        X1_tr    = st.session_state["X1_tr"]
        y1_tr    = st.session_state["y1_tr"]
        X1_te    = st.session_state["X1_te"]
        X1_te_sc = st.session_state["X1_te_sc"]
        y1_te    = st.session_state["y1_te"]

        st.markdown('<div class="section-header">🤖 Training Summary — Dataset 1</div>',
                    unsafe_allow_html=True)

        is_binary   = st.session_state.get("is_binary", False)
        mode_badge  = "🔵 Binary mode" if is_binary else "🟣 Multilabel mode"
        st.info(f"**Prediction Mode:** {mode_badge} — change in the sidebar and re-run to compare.")

        c1, c2, c3 = st.columns(3)
        for col, (name, model) in zip([c1, c2, c3], trained.items()):
            if name == "Decision Tree":
                tr_acc = accuracy_score(y1_tr, model.predict(X1_tr))
                te_acc = accuracy_score(y1_te, model.predict(X1_te))
            else:
                tr_acc = accuracy_score(y1_tr, model.predict(st.session_state["X1_tr_sc"]))
                te_acc = accuracy_score(y1_te, model.predict(X1_te_sc))
            col.metric(f"🎯 {name}", f"Train: {tr_acc:.3f}", f"Test: {te_acc:.3f}")

        st.markdown('<div class="section-header">📊 Training vs Test Accuracy (Bias-Variance)</div>',
                    unsafe_allow_html=True)

        model_names = list(trained.keys())
        train_accs, test_accs = [], []
        for name, model in trained.items():
            if name == "Decision Tree":
                train_accs.append(accuracy_score(y1_tr, model.predict(X1_tr)))
                test_accs.append(accuracy_score(y1_te, model.predict(X1_te)))
            else:
                train_accs.append(accuracy_score(y1_tr, model.predict(st.session_state["X1_tr_sc"])))
                test_accs.append(accuracy_score(y1_te, model.predict(X1_te_sc)))

        x = np.arange(len(model_names))
        fig, ax = plt.subplots(figsize=(9, 4))
        w = 0.35
        b1 = ax.bar(x - w/2, train_accs, w, label="Train Accuracy",
                    color="#0f3460", edgecolor="black", alpha=0.85)
        b2 = ax.bar(x + w/2, test_accs,  w, label="Test Accuracy",
                    color="#e94560", edgecolor="black", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(model_names)
        ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.15)
        ax.set_title("Train vs Test Accuracy — Dataset 1 Models", fontweight="bold")
        ax.legend()
        for bar, v in zip(list(b1) + list(b2), train_accs + test_accs):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                    f"{v:.3f}", ha="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # st.markdown('<div class="section-header">📄 Classification Report</div>',
        #             unsafe_allow_html=True)
        # lnames      = st.session_state.get("label_names", {})
        # target_cols = st.session_state.get("target_cols", [])
        # is_binary   = st.session_state.get("is_binary", False)

        # if is_binary:
        #     st.caption("Binary mode: single output — does the patient have any top-10 condition?")
        #     for name, model in trained.items():
        #         with st.expander(f"📋 {name} — Binary Classification Report (DS1 test)"):
        #             X_pred = X1_te if name == "Decision Tree" else X1_te_sc
        #             y_pred_arr = model.predict(X_pred)
        #             y_true_arr = y1_te.values if hasattr(y1_te, "values") else np.array(y1_te)
        #             st.text(classification_report(
        #                 y_true_arr, y_pred_arr,
        #                 target_names=["No condition", "Has condition"],
        #                 zero_division=0
        #             ))
        # else:
        #     st.caption(
        #         "For multilabel classification, sklearn's classification_report produces one row per "
        #         "unique label-combination, which is hard to read. Instead, we compute binary "
        #         "Precision / Recall / F1 for each condition individually."
        #     )
        #     for name, model in trained.items():
        #         with st.expander(f"📋 {name} — Per-Condition Metrics (DS1 test)"):
        #             X_pred = X1_te if name == "Decision Tree" else X1_te_sc
        #             y_pred_arr = model.predict(X_pred)
        #             y_true_arr = y1_te.values if hasattr(y1_te, "values") else np.array(y1_te)
        #             rows_report = []
        #             for j, col_key in enumerate(target_cols):
        #                 cond_name = lnames.get(col_key, col_key)[:40]
        #                 p = precision_score(y_true_arr[:, j], y_pred_arr[:, j], zero_division=0)
        #                 r = recall_score(y_true_arr[:, j], y_pred_arr[:, j], zero_division=0)
        #                 f = f1_score(y_true_arr[:, j], y_pred_arr[:, j], zero_division=0)
        #                 support = int(y_true_arr[:, j].sum())
        #                 rows_report.append({
        #                     "Condition":        cond_name,
        #                     "Precision":        round(p, 3),
        #                     "Recall":           round(r, 3),
        #                     "F1-Score":         round(f, 3),
        #                     "# True Positives": support,
        #                 })
        #             st.dataframe(pd.DataFrame(rows_report), width="stretch")

        st.markdown('<div class="section-header">📐 Decision Tree Complexity Analysis</div>',
                    unsafe_allow_html=True)
        dt_model   = trained["Decision Tree"]
        is_binary  = st.session_state.get("is_binary", False)
        # In binary mode the DT is a plain DecisionTreeClassifier (no .estimators_ wrapper).
        dt_sub = dt_model if is_binary else dt_model.estimators_[0]
        sub_label = "tree" if is_binary else "primary condition tree"
        st.markdown(f"""
| Property | Value |
|---|---|
| Max Depth (configured) | {dt_max_depth} |
| Actual Depth ({sub_label}) | {dt_sub.get_depth()} |
| Number of Leaves ({sub_label}) | {dt_sub.get_n_leaves()} |
| Number of Features Used | {dt_sub.n_features_in_} |
        """)

        if st.checkbox("Show Decision Tree (first 3 levels)", key="show_tree"):
            feat_col = st.session_state["feature_cols"]
            # plot_tree requires a plain DecisionTreeClassifier, not MultiOutputClassifier.
            # Use estimators_[0] — the tree trained for the primary (most common) condition.
            primary_cond = list(st.session_state.get("label_names", {}).values() or ["primary"])[0]
            fig_t, ax_t = plt.subplots(figsize=(18, 8))
            plot_tree(dt_sub, max_depth=3, filled=True, feature_names=feat_col,
                      class_names=["No", "Yes"], ax=ax_t, fontsize=7)
            ax_t.set_title(
                f"Decision Tree — Primary Condition: {primary_cond} (max depth = 3 shown)",
                fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_t); plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: EVALUATION & COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
with tab_eval:
    if "results" not in st.session_state:
        st.info("Run the pipeline to see evaluation results.")
    else:
        results  = st.session_state["results"]
        trained  = st.session_state["trained"]

        st.markdown('<div class="section-header">📊 Full Metrics Table: DS1 Models on DS1 & DS2 Test</div>',
                    unsafe_allow_html=True)

        is_binary  = st.session_state.get("is_binary", False)
        mode_badge = "🔵 Binary mode" if is_binary else "🟣 Multilabel mode"
        st.info(f"**Prediction Mode:** {mode_badge} — use the sidebar toggle to switch and re-run.")

        all_results = []
        for name in results:
            all_results.append(results[name]["ds1"])
            all_results.append(results[name]["ds2"])
        st.dataframe(metrics_table(all_results), width="stretch")

        st.markdown('<div class="section-header">📊 Performance Comparison Bar Charts</div>',
                    unsafe_allow_html=True)
        metrics_keys = ["accuracy", "precision", "recall", "f1"]
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle("DS1-Trained Models: DS1 Test vs DS2 Test Performance", fontweight="bold", fontsize=13)

        model_names = list(results.keys())
        x = np.arange(len(model_names))
        w = 0.35
        colors = ["#0f3460", "#e94560"]

        for i, metric in enumerate(metrics_keys):
            ds1_vals = [results[n]["ds1"][metric] for n in model_names]
            ds2_vals = [results[n]["ds2"][metric] for n in model_names]
            b1 = axes[i].bar(x - w/2, ds1_vals, w, label="DS1 Test",
                             color=colors[0], edgecolor="black", alpha=0.85)
            b2 = axes[i].bar(x + w/2, ds2_vals, w, label="DS2 Test",
                             color=colors[1], edgecolor="black", alpha=0.85)
            axes[i].set_xticks(x); axes[i].set_xticklabels(model_names, rotation=15, ha="right")
            axes[i].set_title(metric.capitalize(), fontweight="bold")
            axes[i].set_ylim(0, 1.15)
            axes[i].legend(fontsize=8)
            for bar, v in zip(list(b1)+list(b2), ds1_vals+ds2_vals):
                axes[i].text(bar.get_x()+bar.get_width()/2, v+0.02,
                             f"{v:.2f}", ha="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown('<div class="section-header">📉 Confusion Matrices</div>', unsafe_allow_html=True)
        model_sel = st.selectbox("Select Model:", model_names, key="eval_model")
        fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
        fig2.suptitle(f"{model_sel} — Confusion Matrices", fontweight="bold")
        plot_confusion(results[model_sel]["ds1"]["cm"], "DS1 Test Set", axes2[0])
        plot_confusion(results[model_sel]["ds2"]["cm"], "DS2 Test Set", axes2[1])
        plt.tight_layout()
        st.pyplot(fig2); plt.close()

        st.markdown('<div class="section-header">📈 ROC Curves</div>', unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        ax3.plot([0,1],[0,1],'k--', linewidth=1, label="Random")
        # 6 entries: DS1 and DS2 for each of the 3 models — one colour per entry.
        colors_roc = ["#0f3460", "#5b8db8",   # DT  DS1 (dark blue), DS2 (light blue)
                      "#c0392b", "#e94560",    # SVM DS1 (dark red),  DS2 (pink)
                      "#27ae60", "#2ecc71"]    # MLP DS1 (dark green), DS2 (light green)
        c_idx = 0
        for name, model in trained.items():
            if name == "Decision Tree":
                plot_roc(model, st.session_state["X1_te"],
                         st.session_state["y1_te"], f"{name} (DS1)", ax3, colors_roc[c_idx])
                plot_roc(model, st.session_state["X2_te"],
                         st.session_state["y2_te"], f"{name} (DS2)", ax3, colors_roc[c_idx + 1])
            else:
                plot_roc(model, st.session_state["X1_te_sc"],
                         st.session_state["y1_te"], f"{name} (DS1)", ax3, colors_roc[c_idx])
                plot_roc(model, st.session_state["X2_te_sc"],
                         st.session_state["y2_te"], f"{name} (DS2)", ax3, colors_roc[c_idx + 1])
            c_idx = min(c_idx + 2, len(colors_roc) - 2)
        ax3.set_xlabel("False Positive Rate"); ax3.set_ylabel("True Positive Rate")
        ax3.set_title("ROC Curves — All Models on DS1 & DS2", fontweight="bold")
        ax3.legend(fontsize=8, loc="lower right")
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3); plt.close()

        st.markdown('<div class="section-header">📉 F1 Drop: DS1→DS2 (Temporal Shift Impact)</div>',
                    unsafe_allow_html=True)
        drop_data = {
            "Model": model_names,
            "F1 on DS1 Test": [results[n]["ds1"]["f1"] for n in model_names],
            "F1 on DS2 Test": [results[n]["ds2"]["f1"] for n in model_names],
        }
        drop_df = pd.DataFrame(drop_data)
        drop_df["F1 Drop"] = (drop_df["F1 on DS1 Test"] - drop_df["F1 on DS2 Test"]).round(4)
        drop_df["Drop %"]  = (drop_df["F1 Drop"] / (drop_df["F1 on DS1 Test"] + 1e-9) * 100).round(2)
        st.dataframe(drop_df.round(4), width="stretch")

        # ── Task 3f: Feature Representation Comparison note ───────────────────
        st.markdown('<div class="section-header">🔀 Feature Representation Comparison</div>',
                    unsafe_allow_html=True)
        if is_binary:
            st.success(
                "**Currently in Binary mode.** Switch to **Multilabel** in the sidebar and "
                "re-run the pipeline to generate results for both modes. Comparing the metrics "
                "tables side-by-side directly addresses Assignment Task 3f: *'Analyze how "
                "different feature representations affect model performance.'*\n\n"
                "**Expected findings:**\n"
                "- Binary mode tends to produce higher Precision, Recall, and F1 because the "
                "target is a single, relatively balanced label.\n"
                "- Multilabel mode captures richer per-condition structure but suffers from "
                "extreme class imbalance across labels, pulling weighted averages down.\n"
                "- The Neural Network benefits most from switching to binary, since it can no "
                "longer default to the all-zeros prediction strategy in the absence of "
                "`class_weight`."
            )
        else:
            st.success(
                # "**Currently in Multilabel mode.** Switch to **Binary** in the sidebar and "
                # "re-run the pipeline to generate results for both modes. Comparing the metrics "
                # "tables side-by-side directly addresses Assignment Task 3f: *'Analyze how "
                # "different feature representations affect model performance.'*\n\n"
                "**Expected findings:**\n"
                "- Binary mode tends to produce higher Precision, Recall, and F1 because the "
                "target is a single, relatively balanced label.\n"
                "- Multilabel mode captures richer per-condition structure but is harder to "
                "learn — especially for rare conditions with very few positive samples.\n"
                "- The accuracy gap between modes is often misleading: high multilabel accuracy "
                "is achieved by predicting all-zeros, while binary accuracy reflects genuine "
                "discriminative ability."
            )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: CONTINUAL LEARNING
# ─────────────────────────────────────────────────────────────────────────────
with tab_cl:
    if "cl_results" not in st.session_state:
        st.info("Run the pipeline to see continual learning results.")
    else:
        results    = st.session_state["results"]
        cl_results = st.session_state["cl_results"]
        model_names = list(results.keys())

        st.markdown('<div class="section-header">🔄 Continual Learning Strategy</div>',
                    unsafe_allow_html=True)
        is_binary  = st.session_state.get("is_binary", False)
        mode_badge = "🔵 Binary mode" if is_binary else "🟣 Multilabel mode"
        st.info(f"**Prediction Mode:** {mode_badge} — use the sidebar toggle to switch and re-run.")
        st.markdown("""
| Model | Strategy |
|---|---|
| **Decision Tree** | Retrain on combined DS1 train + DS2 train (accumulated learning) |
| **SVM** | Retrain on combined DS1 train + DS2 train (SVM does not support partial_fit) |
| **Neural Network** | Fine-tune DS1-trained MLP on DS2 train using `warm_start=True` (weights preserved) |

**Rationale:** The goal is to update each model so it adapts to the distribution shift in DS2
while retaining knowledge from DS1. The MLP warm-start approach most closely resembles
transfer learning / continual learning — weights from DS1 training are the starting point
for DS2 fine-tuning.
        """)

        st.markdown('<div class="section-header">📊 Before vs After Continual Learning on DS2 Test</div>',
                    unsafe_allow_html=True)

        rows = []
        for name in model_names:
            before = results[name]["ds2"]
            after  = cl_results[name]
            rows.append({
                "Model":               name,
                "F1 Before CL (DS2)":  f"{before['f1']:.4f}",
                "F1 After CL (DS2)":   f"{after['f1']:.4f}",
                "Δ F1":                f"{after['f1'] - before['f1']:+.4f}",
                "Acc Before":          f"{before['accuracy']:.4f}",
                "Acc After":           f"{after['accuracy']:.4f}",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch")

        st.markdown('<div class="section-header">📊 F1 Score Comparison Chart</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(model_names))
        w = 0.25

        f1_ds1  = [results[n]["ds1"]["f1"] for n in model_names]
        f1_ds2  = [results[n]["ds2"]["f1"] for n in model_names]
        f1_cl   = [cl_results[n]["f1"]     for n in model_names]

        b1 = ax.bar(x - w, f1_ds1, w, label="DS1 Test (baseline)",
                    color="#0f3460", edgecolor="black", alpha=0.85)
        b2 = ax.bar(x,     f1_ds2, w, label="DS2 Test (before CL)",
                    color="#e94560", edgecolor="black", alpha=0.85)
        b3 = ax.bar(x + w, f1_cl,  w, label="DS2 Test (after CL)",
                    color="#2ecc71", edgecolor="black", alpha=0.85)

        ax.set_xticks(x); ax.set_xticklabels(model_names)
        ax.set_ylabel("F1 Score"); ax.set_ylim(0, 1.2)
        ax.set_title("F1 Score: DS1 Baseline vs Before CL vs After CL", fontweight="bold")
        ax.legend()
        for bars in [b1, b2, b3]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x()+bar.get_width()/2, h+0.02,
                        f"{h:.2f}", ha="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown('<div class="section-header">📉 Confusion Matrices — After Continual Learning</div>',
                    unsafe_allow_html=True)
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
        fig2.suptitle("Confusion Matrices — Continually Learned Models on DS2 Test", fontweight="bold")
        for ax2, name in zip(axes2, model_names):
            plot_confusion(cl_results[name]["cm"], name, ax2)
        plt.tight_layout()
        st.pyplot(fig2); plt.close()

        st.markdown('<div class="section-header">💬 Analysis</div>', unsafe_allow_html=True)
        st.markdown("""
**Key observations on continual learning effectiveness:**

1. **Neural Network (fine-tuning)** typically shows the most improvement on DS2 test after CL,
   as warm-start fine-tuning allows gradient updates targeting the new distribution while
   preserving earlier learned representations.

2. **Decision Tree (combined retraining)** may show a different depth/structure after CL because
   the combined dataset has a richer feature distribution, slightly
   reducing DS2 performance depending on class balance.

3. **SVM (combined retraining)** also shows lower performance with an increase in more diversified patient portfolios
    making it difficult to distinguish true patients.

4. **Data drift impact:** If the F1 drop from DS1→DS2 (before CL) is large, it confirms
   meaningful temporal shift in the patient population. The CL step aims to recover that gap.
        """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6: MODEL INTERPRETATION
# ─────────────────────────────────────────────────────────────────────────────
with tab_interp:
    if "trained" not in st.session_state:
        st.info("Run the pipeline to see model interpretation.")
    else:
        trained   = st.session_state["trained"]
        feat_col  = st.session_state["feature_cols"]
        X1_te     = st.session_state["X1_te"]
        X1_te_sc  = st.session_state["X1_te_sc"]
        y1_te     = st.session_state["y1_te"]

        st.markdown('<div class="section-header">🌳 Decision Tree — Feature Importance</div>',
                    unsafe_allow_html=True)
        dt_model  = trained["Decision Tree"]
        is_binary = st.session_state.get("is_binary", False)

        # In binary mode the DT is a plain DecisionTreeClassifier — use feature_importances_
        # directly.  In multilabel mode, average Gini importances across all per-label trees.
        if is_binary:
            importances = dt_model.feature_importances_
            imp_title   = "Decision Tree — Feature Importance (Gini, binary target)"
        else:
            importances = np.mean(
                [e.feature_importances_ for e in dt_model.estimators_], axis=0
            )
            imp_title = "Decision Tree — Avg Feature Importance across all conditions (Gini)"
        # Truncate long feature names for the chart y-axis
        short_names = [(c[:30] + "…") if len(c) > 30 else c for c in feat_col]
        feat_imp_df = pd.DataFrame({
            "Feature":      short_names,
            "Full Name":    feat_col,
            "Importance":   importances
        }).sort_values("Importance", ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(10, max(5, len(feat_imp_df) * 0.4)))
        ax.barh(feat_imp_df["Feature"][::-1], feat_imp_df["Importance"][::-1],
                color="#0f3460", edgecolor="black", alpha=0.85)
        ax.set_title(imp_title, fontweight="bold")
        ax.set_xlabel("Mean Importance Score")
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        # Also show as a table so full names are visible
        st.dataframe(
            feat_imp_df[["Full Name", "Importance"]].rename(
                columns={"Full Name": "Feature", "Importance": "Gini Importance"}
            ).reset_index(drop=True),
            width="stretch"
        )

        # In binary mode use plain f1 (binary average); in multilabel use weighted average.
        # The scorer must match the target shape to avoid sklearn errors.
        if is_binary:
            f1w_scorer = make_scorer(f1_score, average="binary", zero_division=0)
        else:
            f1w_scorer = make_scorer(f1_score, average="weighted", zero_division=0)

        st.markdown('<div class="section-header">🔲 SVM — Permutation Feature Importance</div>',
                    unsafe_allow_html=True)
        svm_model = trained["SVM"]
        with st.spinner("Computing permutation importance for SVM (using F1-weighted scorer)…"):
            try:
                perm = permutation_importance(
                    svm_model, X1_te_sc, y1_te,
                    n_repeats=5, random_state=42, scoring=f1w_scorer
                )
                perm_df = pd.DataFrame({
                    "Feature":   short_names,
                    "Full Name": feat_col,
                    "Importance": perm.importances_mean
                }).sort_values("Importance", ascending=False).head(15)

                fig2, ax2 = plt.subplots(figsize=(10, max(5, len(perm_df) * 0.4)))
                ax2.barh(perm_df["Feature"][::-1], perm_df["Importance"][::-1],
                         color="#e94560", edgecolor="black", alpha=0.85)
                ax2.set_title("SVM — Permutation Feature Importance (F1-weighted drop)",
                              fontweight="bold")
                ax2.set_xlabel("Mean F1-Weighted Drop when feature is permuted")
                plt.tight_layout()
                st.pyplot(fig2); plt.close()
                st.dataframe(
                    perm_df[["Full Name", "Importance"]].rename(
                        columns={"Full Name": "Feature", "Importance": "F1 Drop"}
                    ).reset_index(drop=True),
                    width="stretch"
                )
            except Exception as e:
                st.warning(f"Permutation importance unavailable: {e}")

        st.markdown('<div class="section-header">🧠 Neural Network — Feature Importance</div>',
                    unsafe_allow_html=True)
        mlp_model = trained["Neural Network"]
        with st.spinner("Computing permutation importance for Neural Network (using F1-weighted scorer)…"):
            try:
                perm_mlp = permutation_importance(
                    mlp_model, X1_te_sc, y1_te,
                    n_repeats=5, random_state=42, scoring=f1w_scorer
                )
                mlp_imp_df = pd.DataFrame({
                    "Feature":   short_names,
                    "Full Name": feat_col,
                    "Importance": perm_mlp.importances_mean
                }).sort_values("Importance", ascending=False).head(15)

                fig3, ax3 = plt.subplots(figsize=(10, max(5, len(mlp_imp_df) * 0.4)))
                ax3.barh(mlp_imp_df["Feature"][::-1], mlp_imp_df["Importance"][::-1],
                         color="#2ecc71", edgecolor="black", alpha=0.85)
                ax3.set_title("Neural Network — Permutation Feature Importance (F1-weighted drop)",
                              fontweight="bold")
                ax3.set_xlabel("Mean F1-Weighted Drop when feature is permuted")
                plt.tight_layout()
                st.pyplot(fig3); plt.close()
                st.dataframe(
                    mlp_imp_df[["Full Name", "Importance"]].rename(
                        columns={"Full Name": "Feature", "Importance": "F1 Drop"}
                    ).reset_index(drop=True),
                    width="stretch"
                )
            except Exception as e:
                st.warning(f"Permutation importance unavailable: {e}")

        st.markdown('<div class="section-header">📊 Bias-Variance & Model Complexity Summary</div>',
                    unsafe_allow_html=True)
        st.markdown("""
| Model | Complexity | Bias | Variance | Interpretability | Notes |
|---|---|---|---|---|---|
| **Decision Tree** | Tunable via depth | Low (deep) / High (shallow) | High (deep) / Low (shallow) | ✅ Very high | Prone to overfitting at large depth; feature importance directly available |
| **SVM** | Via C and kernel | Moderate | Moderate | ❌ Low (black-box) | RBF kernel adds non-linearity; C controls regularization strength |
| **Neural Network** | Via layers & neurons | Low | High (large net) | ❌ Low | Most flexible; benefits most from continual learning; requires scaling |

**Bias-Variance Trade-off Observations:**
- A very deep Decision Tree will have near-zero training error (low bias) but high test error (high variance — overfitting).
- A shallow Decision Tree has high bias (underfitting) but low variance.
- SVM with small C has high bias (large margin, many misclassifications); large C has low bias but can overfit.
- MLP with more layers/neurons decreases bias but increases variance — regularization (early stopping, dropout) is needed.
        """)

        st.markdown('<div class="section-header">🌲 Decision Tree Depth vs Test Accuracy</div>',
                    unsafe_allow_html=True)
        X1_tr = st.session_state["X1_tr"]
        y1_tr = st.session_state["y1_tr"]
        depths = list(range(1, min(16, dt_max_depth + 6)))
        tr_accs, te_accs = [], []
        for d in depths:
            if is_binary:
                tmp = DecisionTreeClassifier(max_depth=d, random_state=42,
                                             class_weight="balanced")
            else:
                # Must wrap in MultiOutputClassifier because y1_tr is a multilabel DataFrame.
                # Accuracy here is subset accuracy (all labels correct).
                tmp = MultiOutputClassifier(DecisionTreeClassifier(max_depth=d, random_state=42))
            tmp.fit(X1_tr, y1_tr)
            tr_accs.append(accuracy_score(y1_tr, tmp.predict(X1_tr)))
            te_accs.append(accuracy_score(y1_te, tmp.predict(X1_te)))

        fig4, ax4 = plt.subplots(figsize=(9, 4))
        ax4.plot(depths, tr_accs, 'o-', color="#0f3460", linewidth=2, label="Train Accuracy")
        ax4.plot(depths, te_accs, 's-', color="#e94560", linewidth=2, label="Test Accuracy")
        ax4.axvline(dt_max_depth, color="gray", linestyle="--", linewidth=1.2,
                    label=f"Configured depth={dt_max_depth}")
        ax4.set_xlabel("Max Depth"); ax4.set_ylabel("Accuracy")
        ax4.set_title("Decision Tree: Depth vs Train/Test Accuracy (Bias-Variance Curve)",
                      fontweight="bold")
        ax4.legend(); ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig4); plt.close()

        st.caption("""
When Train Accuracy >> Test Accuracy at high depth → **Overfitting (high variance)**.  
When both are low at shallow depth → **Underfitting (high bias)**.  
The optimal depth minimizes the Test Accuracy gap.
        """)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#666; font-size:0.85rem;'>"
    "BITS F464 Machine Learning · Assignment 2 · Team 30 · "
    "Sailesh Nichenametla · Rohan Harshith Amarthaluri · Dhanush Thirunagari · Aniket Shukla"
    "</p>",
    unsafe_allow_html=True
)
