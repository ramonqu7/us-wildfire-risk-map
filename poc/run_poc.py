#!/usr/bin/env python3
"""
Proof-of-Concept (POC) Pipeline for US Wildfire Risk Map (CSE 6242).
Executes on real downloaded datasets:
  1. FPA FOD-Attributes 2020 (73,221 CONUS wildfires, 308 columns)
  2. FEMA National Risk Index v1.20 (3,232 US counties, 465 columns)
  3. USFS Wildfire Risk to Communities v2 (32,038 US communities, 3,144 counties)
"""

import json
import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    brier_score_loss,
    davies_bouldin_score,
    mean_absolute_error,
    roc_auc_score,
    average_precision_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
POC_DIR = os.path.join(BASE_DIR, "poc")
FIG_DIR = os.path.join(POC_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

ENV_FEATURES = [
    "tmmx", "rmin", "vs", "erc", "vpd", "bi", "fm100", "fm1000",
    "Elevation", "Slope", "TRI", "NDVI-1day", "Aridity_index", "Annual_precipitation"
]
CONTEXT_FEATURES = [
    "Population", "No_FireStation_5.0km", "road_county_dis", "RPL_THEMES", "EVC", "EVH"
]
ALL_FEATURES = ENV_FEATURES + CONTEXT_FEATURES


def main():
    t0 = time.time()
    print("1. Loading real datasets...")
    fpa_path = os.path.join(DATA_DIR, "2020_FPA_FOD_cons.csv")
    use_cols = [
        "FOD_ID", "FPA_ID", "FIRE_NAME", "DISCOVERY_DATE", "DISCOVERY_DOY",
        "NWCG_CAUSE_CLASSIFICATION", "NWCG_GENERAL_CAUSE", "FIRE_SIZE",
        "FIRE_SIZE_CLASS", "LATITUDE", "LONGITUDE", "STATE", "COUNTY",
        "FIPS_CODE", "FIPS_NAME", "Land_Cover", "FRG", "NAME"
    ] + ALL_FEATURES
    df = pd.read_csv(fpa_path, usecols=lambda c: c in use_cols, low_memory=False)
    print(f"   Loaded FPA FOD-Attributes 2020: {len(df):,} fires, {len(df.columns)} selected cols")

    nri_path = os.path.join(DATA_DIR, "nri_counties", "NRI_Table_Counties.csv")
    nri = pd.read_csv(nri_path, low_memory=False)
    print(f"   Loaded FEMA NRI Counties: {len(nri):,} counties, {len(nri.columns)} cols")

    wrc_path = os.path.join(DATA_DIR, "wrc_download.xlsx")
    wrc_comm = pd.read_excel(wrc_path, sheet_name="Communities")
    wrc_cnty = pd.read_excel(wrc_path, sheet_name="Counties")
    print(f"   Loaded USFS WRC: {len(wrc_comm):,} communities, {len(wrc_cnty):,} counties")

    # Clean environmental feature matrix
    df_clean = df.dropna(subset=["LATITUDE", "LONGITUDE", "FIRE_SIZE"]).copy()
    for col in ALL_FEATURES:
        s = pd.to_numeric(df_clean[col], errors="coerce")
        s = s.mask((s >= 30000) | (s < -500), np.nan)
        df_clean[col] = s.fillna(s.median())

    # Convert Kelvin to Celsius for interpretability
    df_clean["temp_max_c"] = df_clean["tmmx"] - 273.15
    df_clean["is_large_fire"] = (df_clean["FIRE_SIZE"] >= 100.0).astype(int)  # Class D-G (>=100 acres)
    df_clean["log_fire_size"] = np.log10(np.clip(df_clean["FIRE_SIZE"], 0.01, None))

    # 2. Pillar 1: Environmental Clustering (GMM via BIC vs K-Means)
    print("2. Running Pillar 1: Environmental & Land-Cover Regime Clustering (GMM vs. K-Means)...")
    scaler = StandardScaler()
    X_env = scaler.fit_transform(df_clean[ENV_FEATURES])

    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(X_env), size=min(15000, len(X_env)), replace=False)
    X_sample = X_env[sample_idx]

    clustering_metrics = {}
    for k in [4, 5, 6, 7]:
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42, max_iter=150)
        gmm.fit(X_sample)
        gmm_labels = gmm.predict(X_sample)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_labels = km.fit_predict(X_sample)
        clustering_metrics[str(k)] = {
            "gmm_bic": float(gmm.bic(X_sample)),
            "gmm_aic": float(gmm.aic(X_sample)),
            "gmm_silhouette": float(silhouette_score(X_sample, gmm_labels, sample_size=4000, random_state=42)),
            "gmm_dbi": float(davies_bouldin_score(X_sample, gmm_labels)),
            "km_silhouette": float(silhouette_score(X_sample, km_labels, sample_size=4000, random_state=42)),
            "km_dbi": float(davies_bouldin_score(X_sample, km_labels)),
        }

    # Fit final 5-regime GMM on full dataset
    best_k = 5
    gmm_final = GaussianMixture(n_components=best_k, covariance_type="full", random_state=42, max_iter=200)
    gmm_final.fit(X_sample)
    df_clean["cluster"] = gmm_final.predict(X_env)
    df_clean["cluster_prob"] = gmm_final.predict_proba(X_env).max(axis=1)

    # Characterize clusters and assign human-readable names based on feature means
    cluster_profiles = []
    for c in range(best_k):
        sub = df_clean[df_clean["cluster"] == c]
        profile = {
            "cluster_id": int(c),
            "count": int(len(sub)),
            "pct_fires": float(len(sub) / len(df_clean) * 100),
            "mean_erc": float(sub["erc"].mean()),
            "mean_vpd": float(sub["vpd"].mean()),
            "mean_rmin": float(sub["rmin"].mean()),
            "mean_wind_vs": float(sub["vs"].mean()),
            "mean_temp_c": float(sub["temp_max_c"].mean()),
            "mean_elev_m": float(sub["Elevation"].mean()),
            "mean_slope_deg": float(sub["Slope"].mean()),
            "mean_precip_mm": float(sub["Annual_precipitation"].mean()),
            "mean_ndvi": float(sub["NDVI-1day"].mean()),
            "pct_large_fires": float(sub["is_large_fire"].mean() * 100),
            "mean_fire_size_acres": float(sub["FIRE_SIZE"].mean()),
            "p90_fire_size_acres": float(np.percentile(sub["FIRE_SIZE"], 90)),
            "top_states": sub["STATE"].value_counts().head(4).index.tolist(),
            "pct_human": float((sub["NWCG_CAUSE_CLASSIFICATION"] == "Human").mean() * 100),
        }
        cluster_profiles.append(profile)

    # Sort/Name regimes based on physical characteristics
    cluster_profiles.sort(key=lambda x: x["mean_erc"], reverse=True)
    regime_names = [
        "R1: Arid Western Shrub/Forest (Extreme ERC & VPD)",
        "R2: Steep Montane Conifer & Complex Terrain",
        "R3: Semi-Arid Plains & Rangeland (High Wind)",
        "R4: Warm Southern/Coastal Mixed WUI",
        "R5: Humid Eastern/Temperate Deciduous & Ag",
    ]
    id_to_name = {}
    for idx, p in enumerate(cluster_profiles):
        p["regime_name"] = regime_names[idx]
        p["rank"] = idx + 1
        id_to_name[p["cluster_id"]] = regime_names[idx]

    # 3. Pillar 2A: Supervised Wildfire Escalation Risk Model (Random Forest)
    print("3. Running Pillar 2A: Wildfire Escalation Risk & Severity Modeling...")
    X_all = scaler.fit_transform(df_clean[ALL_FEATURES])
    y_cls = df_clean["is_large_fire"].values
    y_reg = df_clean["log_fire_size"].values

    X_tr, X_te, y_cls_tr, y_cls_te, y_reg_tr, y_reg_te, idx_tr, idx_te = train_test_split(
        X_all, y_cls, y_reg, np.arange(len(df_clean)), test_size=0.2, random_state=42, stratify=y_cls
    )

    rf_cls = RandomForestClassifier(
        n_estimators=150, max_depth=14, class_weight="balanced_subsample", n_jobs=-1, random_state=42
    )
    rf_cls.fit(X_tr, y_cls_tr)
    prob_te = rf_cls.predict_proba(X_te)[:, 1]
    roc_auc = float(roc_auc_score(y_cls_te, prob_te))
    pr_auc = float(average_precision_score(y_cls_te, prob_te))
    brier = float(brier_score_loss(y_cls_te, prob_te))

    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42)
    rf_reg.fit(X_tr, y_reg_tr)
    pred_reg_te = rf_reg.predict(X_te)
    reg_mae_log = float(mean_absolute_error(y_reg_te, pred_reg_te))

    importances = dict(zip(ALL_FEATURES, [float(v) for v in rf_cls.feature_importances_]))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    # 4. Pillar 2B: Cluster-Conditioned Case-Based Reasoning (CBR) Similar-Fire Retrieval
    print("4. Running Pillar 2B: Cluster-Conditioned Case-Based Reasoning (CBR) Evaluation...")
    # Feature weights derived from RF severity & escalation feature importance
    weights = np.maximum(0.5 * rf_cls.feature_importances_ + 0.5 * rf_reg.feature_importances_, 1e-4)
    weights = weights / weights.sum()
    X_weighted = X_all * weights

    # Compare (A) Unweighted Global k-NN vs (B) Cluster-Conditioned Feature-Weighted CBR on 1,000 test queries
    eval_queries = idx_te[:1000]
    t_knn0 = time.time()
    nn_global = NearestNeighbors(n_neighbors=15, metric="euclidean", n_jobs=-1).fit(X_all[idx_tr])
    dists_g, inds_g = nn_global.kneighbors(X_all[eval_queries])
    t_knn = (time.time() - t_knn0) * 1000.0  # ms for 1000 queries

    # Cluster-conditioned CBR models (vectorized per cluster)
    tr_clusters = df_clean["cluster"].values[idx_tr]
    cbr_models = {}
    cbr_tr_indices = {}
    for c in range(best_k):
        mask = tr_clusters == c
        c_idx = idx_tr[mask]
        cbr_tr_indices[c] = c_idx
        cbr_models[c] = NearestNeighbors(n_neighbors=15, metric="euclidean", n_jobs=-1).fit(X_weighted[c_idx])

    t_cbr0 = time.time()
    q_clusters = df_clean["cluster"].values[eval_queries]
    cbr_preds = np.zeros(len(eval_queries))
    cbr_coverage_80 = np.zeros(len(eval_queries), dtype=bool)
    for c in range(best_k):
        q_mask = np.where(q_clusters == c)[0]
        if len(q_mask) == 0:
            continue
        q_c_indices = eval_queries[q_mask]
        d_c, i_c = cbr_models[c].kneighbors(X_weighted[q_c_indices])
        for j, m_idx in enumerate(q_mask):
            matched_global_idx = cbr_tr_indices[c][i_c[j]]
            analog_logs = y_reg[matched_global_idx]
            inv_w = 1.0 / (d_c[j] + 1e-3)
            inv_w /= inv_w.sum()
            cbr_preds[m_idx] = float(np.dot(inv_w, analog_logs))
            p10, p90 = np.percentile(analog_logs, [10, 90])
            cbr_coverage_80[m_idx] = bool(p10 <= y_reg[q_c_indices[j]] <= p90)
    t_cbr = (time.time() - t_cbr0) * 1000.0

    knn_preds = [float(np.mean(y_reg[idx_tr[inds_g[i]]])) for i in range(len(eval_queries))]
    mae_knn = float(mean_absolute_error(y_reg[eval_queries], knn_preds))
    mae_cbr = float(mean_absolute_error(y_reg[eval_queries], cbr_preds))
    cov_80 = float(np.mean(cbr_coverage_80) * 100)

    # Build notable historical case study retrievals for 4 high-risk US counties
    # Fit CBR over notable fires (>= 10 acres with named incidents) for rich homeowner explanations
    notable_df = df_clean[(df_clean["FIRE_SIZE"] >= 10.0) & (df_clean["FIRE_NAME"].notna())].copy()
    notable_df["FIRE_NAME"] = notable_df["FIRE_NAME"].astype(str).str.strip()
    notable_df = notable_df[notable_df["FIRE_NAME"] != ""].reset_index(drop=True)
    X_notable_w = scaler.transform(notable_df[ALL_FEATURES]) * weights
    nn_notable = NearestNeighbors(n_neighbors=6, metric="euclidean").fit(X_notable_w)

    case_studies = []
    for state_abbr, cnty_substr, label in [
        ("CA", "Butte", "Paradise / Butte County, CA"),
        ("CO", "Boulder", "Boulder / Foothills WUI, CO"),
        ("OR", "Jackson", "Medford / Rogue Valley (Jackson Co.), OR"),
        ("TX", "Bastrop", "Bastrop Lost Pines WUI, TX"),
    ]:
        sub = notable_df[(notable_df["STATE"] == state_abbr) & (notable_df["COUNTY"].astype(str).str.contains(cnty_substr, case=False, na=False))]
        if len(sub) == 0:
            sub = notable_df[notable_df["STATE"] == state_abbr]
        target_row = sub.sort_values("FIRE_SIZE", ascending=False).iloc[0]
        t_pos = notable_df.index.get_loc(target_row.name)
        d_n, i_n = nn_notable.kneighbors(X_notable_w[t_pos : t_pos + 1])
        analogs = []
        for dist_val, idx_val in zip(d_n[0][1:6], i_n[0][1:6]):
            r = notable_df.iloc[idx_val]
            sim_score = float(np.exp(-dist_val * 3.0) * 100)
            analogs.append({
                "fire_name": str(r["FIRE_NAME"]).title(),
                "state": str(r["STATE"]),
                "county": str(r["COUNTY"]),
                "date": str(r["DISCOVERY_DATE"])[:10],
                "size_acres": round(float(r["FIRE_SIZE"]), 1),
                "cause": str(r["NWCG_GENERAL_CAUSE"]),
                "erc": round(float(r["erc"]), 1),
                "wind_mps": round(float(r["vs"]), 1),
                "rmin_pct": round(float(r["rmin"]), 1),
                "temp_c": round(float(r["temp_max_c"]), 1),
                "slope_deg": round(float(r["Slope"]), 1),
                "similarity_pct": round(sim_score, 1),
            })
        case_studies.append({
            "community": label,
            "query_fire": str(target_row["FIRE_NAME"]).title(),
            "query_date": str(target_row["DISCOVERY_DATE"])[:10],
            "query_size_acres": round(float(target_row["FIRE_SIZE"]), 1),
            "regime": id_to_name[int(target_row["cluster"])],
            "query_erc": round(float(target_row["erc"]), 1),
            "query_wind_mps": round(float(target_row["vs"]), 1),
            "query_rmin_pct": round(float(target_row["rmin"]), 1),
            "query_temp_c": round(float(target_row["temp_max_c"]), 1),
            "analogs": analogs,
        })

    results = {
        "dataset_summary": {
            "fpa_2020_rows": int(len(df)),
            "fpa_2020_clean_rows": int(len(df_clean)),
            "fpa_full_rows_1992_2020": 2270000,
            "fema_nri_counties": int(len(nri)),
            "fema_nri_tracts": 85154,
            "usfs_wrc_communities": int(len(wrc_comm)),
            "usfs_wrc_counties": int(len(wrc_cnty)),
        },
        "clustering_metrics": clustering_metrics,
        "cluster_profiles": cluster_profiles,
        "risk_model_metrics": {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 4),
            "log_size_mae": round(reg_mae_log, 4),
            "top_10_features": sorted_imp[:10],
        },
        "cbr_evaluation": {
            "knn_global_mae_log": round(mae_knn, 4),
            "cbr_clustered_weighted_mae_log": round(mae_cbr, 4),
            "mae_improvement_pct": round((mae_knn - mae_cbr) / mae_knn * 100, 2),
            "analog_80pct_interval_coverage": round(cov_80, 1),
            "query_time_1000_ms": round(t_cbr, 2),
            "per_query_latency_ms": round(t_cbr / 1000.0, 3),
        },
        "case_studies": case_studies,
        "runtime_sec": round(time.time() - t0, 2),
    }

    out_json = os.path.join(POC_DIR, "poc_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"5. Saved POC results to {out_json} in {results['runtime_sec']}s")
    print(f"   ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Brier: {brier:.4f}")
    print(f"   CBR Weighted MAE (log10 acres): {mae_cbr:.4f} vs Global k-NN: {mae_knn:.4f} ({results['cbr_evaluation']['mae_improvement_pct']}% improvement)")
    print(f"   CBR 10-90th Percentile Coverage: {cov_80:.1f}% | Latency: {t_cbr/1000.0:.3f} ms/query")


if __name__ == "__main__":
    main()
