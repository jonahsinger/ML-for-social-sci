# model_perf_and_feature_importance.py
# - 80/20 held-out test split (stratified)
# - Tune on TRAIN ONLY (5-fold CV):
#     * Logistic: plain (fixed C=1), ridge (L2), lasso (L1), elastic-net (L1+L2)
#     * XGBoost: RandomizedSearchCV
#     * Neural Net (MLP): RandomizedSearchCV
# - Feature sets: SES only, Lifestyle only, SES+Lifestyle
# - Saves:
#     * Feature importance plots -> ./feature_imp/
#     * ROC + PR curves (with baseline prevalence line) -> ./model_preformance/
#     * Metrics CSV -> ./model_preformance/metrics.csv
#
# Run:
#   python model_perf_and_feature_importance.py
#
# Assumes:
#   ./data/2015.csv exists

import os
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.inspection import permutation_importance

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neural_network import MLPClassifier

import xgboost as xgb
from xgboost import XGBClassifier


# ----------------------------
# Settings / I/O
# ----------------------------
DATA_PATH = os.path.join("data", "2015.csv")

FEATURE_IMP_DIR = os.path.join("feature_imp")
MODEL_PERF_DIR  = os.path.join("model_preformance")  # (spelled as you requested)
os.makedirs(FEATURE_IMP_DIR, exist_ok=True)
os.makedirs(MODEL_PERF_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


# ----------------------------
# Feature sets
# ----------------------------
SES_FEATURES = [
    "SEX", "MARITAL", "EDUCA", "EMPLOY1", "INCOME2", "RENTHOM1", "CHILDREN", "VETERAN3", "INTERNET",
    "HLTHPLN1", "PERSDOC2",
]

LIFESTYLE_FEATURES = [
    "SMOKE100", "SMOKDAY2", "USENOW3",
    "ALCDAY5", "AVEDRNK2", "DRNK3GE5", "MAXDRNKS",
    "FRUITJU1", "FRUIT1", "VEGETAB1", "FVBEANS", "FVGREEN", "FVORANG",
    "EXERANY2", "STRENGTH",
    "WEIGHT2", "HEIGHT3",
]

FEATURE_SETS = {
    "ses_only": SES_FEATURES,
    "lifestyle_only": LIFESTYLE_FEATURES,
    "ses_plus_lifestyle": SES_FEATURES + LIFESTYLE_FEATURES,
}

# BRFSS-style missing codes
MISSING_CODES = {7, 9, 77, 88, 99, 777, 888, 999, 9999, 99999}


# ----------------------------
# Helpers
# ----------------------------
def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", s)

def replace_missing_codes(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.where(~s.isin(MISSING_CODES), np.nan)

def plot_roc(y_true, y_prob, title, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_pr(y_true, y_prob, title, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    prevalence = float(np.mean(y_true))

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.hlines(prevalence, 0, 1, linestyles="--", label=f"Baseline prevalence = {prevalence:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_barh(values, names, outpath, title, xlabel, top_k=30):
    values = np.asarray(values).ravel()
    names = list(names)

    order = np.argsort(np.abs(values))[::-1]
    top_k = min(top_k, len(names))
    idx = order[:top_k]

    plt.figure(figsize=(10, 8))
    plt.barh(range(top_k)[::-1], values[idx][::-1])
    plt.yticks(range(top_k)[::-1], [names[i] for i in idx][::-1])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def xgb_gain_importance(model, feature_names):
    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")
    gains = np.zeros(len(feature_names), dtype=float)
    for k, v in score.items():
        if k.startswith("f"):
            gains[int(k[1:])] = v
    return gains

def mean_abs_contrib(contrib_matrix):
    # contrib_matrix shape: (n_samples, n_features+1) with last col = bias
    return np.mean(np.abs(contrib_matrix[:, :-1]), axis=0)

def try_import_shap():
    try:
        import shap  # noqa
        return True
    except Exception:
        return False


# ----------------------------
# Load + clean
# ----------------------------
df = pd.read_csv(DATA_PATH, low_memory=False)

# Strip byte-string-looking fields: b'01292015' -> 01292015
for col in df.columns:
    if df[col].dtype == object:
        df[col] = (
            df[col].astype(str)
            .str.replace(r"^b'", "", regex=True)
            .str.replace(r"'$", "", regex=True)
        )

# Target:
# Good (1) if GENHLTH in {1,2,3}; Not good (0) if {4,5}
df["GENHLTH"] = pd.to_numeric(df["GENHLTH"], errors="coerce")
df = df[df["GENHLTH"].isin([1, 2, 3, 4, 5])].copy()
df["y"] = df["GENHLTH"].isin([1, 2, 3]).astype(int)

# Ensure features exist and replace missing codes with NaN
ALL_FEATURES = list(dict.fromkeys(SES_FEATURES + LIFESTYLE_FEATURES))
missing_cols = [c for c in ALL_FEATURES if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in 2015.csv: {missing_cols}")

for c in ALL_FEATURES:
    df[c] = replace_missing_codes(df[c])

y = df["y"].to_numpy()

# One shared split across all feature sets
idx = np.arange(len(df))
idx_tr, idx_te, y_tr, y_te = train_test_split(
    idx, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Preprocessing for linear + MLP (NOTE: this does NOT one-hot encode; it treats codes as numeric)
imputer = SimpleImputer(strategy="most_frequent")
scaler = StandardScaler()


# ----------------------------
# Model runners
# ----------------------------
def eval_and_save_curves(y_true, y_prob, model_name, feature_set_tag):
    tag = sanitize(feature_set_tag)
    m = sanitize(model_name)

    roc_path = os.path.join(MODEL_PERF_DIR, f"roc_{m}_{tag}.png")
    pr_path  = os.path.join(MODEL_PERF_DIR, f"pr_{m}_{tag}.png")

    plot_roc(y_true, y_prob, title=f"ROC: {model_name} ({feature_set_tag})", out_path=roc_path)
    plot_pr(y_true, y_prob, title=f"PR: {model_name} ({feature_set_tag})", out_path=pr_path)

def run_feature_set(feature_set_tag, features, rows_out):
    X = df[features].copy()
    X_tr = X.iloc[idx_tr]
    X_te = X.iloc[idx_te]

    # Impute first (shared)
    X_tr_i = imputer.fit_transform(X_tr)
    X_te_i = imputer.transform(X_te)

    # Scaled version for linear + MLP
    X_tr_s = scaler.fit_transform(X_tr_i)
    X_te_s = scaler.transform(X_te_i)

    # ----------------------------
    # Logistic: plain + ridge/lasso/elasticnet (CV on TRAIN only)
    # ----------------------------
    Cs = np.logspace(-3, 3, 20)

    # Plain logistic (L2, fixed C=1.0)
    plain = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=8000,
        random_state=RANDOM_STATE,
    )
    plain.fit(X_tr_s, y_tr)
    p = plain.predict_proba(X_te_s)[:, 1]
    auc = roc_auc_score(y_te, p)
    ap  = average_precision_score(y_te, p)
    rows_out.append([feature_set_tag, "logreg_plain", auc, ap, "C=1.0 (fixed)"])
    eval_and_save_curves(y_te, p, "LogReg plain (L2, C=1.0)", feature_set_tag)
    save_barh(
        plain.coef_.ravel(), features,
        os.path.join(FEATURE_IMP_DIR, f"logreg_plain_{sanitize(feature_set_tag)}_coef.png"),
        title=f"LogReg plain — {feature_set_tag}\nAUC={auc:.3f} AP={ap:.3f}",
        xlabel="Coefficient (after scaling)"
    )

    # Ridge (L2) CV
    ridge_cv = LogisticRegressionCV(
        Cs=Cs, cv=cv, penalty="l2", solver="lbfgs",
        scoring="roc_auc", n_jobs=-1, max_iter=8000, refit=True
    )
    ridge_cv.fit(X_tr_s, y_tr)
    p = ridge_cv.predict_proba(X_te_s)[:, 1]
    auc = roc_auc_score(y_te, p)
    ap  = average_precision_score(y_te, p)
    best_C = float(ridge_cv.C_[0])
    rows_out.append([feature_set_tag, "logreg_ridge_cv", auc, ap, f"best_C={best_C:.6g}"])
    eval_and_save_curves(y_te, p, f"LogReg ridge CV (best C={best_C:.3g})", feature_set_tag)
    save_barh(
        ridge_cv.coef_.ravel(), features,
        os.path.join(FEATURE_IMP_DIR, f"logreg_ridge_cv_{sanitize(feature_set_tag)}_coef.png"),
        title=f"LogReg ridge CV — {feature_set_tag}\nAUC={auc:.3f} AP={ap:.3f} | best C={best_C:.3g}",
        xlabel="Coefficient (after scaling)"
    )

    # Lasso (L1) CV
    lasso_cv = LogisticRegressionCV(
        Cs=Cs, cv=cv, penalty="l1", solver="saga",
        scoring="roc_auc", n_jobs=-1, max_iter=12000, refit=True
    )
    lasso_cv.fit(X_tr_s, y_tr)
    p = lasso_cv.predict_proba(X_te_s)[:, 1]
    auc = roc_auc_score(y_te, p)
    ap  = average_precision_score(y_te, p)
    best_C = float(lasso_cv.C_[0])
    rows_out.append([feature_set_tag, "logreg_lasso_cv", auc, ap, f"best_C={best_C:.6g}"])
    eval_and_save_curves(y_te, p, f"LogReg lasso CV (best C={best_C:.3g})", feature_set_tag)
    save_barh(
        lasso_cv.coef_.ravel(), features,
        os.path.join(FEATURE_IMP_DIR, f"logreg_lasso_cv_{sanitize(feature_set_tag)}_coef.png"),
        title=f"LogReg lasso CV — {feature_set_tag}\nAUC={auc:.3f} AP={ap:.3f} | best C={best_C:.3g}",
        xlabel="Coefficient (after scaling)"
    )

    # Elastic Net CV
    enet_cv = LogisticRegressionCV(
        Cs=np.logspace(-3, 3, 12),
        cv=cv,
        penalty="elasticnet",
        solver="saga",
        l1_ratios=[0.1, 0.3, 0.5, 0.7, 0.9],
        scoring="roc_auc",
        n_jobs=-1,
        max_iter=16000,
        refit=True,
    )
    enet_cv.fit(X_tr_s, y_tr)
    p = enet_cv.predict_proba(X_te_s)[:, 1]
    auc = roc_auc_score(y_te, p)
    ap  = average_precision_score(y_te, p)

    best_C = float(enet_cv.C_[0])
    best_l1 = float(enet_cv.l1_ratio_[0]) if isinstance(enet_cv.l1_ratio_, (list, np.ndarray)) else float(enet_cv.l1_ratio_)
    rows_out.append([feature_set_tag, "logreg_elasticnet_cv", auc, ap, f"best_C={best_C:.6g}, best_l1={best_l1:.2f}"])
    eval_and_save_curves(y_te, p, f"LogReg elasticnet CV (C={best_C:.3g}, l1={best_l1:.2f})", feature_set_tag)
    save_barh(
        enet_cv.coef_.ravel(), features,
        os.path.join(FEATURE_IMP_DIR, f"logreg_elasticnet_cv_{sanitize(feature_set_tag)}_coef.png"),
        title=f"LogReg elasticnet CV — {feature_set_tag}\nAUC={auc:.3f} AP={ap:.3f} | C={best_C:.3g}, l1={best_l1:.2f}",
        xlabel="Coefficient (after scaling)"
    )

    # ----------------------------
    # XGBoost tuned (CV on TRAIN only)
    # ----------------------------
    xgb_base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb_param_dist = {
        "n_estimators": [200, 400, 800],
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 10],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
        "gamma": [0.0, 0.5, 1.0, 2.0],
    }
    xgb_search = RandomizedSearchCV(
        xgb_base,
        param_distributions=xgb_param_dist,
        n_iter=25,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    xgb_search.fit(X_tr_i, y_tr)
    best_xgb = xgb_search.best_estimator_

    p = best_xgb.predict_proba(X_te_i)[:, 1]
    auc = roc_auc_score(y_te, p)
    ap  = average_precision_score(y_te, p)
    rows_out.append([feature_set_tag, "xgboost_tuned", auc, ap, f"best_params={xgb_search.best_params_}"])
    eval_and_save_curves(y_te, p, "XGBoost tuned", feature_set_tag)

    # XGB gain importance
    gains = xgb_gain_importance(best_xgb, features)
    save_barh(
        gains, features,
        os.path.join(FEATURE_IMP_DIR, f"xgb_{sanitize(feature_set_tag)}_gain.png"),
        title=f"XGBoost gain — {feature_set_tag}\nAUC={auc:.3f} AP={ap:.3f}",
        xlabel="Gain"
    )

    # XGB SHAP-style contributions via pred_contribs (no shap package needed)
    X_explain = X_te_i
    if X_explain.shape[0] > 5000:
        rs = np.random.RandomState(RANDOM_STATE)
        X_explain = X_explain[rs.choice(X_explain.shape[0], 5000, replace=False)]
    contrib = best_xgb.get_booster().predict(
        xgb.DMatrix(X_explain),
        pred_contribs=True
    )
    mean_abs = mean_abs_contrib(contrib)
    save_barh(
        mean_abs, features,
        os.path.join(FEATURE_IMP_DIR, f"xgb_{sanitize(feature_set_tag)}_meanabs_contrib.png"),
        title=f"XGBoost mean(|contrib|) — {feature_set_tag}\nAUC={auc:.3f} AP={ap:.3f}",
        xlabel="Mean |contribution| (pred_contribs)"
    )

    # Optional: SHAP plots if shap imports cleanly
    if try_import_shap():
        import shap
        X_explain2 = X_te_i
        if X_explain2.shape[0] > 2000:
            rs = np.random.RandomState(RANDOM_STATE)
            X_explain2 = X_explain2[rs.choice(X_explain2.shape[0], 2000, replace=False)]

        explainer = shap.TreeExplainer(best_xgb)
        shap_vals = explainer.shap_values(X_explain2)

        # Bar
        plt.figure()
        shap.summary_plot(shap_vals, X_explain2, feature_names=features, plot_type="bar", show=False)
        plt.title(f"XGBoost SHAP (bar) — {feature_set_tag}")
        plt.tight_layout()
        plt.savefig(os.path.join(FEATURE_IMP_DIR, f"xgb_{sanitize(feature_set_tag)}_shap_bar.png"), dpi=200)
        plt.close()

        # Beeswarm
        plt.figure()
        shap.summary_plot(shap_vals, X_explain2, feature_names=features, show=False)
        plt.title(f"XGBoost SHAP (beeswarm) — {feature_set_tag}")
        plt.tight_layout()
        plt.savefig(os.path.join(FEATURE_IMP_DIR, f"xgb_{sanitize(feature_set_tag)}_shap_beeswarm.png"), dpi=200)
        plt.close()

    # ----------------------------
    # MLP tuned (CV on TRAIN only)
    # ----------------------------
    mlp = MLPClassifier(
        max_iter=250,
        random_state=RANDOM_STATE,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.1,
    )
    mlp_param_dist = {
        "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "learning_rate_init": [1e-4, 5e-4, 1e-3, 5e-3],
        "activation": ["relu", "tanh"],
    }
    mlp_search = RandomizedSearchCV(
        mlp,
        param_distributions=mlp_param_dist,
        n_iter=20,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    mlp_search.fit(X_tr_s, y_tr)
    best_mlp = mlp_search.best_estimator_

    p = best_mlp.predict_proba(X_te_s)[:, 1]
    auc = roc_auc_score(y_te, p)
    ap  = average_precision_score(y_te, p)
    rows_out.append([feature_set_tag, "mlp_tuned", auc, ap, f"best_params={mlp_search.best_params_}"])
    eval_and_save_curves(y_te, p, "MLP tuned", feature_set_tag)

    # MLP feature importance: permutation importance on test (ROC-AUC drop)
    perm = permutation_importance(
        best_mlp, X_te_s, y_te,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="roc_auc"
    )
    save_barh(
        perm.importances_mean, features,
        os.path.join(FEATURE_IMP_DIR, f"mlp_{sanitize(feature_set_tag)}_perm_importance.png"),
        title=f"MLP permutation importance — {feature_set_tag}\nAUC={auc:.3f} AP={ap:.3f}",
        xlabel="Δ ROC-AUC (permutation)"
    )

    # Optional: Kernel SHAP for MLP if shap imports cleanly (can be slow)
    if try_import_shap():
        import shap
        f = lambda X_: best_mlp.predict_proba(X_)[:, 1]

        rs = np.random.RandomState(RANDOM_STATE)
        bg_n = min(100, X_tr_s.shape[0])
        ex_n = min(200, X_te_s.shape[0])
        bg_idx = rs.choice(X_tr_s.shape[0], size=bg_n, replace=False)
        ex_idx = rs.choice(X_te_s.shape[0], size=ex_n, replace=False)

        background = X_tr_s[bg_idx]
        explain = X_te_s[ex_idx]

        explainer = shap.KernelExplainer(f, background)
        shap_vals = explainer.shap_values(explain, nsamples=200)

        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
        save_barh(
            mean_abs_shap, features,
            os.path.join(FEATURE_IMP_DIR, f"mlp_{sanitize(feature_set_tag)}_shap_meanabs.png"),
            title=f"MLP mean(|SHAP|) — {feature_set_tag}",
            xlabel="Mean |SHAP| (KernelExplainer)"
        )


# ----------------------------
# Run all feature sets + save metrics
# ----------------------------
metrics_rows = []
for fs_tag, feats in FEATURE_SETS.items():
    run_feature_set(fs_tag, feats, metrics_rows)

metrics_df = pd.DataFrame(
    metrics_rows,
    columns=["feature_set", "model", "roc_auc", "avg_precision", "notes"]
).sort_values(["feature_set", "roc_auc"], ascending=[True, False])

metrics_csv_path = os.path.join(MODEL_PERF_DIR, "metrics.csv")
metrics_df.to_csv(metrics_csv_path, index=False)

print(f"Saved feature importance plots to: {FEATURE_IMP_DIR}/")
print(f"Saved ROC/PR curves + metrics to: {MODEL_PERF_DIR}/")
print(metrics_df.to_string(index=False))
