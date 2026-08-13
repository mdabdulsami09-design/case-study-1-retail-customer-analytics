"""
Supporting experiments for Appendix E (development process).

E1  Baseline run: library-default hyperparameters and a fixed 0.5 decision
    threshold, to establish what the tuned pipeline actually bought us and
    whether the neural network degenerates on the imbalanced target.

E2  Collinearity ablation on D2: the engineered feature set contains raw spend,
    log spend, spend shares and totals, which are mutually collinear. Refit
    with only Region + log spend to measure what the redundancy costs.

Both use the same stratified 5-fold protocol as analysis.py so the numbers are
directly comparable to Table 1 of the report.

Run with:  python3 experiments.py
Expects:   data/marketing_campaign.csv, data/wholesale_customers.csv
Writes:    experiments.json
"""
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import warnings
warnings.filterwarnings("ignore")

RS = 42
DATA = "data"
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
out = {}


def pre(num, cat, scale):
    steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("sc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(steps), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])


def score(X, y, clf, num, cat, scale):
    pipe = Pipeline([("pre", pre(num, cat, scale)), ("clf", clf)])
    p = cross_val_predict(pipe, X, y, cv=CV, method="predict_proba")[:, 1]
    yhat = (p >= 0.5).astype(int)          # fixed default threshold
    return {
        "PR_AUC": round(float(average_precision_score(y, p)), 4),
        "ROC_AUC": round(float(roc_auc_score(y, p)), 4),
        "recall_at_0.5": round(float(recall_score(y, yhat, zero_division=0)), 4),
        "positives_predicted": int(yhat.sum()),
        "actual_positives": int(y.sum()),
    }


# ---------------------------------------------------------------- data -----
def load_d1():
    df = pd.read_csv(f"{DATA}/marketing_campaign.csv")
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"])
    df["Tenure_Days"] = (df["Dt_Customer"].max() - df["Dt_Customer"]).dt.days
    df["Age"] = 2014 - df["Year_Birth"]
    mnt = ["MntWines", "MntFruits", "MntMeatProducts",
           "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    npur = ["NumDealsPurchases", "NumWebPurchases",
            "NumCatalogPurchases", "NumStorePurchases"]
    df["TotalSpend"] = df[mnt].sum(axis=1)
    df["TotalPurchases"] = df[npur].sum(axis=1)
    df["AvgBasketValue"] = df["TotalSpend"] / df["TotalPurchases"].replace(0, np.nan)
    df["PriorCampaignsAccepted"] = df[[f"AcceptedCmp{i}" for i in range(1, 6)]].sum(axis=1)
    df["WineShare"] = df["MntWines"] / df["TotalSpend"].replace(0, np.nan)
    df["Dependents"] = df["Kidhome"] + df["Teenhome"]
    y = df["Response"].values
    X = df.drop(columns=["ID", "Response", "Z_CostContact", "Z_Revenue",
                         "Dt_Customer", "Year_Birth"])
    cat = ["Education", "Marital_Status"]
    return X, y, [c for c in X.columns if c not in cat], cat


spend = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]


def load_d2(full=True):
    w = pd.read_csv(f"{DATA}/wholesale_customers.csv")
    y = (w["Channel"] == 2).astype(int).values
    X = w.drop(columns=["Channel"]).copy()
    for c in spend:
        X[f"log_{c}"] = np.log1p(X[c])
    if full:
        X["TotalSpend"] = w[spend].sum(axis=1)
        X["log_TotalSpend"] = np.log1p(X["TotalSpend"])
        for c in spend:
            X[f"share_{c}"] = w[c] / X["TotalSpend"]
    else:
        X = X.drop(columns=spend)          # keep Region + log spend only
    X["Region"] = X["Region"].astype(str)
    cat = ["Region"]
    return X, y, [c for c in X.columns if c not in cat], cat


# ------------------------------------------------- E1: default baseline ----
X1, y1, num1, cat1 = load_d1()
X2f, y2, num2f, cat2 = load_d2(full=True)

out["E1_defaults_threshold_0.5"] = {
    "D1_gradient_boosting": score(X1, y1, HistGradientBoostingClassifier(random_state=RS), num1, cat1, False),
    "D1_neural_network":    score(X1, y1, MLPClassifier(random_state=RS), num1, cat1, True),
    "D2_gradient_boosting": score(X2f, y2, HistGradientBoostingClassifier(random_state=RS), num2f, cat2, False),
    "D2_neural_network":    score(X2f, y2, MLPClassifier(random_state=RS), num2f, cat2, True),
}

# ------------------------------------------- E2: collinearity ablation -----
X2r, _, num2r, _ = load_d2(full=False)
tuned_gb = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=15,
                min_samples_leaf=25, l2_regularization=1.0,
                early_stopping=True, validation_fraction=0.15, random_state=RS)
tuned_nn = dict(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-3,
                learning_rate_init=3e-3, max_iter=1500, early_stopping=True,
                n_iter_no_change=25, random_state=RS)

full_gb = score(X2f, y2, HistGradientBoostingClassifier(**tuned_gb), num2f, cat2, False)
red_gb = score(X2r, y2, HistGradientBoostingClassifier(**tuned_gb), num2r, cat2, False)
full_nn = score(X2f, y2, MLPClassifier(**tuned_nn), num2f, cat2, True)
red_nn = score(X2r, y2, MLPClassifier(**tuned_nn), num2r, cat2, True)

out["E2_collinearity_ablation_D2"] = {
    "n_features_full": len(num2f) + 1,
    "n_features_reduced": len(num2r) + 1,
    "gradient_boosting": {"full": full_gb, "reduced": red_gb,
                          "delta_PR_AUC": round(red_gb["PR_AUC"] - full_gb["PR_AUC"], 4)},
    "neural_network": {"full": full_nn, "reduced": red_nn,
                       "delta_PR_AUC": round(red_nn["PR_AUC"] - full_nn["PR_AUC"], 4)},
}

with open("experiments.json", "w") as f:
    json.dump(out, f, indent=1)

print(json.dumps(out, indent=1))
