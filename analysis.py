"""
Case Study 1 - Part 1.3 Data Analysis
Retail customer analytics: campaign response (B2C) and channel identification (trade).

Models (both deliberately different from the decision tree / random forest /
logistic regression used in earlier coursework):
  M1 - Histogram-based Gradient Boosting  (sequential boosting of shallow trees)
  M2 - Multi-layer Perceptron neural network

Datasets:
  D1 - Customer Personality Analysis (iFood CRM extract), 2240 x 29
  D2 - UCI Wholesale Customers, 440 x 8

Run with:  python3 analysis.py
Expects:   data/marketing_campaign.csv, data/wholesale_customers.csv
Writes:    figs/*.png, results.json
"""
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve, balanced_accuracy_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
RS = 42
DATA = "data"
FIGS = "figs"
plt.rcParams.update({"figure.dpi": 160, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})
C1, C2 = "#1f4e79", "#c0504d"
results = {}


# --------------------------------------------------------------------------
# Dataset 1: Customer Personality Analysis - predict campaign Response
# --------------------------------------------------------------------------
def load_d1():
    df = pd.read_csv(f"{DATA}/marketing_campaign.csv")
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"])
    ref = df["Dt_Customer"].max()
    df["Tenure_Days"] = (ref - df["Dt_Customer"]).dt.days
    df["Age"] = 2014 - df["Year_Birth"]              # data collected 2014
    mnt = ["MntWines", "MntFruits", "MntMeatProducts",
           "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    npur = ["NumDealsPurchases", "NumWebPurchases",
            "NumCatalogPurchases", "NumStorePurchases"]
    df["TotalSpend"] = df[mnt].sum(axis=1)
    df["TotalPurchases"] = df[npur].sum(axis=1)
    df["AvgBasketValue"] = df["TotalSpend"] / df["TotalPurchases"].replace(0, np.nan)
    df["PriorCampaignsAccepted"] = df[["AcceptedCmp1", "AcceptedCmp2",
                                       "AcceptedCmp3", "AcceptedCmp4",
                                       "AcceptedCmp5"]].sum(axis=1)
    df["WineShare"] = df["MntWines"] / df["TotalSpend"].replace(0, np.nan)
    df["Dependents"] = df["Kidhome"] + df["Teenhome"]

    y = df["Response"].values
    drop = ["ID", "Response", "Z_CostContact", "Z_Revenue",
            "Dt_Customer", "Year_Birth"]
    X = df.drop(columns=drop)
    cat = ["Education", "Marital_Status"]
    num = [c for c in X.columns if c not in cat]
    return X, y, num, cat, df


def make_pipes(num, cat):
    pre_tree = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])
    pre_net = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])
    gb = Pipeline([("pre", pre_tree),
                   ("clf", HistGradientBoostingClassifier(
                       max_iter=400, learning_rate=0.06, max_leaf_nodes=15,
                       min_samples_leaf=25, l2_regularization=1.0,
                       early_stopping=True, validation_fraction=0.15,
                       random_state=RS))])
    nn = Pipeline([("pre", pre_net),
                   ("clf", MLPClassifier(hidden_layer_sizes=(64, 32),
                                         activation="relu", alpha=1e-3,
                                         learning_rate_init=3e-3,
                                         max_iter=1500, early_stopping=True,
                                         n_iter_no_change=25, random_state=RS))])
    return {"Gradient Boosting": gb, "Neural Network (MLP)": nn}


def lift_at(y, p, frac=0.20):
    k = max(1, int(len(y) * frac))
    idx = np.argsort(-p)[:k]
    return (y[idx].mean() / y.mean()) if y.mean() > 0 else np.nan


def evaluate(name, y, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    return {
        "model": name, "threshold": round(float(thr), 4),
        "PR_AUC": average_precision_score(y, p),
        "ROC_AUC": roc_auc_score(y, p),
        "Precision": precision_score(y, yhat, zero_division=0),
        "Recall": recall_score(y, yhat, zero_division=0),
        "F1": f1_score(y, yhat, zero_division=0),
        "BalancedAcc": balanced_accuracy_score(y, yhat),
        "Brier": brier_score_loss(y, p),
        "Lift_top20": lift_at(y, p, 0.20),
        "cm": [int(tn), int(fp), int(fn), int(tp)],
    }


def run(tag, X, y, pipes, positive_label, thr_mode="f1"):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    out, probs = [], {}
    for name, pipe in pipes.items():
        p = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        probs[name] = p
        if thr_mode == "f1":
            pr, rc, th = precision_recall_curve(y, p)
            f1 = 2 * pr[:-1] * rc[:-1] / np.clip(pr[:-1] + rc[:-1], 1e-9, None)
            thr = th[int(np.nanargmax(f1))]
        else:
            thr = 0.5
        out.append(evaluate(name, y, p, thr))
    results[tag] = {"positive": positive_label, "prevalence": float(np.mean(y)),
                    "n": int(len(y)), "metrics": out}
    return probs


X1, y1, num1, cat1, raw1 = load_d1()
pipes1 = make_pipes(num1, cat1)
probs1 = run("D1_campaign_response", X1, y1, pipes1,
             "Responded to final campaign", thr_mode="f1")

# --------------------------------------------------------------------------
# Dataset 2: UCI Wholesale Customers - predict Channel (Retail vs Horeca)
# --------------------------------------------------------------------------
w = pd.read_csv(f"{DATA}/wholesale_customers.csv")
y2 = (w["Channel"] == 2).astype(int).values          # 1 = Retail outlet
spend = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
X2 = w.drop(columns=["Channel"]).copy()
for c in spend:                                       # heavy right skew
    X2[f"log_{c}"] = np.log1p(X2[c])
X2["TotalSpend"] = w[spend].sum(axis=1)
X2["log_TotalSpend"] = np.log1p(X2["TotalSpend"])
for c in spend:
    X2[f"share_{c}"] = w[c] / X2["TotalSpend"]
X2["Region"] = X2["Region"].astype(str)
cat2 = ["Region"]
num2 = [c for c in X2.columns if c not in cat2]
pipes2 = make_pipes(num2, cat2)
probs2 = run("D2_channel", X2, y2, pipes2, "Retail channel", thr_mode="f1")

# --------------------------------------------------------------------------
# Permutation importance (fit on a hold-out split, report top drivers)
# --------------------------------------------------------------------------
def importances(X, y, pipe, cat, top=10):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=RS)
    pipe.fit(Xtr, ytr)
    r = permutation_importance(pipe, Xte, yte, n_repeats=20,
                               random_state=RS, scoring="average_precision")
    s = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)
    return s.head(top), pipe


imp1, fit1 = importances(X1, y1, pipes1["Gradient Boosting"], cat1)
imp2, fit2 = importances(X2, y2, pipes2["Gradient Boosting"], cat2)
results["importance"] = {"D1": {k: float(v) for k, v in imp1.items()},
                         "D2": {k: float(v) for k, v in imp2.items()}}
# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def curves(tag, y, probs, fname, title):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for (name, p), c in zip(probs.items(), (C1, C2)):
        fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(fpr, tpr, color=c, lw=1.6,
                   label=f"{name} (AUC={roc_auc_score(y,p):.3f})")
        pr, rc, _ = precision_recall_curve(y, p)
        ax[1].plot(rc, pr, color=c, lw=1.6,
                   label=f"{name} (AP={average_precision_score(y,p):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate",
              title="ROC curve")
    ax[1].axhline(y.mean(), color="k", ls="--", lw=0.8)
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    for a in ax:
        a.legend(fontsize=6.5, loc="lower left" if a is ax[1] else "lower right")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/{fname}", bbox_inches="tight")
    plt.close(fig)


curves("D1", y1, probs1, "d1_curves.png",
       "Dataset 1 - predicting marketing campaign response (5-fold CV)")
curves("D2", y2, probs2, "d2_curves.png",
       "Dataset 2 - predicting retail vs food-service channel (5-fold CV)")


def imp_fig(s1, s2, fname):
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.2))
    for a, s, c, t in ((ax[0], s1[::-1], C1, "Dataset 1: campaign response"),
                       (ax[1], s2[::-1], C2, "Dataset 2: channel")):
        a.barh(range(len(s)), s.values, color=c, alpha=0.85)
        a.set_yticks(range(len(s)))
        a.set_yticklabels(s.index, fontsize=7)
        a.set_xlabel("Drop in average precision when shuffled")
        a.set_title(t, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/{fname}", bbox_inches="tight")
    plt.close(fig)


imp_fig(imp1, imp2, "importance.png")


def lift_fig(y, probs, fname, title):
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    deciles = np.arange(0.05, 1.01, 0.05)
    for (name, p), c in zip(probs.items(), (C1, C2)):
        ax.plot(deciles * 100, [lift_at(y, p, d) for d in deciles],
                color=c, lw=1.6, marker="o", ms=2.5, label=name)
    ax.axhline(1, color="k", ls="--", lw=0.8, label="Random targeting")
    ax.set(xlabel="% of customer base contacted (highest score first)",
           ylabel="Lift vs random", title=title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/{fname}", bbox_inches="tight")
    plt.close(fig)


lift_fig(y1, probs1, "d1_lift.png", "Campaign targeting lift (Dataset 1)")

# class balance / EDA figure
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
raw1.groupby("Response")[["MntWines", "MntMeatProducts", "MntGoldProds"]].mean().plot(
    kind="bar", ax=ax[0], color=[C1, C2, "#7f9db9"], rot=0)
ax[0].set(xlabel="Responded to campaign (0 = no, 1 = yes)",
          ylabel="Mean 2-year spend (currency units)",
          title="Dataset 1: category spend by response")
ax[0].legend(fontsize=6.5)
w.assign(Chan=w["Channel"].map({1: "Food service", 2: "Retail"})).groupby("Chan")[
    ["Fresh", "Grocery", "Detergents_Paper"]].mean().plot(
    kind="bar", ax=ax[1], color=[C1, C2, "#7f9db9"], rot=0)
ax[1].set(xlabel="Channel", ylabel="Mean annual spend (m.u.)",
          title="Dataset 2: category spend by channel")
ax[1].legend(fontsize=6.5)
fig.tight_layout()
fig.savefig(f"{FIGS}/eda.png", bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------
with open("results.json", "w") as f:
    json.dump(results, f, indent=1, default=float)

for tag, blk in results.items():
    if tag == "importance":
        continue
    print(f"\n=== {tag}  n={blk['n']}  prevalence={blk['prevalence']:.3f}")
    for m in blk["metrics"]:
        print(f"  {m['model']:<22} PR-AUC={m['PR_AUC']:.3f}  ROC-AUC={m['ROC_AUC']:.3f} "
              f"P={m['Precision']:.3f} R={m['Recall']:.3f} F1={m['F1']:.3f} "
              f"BA={m['BalancedAcc']:.3f} Brier={m['Brier']:.3f} "
              f"Lift@20%={m['Lift_top20']:.2f} thr={m['threshold']:.3f} cm={m['cm']}")
print(f"\nTop drivers D1:", {k: round(v, 4) for k, v in imp1.items()})
print("Top drivers D2:", {k: round(v, 4) for k, v in imp2.items()})
