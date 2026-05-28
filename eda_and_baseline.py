"""
Explorative & Deskriptive Datenanalyse + Logistische Regression Baseline
Telco Customer Churn – Zwischenstandspräsentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score
)

def eda_baseline(df):
    warnings.filterwarnings("ignore")

    # ── Output directory ──────────────────────────────────────────────────────────
    OUT = "output/plots/presentation"
    os.makedirs(OUT, exist_ok=True)

    # ── Style ─────────────────────────────────────────────────────────────────────
    CHURN_PALETTE = {"No": "#4C9BE8", "Yes": "#E8614C"}
    CHURN_COLORS  = [CHURN_PALETTE["No"], CHURN_PALETTE["Yes"]]
    sns.set_theme(style="whitegrid", font_scale=1.15)
    plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight",
                         "font.family": "DejaVu Sans"})

    # ── 1. Load & fix ─────────────────────────────────────────────────────────────
    df["TotalCharges"]  = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})


    # ══════════════════════════════════════════════════════════════════════════════
    # PLOT 1 – Churn-Verteilung (Donut + Absolute Zahlen)
    # ══════════════════════════════════════════════════════════════════════════════
    churn_counts = df["Churn"].value_counts()
    labels = [f"No Churn\n{churn_counts['No']:,}", f"Churn\n{churn_counts['Yes']:,}"]

    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, texts, autotexts = ax.pie(
        churn_counts, labels=labels,
        colors=CHURN_COLORS, autopct="%1.1f%%",
        startangle=90, pctdistance=0.78,
        wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(13); at.set_fontweight("bold"); at.set_color("white")
    ax.set_title("Churn-Verteilung im Datensatz", fontsize=14, fontweight="bold", pad=15)
    fig.savefig(f"{OUT}/01_churn_distribution.png")
    plt.close()
    print("✓ Plot 1: Churn-Verteilung")


    # ══════════════════════════════════════════════════════════════════════════════
    # PLOT 2 – Churn-Rate nach Schlüssel-Kategorien
    # ══════════════════════════════════════════════════════════════════════════════
    cat_features = ["Contract", "InternetService", "PaymentMethod", "SeniorCitizen"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for i, feat in enumerate(cat_features):
        ct = df.groupby(feat)["Churn"].value_counts(normalize=True).unstack().fillna(0) * 100
        ct = ct.sort_values("Yes", ascending=True)
        ct[["No", "Yes"]].plot(
            kind="barh", stacked=True, ax=axes[i],
            color=CHURN_COLORS, edgecolor="white", linewidth=0.8
        )
        axes[i].set_title(feat, fontweight="bold")
        axes[i].set_xlabel("Anteil (%)")
        axes[i].set_xlim(0, 100)
        axes[i].legend(["No Churn", "Churn"], loc="lower right", fontsize=9)
        axes[i].set_ylabel("")
        # add % labels on churn bars
        for bar in axes[i].patches[len(ct):]:
            w = bar.get_width()
            if w > 4:
                axes[i].text(bar.get_x() + w / 2, bar.get_y() + bar.get_height() / 2,
                             f"{w:.0f}%", va="center", ha="center",
                             fontsize=9, color="white", fontweight="bold")

    fig.suptitle("Churn-Rate nach Kategorie", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(f"{OUT}/02_churn_by_category.png")
    plt.close()
    print("✓ Plot 2: Churn nach Kategorie")


    # ══════════════════════════════════════════════════════════════════════════════
    # PLOT 3 – Numerische Verteilungen: Tenure, MonthlyCharges, TotalCharges
    # ══════════════════════════════════════════════════════════════════════════════
    num_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    titles = ["Tenure (Monate)", "Monatliche Kosten ($)", "Gesamtkosten ($)"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, feat, title in zip(axes, num_features, titles):
        for label, color in CHURN_PALETTE.items():
            subset = df[df["Churn"] == label][feat]
            ax.hist(subset, bins=30, alpha=0.65, color=color,
                    label=f"{'Churn' if label=='Yes' else 'No Churn'} (n={len(subset):,})",
                    density=True, edgecolor="white", linewidth=0.4)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(title)
        ax.set_ylabel("Dichte")
        ax.legend(fontsize=9)

    fig.suptitle("Verteilung numerischer Features nach Churn-Status",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(f"{OUT}/03_numeric_distributions.png")
    plt.close()
    print("✓ Plot 3: Numerische Verteilungen")


    # ══════════════════════════════════════════════════════════════════════════════
    # PLOT 4 – Korrelations-Heatmap (numerisch + encoded)
    # ══════════════════════════════════════════════════════════════════════════════
    df_enc = df.copy()
    df_enc["Churn_bin"] = (df_enc["Churn"] == "Yes").astype(int)

    # Encode binary/ordinal features for correlation
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for col in ["gender", "Partner", "Dependents", "PhoneService",
                "PaperlessBilling", "SeniorCitizen"]:
        df_enc[col] = df_enc[col].map(binary_map)

    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    df_enc["Contract_ord"] = df_enc["Contract"].map(contract_map)

    corr_cols = ["tenure", "MonthlyCharges", "TotalCharges",
                 "SeniorCitizen", "Partner", "Dependents",
                 "PaperlessBilling", "Contract_ord", "Churn_bin"]
    corr_labels = ["Tenure", "Monthly\nCharges", "Total\nCharges",
                   "Senior\nCitizen", "Partner", "Dependents",
                   "Paperless\nBilling", "Contract\n(ordinal)", "Churn"]

    corr_matrix = df_enc[corr_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", vmin=-1, vmax=1, center=0,
        xticklabels=corr_labels, yticklabels=corr_labels,
        linewidths=0.5, ax=ax,
        annot_kws={"size": 9}
    )
    ax.set_title("Korrelationsmatrix (numerische & ordinale Features)",
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(f"{OUT}/04_correlation_heatmap.png")
    plt.close()
    print("✓ Plot 4: Korrelations-Heatmap")


    # ══════════════════════════════════════════════════════════════════════════════
    # PLOT 5 – Churn-Rate nach Tenure-Kohorten
    # ══════════════════════════════════════════════════════════════════════════════
    df["tenure_bin"] = pd.cut(df["tenure"],
                              bins=[0, 12, 24, 36, 48, 60, 72],
                              labels=["0–12", "13–24", "25–36", "37–48", "49–60", "61–72"])
    cohort = df.groupby("tenure_bin", observed=True)["Churn"].apply(
        lambda x: (x == "Yes").mean() * 100
    ).reset_index()
    cohort.columns = ["Tenure (Monate)", "Churn-Rate (%)"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(cohort["Tenure (Monate)"], cohort["Churn-Rate (%)"],
                  color=sns.color_palette("RdYlGn_r", len(cohort)),
                  edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, cohort["Churn-Rate (%)"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Churn-Rate nach Tenure-Kohorte", fontsize=13, fontweight="bold")
    ax.set_xlabel("Vertragsdauer (Monate)")
    ax.set_ylabel("Churn-Rate (%)")
    ax.set_ylim(0, cohort["Churn-Rate (%)"].max() * 1.2)
    plt.tight_layout()
    fig.savefig(f"{OUT}/05_churn_by_tenure_cohort.png")
    plt.close()
    print("✓ Plot 5: Churn nach Tenure-Kohorte")


    # ══════════════════════════════════════════════════════════════════════════════
    # BASELINE – Logistische Regression
    # ══════════════════════════════════════════════════════════════════════════════
    df_model = df.drop(columns=["customerID", "tenure_bin"], errors="ignore")
    df_model["Churn"] = (df_model["Churn"] == "Yes").astype(int)
    cat_cols = df_model.select_dtypes("object").columns.tolist()
    df_model = pd.get_dummies(df_model, columns=cat_cols, drop_first=True)

    X = df_model.drop(columns=["Churn"])
    y = df_model["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced",
                                  solver="lbfgs", random_state=42))
    ])
    pipe.fit(X_train, y_train)
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]


    # ══════════════════════════════════════════════════════════════════════════════
    # PLOT 6 – ROC-Kurve
    # ══════════════════════════════════════════════════════════════════════════════
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#E8614C", lw=2.5, label=f"Logistische Regression (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Baseline (AUC = 0.500)")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#E8614C")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-Kurve – Baseline Logistische Regression", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(f"{OUT}/06_roc_curve.png")
    plt.close()
    print("✓ Plot 6: ROC-Kurve")


    # ══════════════════════════════════════════════════════════════════════════════
    # PLOT 7 – Konfusionsmatrix
    # ══════════════════════════════════════════════════════════════════════════════
    cm = confusion_matrix(y_test, y_pred)
    cm_labels = np.array([["TN", "FP"], ["FN", "TP"]])

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", ax=ax,
                linewidths=1, linecolor="white",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.35, cm_labels[i, j],
                    ha="center", va="center", fontsize=11, color="gray")
            ax.text(j + 0.5, i + 0.65, str(cm[i, j]),
                    ha="center", va="center", fontsize=18, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    ax.set_xlabel("Vorhergesagt", fontweight="bold")
    ax.set_ylabel("Tatsächlich", fontweight="bold")
    ax.set_title("Konfusionsmatrix – Testset", fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{OUT}/07_confusion_matrix.png")
    plt.close()
    print("✓ Plot 7: Konfusionsmatrix")


    # ══════════════════════════════════════════════════════════════════════════════
    # PLOT 8 – Feature Importance (Koeffizientenbeträge)
    # ══════════════════════════════════════════════════════════════════════════════
    coef = pipe.named_steps["lr"].coef_[0]
    coef_df = (
        pd.DataFrame({"Feature": X.columns, "Coefficient": coef})
        .assign(Abs=lambda d: d["Coefficient"].abs())
        .sort_values("Abs", ascending=False)
        .head(15)
        .sort_values("Abs")
    )
    colors = ["#E8614C" if c > 0 else "#4C9BE8" for c in coef_df["Coefficient"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors,
                   edgecolor="white", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Koeffizient (standardisiert)")
    ax.set_title("Top 15 Feature-Koeffizienten – Logistische Regression",
                 fontsize=13, fontweight="bold")

    churn_patch   = mpatches.Patch(color="#E8614C", label="↑ Churn-Risiko")
    nochurn_patch = mpatches.Patch(color="#4C9BE8", label="↓ Churn-Risiko")
    ax.legend(handles=[churn_patch, nochurn_patch], loc="lower right")
    plt.tight_layout()
    fig.savefig(f"{OUT}/08_feature_importance.png")
    plt.close()
    print("✓ Plot 8: Feature Importance")


    # ══════════════════════════════════════════════════════════════════════════════
    # PLOT 9 – Metriken-Übersicht (Balkendiagramm)
    # ══════════════════════════════════════════════════════════════════════════════
    metrics = {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall":    recall_score(y_test, y_pred),
        "F1-Score":  f1_score(y_test, y_pred),
        "ROC-AUC":   auc,
    }
    # CV for error bars
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_res = cross_validate(pipe, X_train, y_train, cv=cv,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
    cv_means = {
        "Accuracy":  cv_res["test_accuracy"].mean(),
        "Precision": cv_res["test_precision"].mean(),
        "Recall":    cv_res["test_recall"].mean(),
        "F1-Score":  cv_res["test_f1"].mean(),
        "ROC-AUC":   cv_res["test_roc_auc"].mean(),
    }
    cv_stds = {
        "Accuracy":  cv_res["test_accuracy"].std(),
        "Precision": cv_res["test_precision"].std(),
        "Recall":    cv_res["test_recall"].std(),
        "F1-Score":  cv_res["test_f1"].std(),
        "ROC-AUC":   cv_res["test_roc_auc"].std(),
    }

    names  = list(metrics.keys())
    test_v = list(metrics.values())
    cv_v   = [cv_means[n] for n in names]
    cv_e   = [cv_stds[n]  for n in names]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, test_v, w, label="Testset", color="#4C9BE8",
                edgecolor="white", linewidth=0.8)
    b2 = ax.bar(x + w/2, cv_v,   w, label="CV Mean (5-Fold)", color="#E8A84C",
                yerr=cv_e, capsize=4, edgecolor="white", linewidth=0.8,
                error_kw={"elinewidth": 1.5, "ecolor": "gray"})

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Baseline-Metriken – Logistische Regression (Testset vs. 5-Fold CV)",
                 fontsize=12, fontweight="bold")
    ax.legend()
    ax.axhline(0.5, color="red", linestyle=":", linewidth=1, alpha=0.5, label="Zufalls-Baseline")
    plt.tight_layout()
    fig.savefig(f"{OUT}/09_metrics_overview.png")
    plt.close()
    print("✓ Plot 9: Metriken-Übersicht")

    print(f"\n✅ Alle 9 Plots gespeichert unter: {OUT}/")
