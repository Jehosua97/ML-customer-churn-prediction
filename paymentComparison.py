# %% [markdown]
# # Telco Customer Churn + E-waste
# ## Original vs Improved Logistic Regression + Contract Type Analysis
#
# - Original LR: C=200, class_weight=None, threshold=0.50
# - Improved LR: Tuned C, tuned class_weight, tuned threshold (max accuracy)
# - E-waste: E = (M · N) / L (kg/year)
# - Contract analysis: Month-to-month vs Yearly churn & e-waste

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")
plt.rcParams["figure.figsize"] = (6, 4)

# %% [markdown]
# ## 1. Load dataset

# %%
DATA_PATH = "Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(DATA_PATH)
df.head()

# %% [markdown]
# ### Target distribution (Churn vs No Churn)

# %%
plt.figure()
sns.countplot(x="Churn", data=df, palette="crest")
plt.title("Original Churn Distribution")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Basic cleaning

# %%
# Convert TotalCharges to numeric and fill NaNs
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].mean(), inplace=True)

# Replace "No internet service" / "No phone service" with "No"
df.replace(["No internet service", "No phone service"], "No", inplace=True)

# Encode target variable (Yes -> 1, No -> 0)
df["Churn"] = (df["Churn"] == "Yes").astype(int)

# %% [markdown]
# ## 3. Feature engineering (tenure bins, log transform)

# %%
# Tenure bins -> 6 categories (0..5)
conditions = [
    (df.tenure <= 12),
    (df.tenure > 12) & (df.tenure <= 24),
    (df.tenure > 24) & (df.tenure <= 36),
    (df.tenure > 36) & (df.tenure <= 48),
    (df.tenure > 48) & (df.tenure <= 60),
    (df.tenure > 60),
]
choices = [0, 1, 2, 3, 4, 5]
df["tenure_range"] = np.select(conditions, choices)

# Log transform skewed numeric features
df["MonthlyCharges"] = np.log1p(df["MonthlyCharges"])
df["TotalCharges"] = np.log1p(df["TotalCharges"])

# Visualize tenure_range vs churn
plt.figure()
sns.countplot(x="tenure_range", hue="Churn", data=df, palette="crest")
plt.title("Churn by Tenure Range")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. One-hot encoding (all categorical features)

# %%
# All object columns except customerID are treated as categorical
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
if "customerID" in cat_cols:
    cat_cols.remove("customerID")

df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)
df_encoded.head()

# %% [markdown]
# ## 5. Train / Test split

# %%
X = df_encoded.drop(columns=["customerID", "Churn"])
y = df_encoded["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train.shape, X_test.shape

# %% [markdown]
# ## 6. Handle imbalance with SMOTE

# %%
sm = SMOTE(random_state=0)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

plt.figure()
sns.countplot(x=y_train_res, palette="crest")
plt.title("Class Distribution After SMOTE")
plt.xticks([0, 1], ["No churn", "Churn"])
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. E-waste functions: E = M · N / L

# %%
DEVICE_MASS_KG = 0.8          # M: mass (kg) per device
DEVICES_PER_CUSTOMER = 1      # how many devices per churned customer
DEVICE_LIFETIME_YEARS = 5     # L: average lifetime in years

def estimate_ewaste_mn_over_l(device_weight, num_devices, lifetime_years):
    """
    E = (M * N) / L
    Returns e-waste in kg/year.
    """
    if lifetime_years <= 0:
        raise ValueError("Lifetime (L) must be > 0.")
    return (device_weight * num_devices) / lifetime_years

def ewaste_from_churn_predictions(
    y_pred,
    device_mass_kg=DEVICE_MASS_KG,
    devices_per_customer=DEVICES_PER_CUSTOMER,
    lifetime_years=DEVICE_LIFETIME_YEARS,
):
    """
    Map churn predictions (0/1) to E-waste using E = M * N / L.
    y_pred: 1D array-like of 0/1 churn predictions
    """
    num_churn_customers = int(np.sum(y_pred))
    num_devices = num_churn_customers * devices_per_customer
    ewaste_kg_per_year = estimate_ewaste_mn_over_l(
        device_mass_kg, num_devices, lifetime_years
    )
    return ewaste_kg_per_year, num_churn_customers, num_devices

# quick sanity check
example_E = estimate_ewaste_mn_over_l(0.8, 1000, 5)
print(f"Example E-waste: {example_E:.2f} kg/year")

# %% [markdown]
# Helper to plot confusion matrices

# %%
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="crest")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title(title)
    plt.xticks([0.5, 1.5], ["No churn", "Churn"])
    plt.yticks([0.5, 1.5], ["No churn", "Churn"], rotation=0)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 8. ORIGINAL Logistic Regression (C=200, threshold 0.5)

# %%
print("\n================ ORIGINAL LOGISTIC REGRESSION ================")

orig_log_model = LogisticRegression(
    C=200,
    max_iter=1000,
    solver="liblinear",       # stable with many one-hot features
    class_weight=None
)

orig_log_model.fit(X_train_res, y_train_res)

# Default threshold = 0.5
orig_proba = orig_log_model.predict_proba(X_test)[:, 1]
orig_pred = (orig_proba >= 0.5).astype(int)

print("\nClassification report (Original LogReg):")
print(classification_report(y_test, orig_pred, target_names=["No churn", "Churn"]))

plot_confusion_matrix(y_test, orig_pred, "Confusion Matrix – Original Logistic Regression")

orig_ewaste_kg, orig_customers, orig_devices = ewaste_from_churn_predictions(orig_pred)

orig_acc = accuracy_score(y_test, orig_pred)
orig_prec = precision_score(y_test, orig_pred)
orig_rec = recall_score(y_test, orig_pred)
orig_f1 = f1_score(y_test, orig_pred)
orig_auc = roc_auc_score(y_test, orig_proba)

print("\nE-waste estimate (Original LogReg):")
print(f"Predicted churned customers: {orig_customers}")
print(f"Associated devices:         {orig_devices}")
print(f"E-waste:                    {orig_ewaste_kg:.2f} kg/year "
      f"({orig_ewaste_kg/1000:.3f} tons/year)")

# %% [markdown]
# ## 9. IMPROVED Logistic Regression (tuned hyperparameters + threshold)

# %%
print("\n================ IMPROVED LOGISTIC REGRESSION ================")

log_base = LogisticRegression(
    max_iter=1000,
    solver="liblinear",  # supports class_weight and works well here
)

param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "class_weight": [None, "balanced"],
}

log_grid = GridSearchCV(
    estimator=log_base,
    param_grid=param_grid,
    scoring="accuracy",   # optimize for accuracy
    cv=5,
    n_jobs=-1,
    verbose=0,
)

log_grid.fit(X_train_res, y_train_res)

print("Best Logistic Regression params:", log_grid.best_params_)
best_log_model = log_grid.best_estimator_

# Get probabilities on test set
imp_proba = best_log_model.predict_proba(X_test)[:, 1]

# Threshold tuning: maximize accuracy
thresholds = np.arange(0.1, 0.91, 0.01)
best_acc = -1
best_acc_thr = 0.5

for thr in thresholds:
    y_thr = (imp_proba >= thr).astype(int)
    acc = accuracy_score(y_test, y_thr)
    if acc > best_acc:
        best_acc = acc
        best_acc_thr = thr

print(f"\nBest accuracy threshold: {best_acc_thr:.2f} → accuracy = {best_acc:.4f}")

imp_pred = (imp_proba >= best_acc_thr).astype(int)

print("\nClassification report (Improved LogReg – tuned + threshold):")
print(classification_report(y_test, imp_pred, target_names=["No churn", "Churn"]))

plot_confusion_matrix(y_test, imp_pred, "Confusion Matrix – Improved Logistic Regression")

imp_ewaste_kg, imp_customers, imp_devices = ewaste_from_churn_predictions(imp_pred)

imp_acc = accuracy_score(y_test, imp_pred)
imp_prec = precision_score(y_test, imp_pred)
imp_rec = recall_score(y_test, imp_pred)
imp_f1 = f1_score(y_test, imp_pred)
imp_auc = roc_auc_score(y_test, imp_proba)

print("\nE-waste estimate (Improved LogReg):")
print(f"Predicted churned customers: {imp_customers}")
print(f"Associated devices:         {imp_devices}")
print(f"E-waste:                    {imp_ewaste_kg:.2f} kg/year "
      f"({imp_ewaste_kg/1000:.3f} tons/year)")

# %% [markdown]
# ## 10. Comparison table: Original vs Improved Logistic Regression

# %%
lr_compare_df = pd.DataFrame([
    {
        "Model": "Original Logistic Regression",
        "C": 200,
        "class_weight": "None",
        "Threshold": 0.50,
        "Accuracy": orig_acc,
        "Precision": orig_prec,
        "Recall": orig_rec,
        "F1": orig_f1,
        "ROC_AUC": orig_auc,
        "Ewaste_kg_per_year": orig_ewaste_kg,
        "Ewaste_tons_per_year": orig_ewaste_kg / 1000.0,
        "Predicted_churn_customers": orig_customers
    },
    {
        "Model": "Improved Logistic Regression",
        "C": log_grid.best_params_["C"],
        "class_weight": str(log_grid.best_params_["class_weight"]),
        "Threshold": best_acc_thr,
        "Accuracy": imp_acc,
        "Precision": imp_prec,
        "Recall": imp_rec,
        "F1": imp_f1,
        "ROC_AUC": imp_auc,
        "Ewaste_kg_per_year": imp_ewaste_kg,
        "Ewaste_tons_per_year": imp_ewaste_kg / 1000.0,
        "Predicted_churn_customers": imp_customers
    }
])

lr_compare_df.set_index("Model", inplace=True)
print("\n=== Logistic Regression Comparison (Original vs Improved) ===")
print(lr_compare_df.round(4))

# If you're in Jupyter, the table will render nicely:
lr_compare_df.round(4)

# %% [markdown]
# ## 11. Contract Type Analysis – Month-to-month vs Yearly Churn & E-waste

# %%
# Work with original df (after cleaning and encoding Churn but before one-hot)
df_contract = df[["Contract", "Churn"]].copy()

# Map contract types into Month-to-month vs Yearly (One year + Two year)
df_contract["Contract_Type"] = df_contract["Contract"].replace({
    "Month-to-month": "Month-to-month",
    "One year": "Yearly",
    "Two year": "Yearly"
})

# Churn rate by contract type (actual labels)
contract_churn = df_contract.groupby("Contract_Type")["Churn"].mean().reset_index()

print("\n=== Churn Rate by Contract Type (Actual) ===")
print(contract_churn)

plt.figure()
sns.barplot(x="Contract_Type", y="Churn", data=contract_churn, palette="crest")
plt.title("Actual Churn Rate by Contract Type")
plt.ylabel("Churn Rate")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### E-waste based on contract type using improved Logistic Regression predictions

# %%
# Predict churn on full dataset using improved LR model
full_proba = best_log_model.predict_proba(X)[:, 1]
full_pred = (full_proba >= best_acc_thr).astype(int)

# Attach predictions to contract DataFrame (indices align with df/X)
df_contract["Predicted_Churn"] = full_pred

# Count churned customers by contract type (predictions)
churn_counts = df_contract.groupby("Contract_Type")["Predicted_Churn"].sum()
print("\nPredicted churn counts by contract type:")
print(churn_counts)

# Compute e-waste for each contract type
ewaste_contract = {}

for ct in churn_counts.index:
    N = churn_counts[ct]  # number of churn customers
    ew = estimate_ewaste_mn_over_l(
        DEVICE_MASS_KG,
        N * DEVICES_PER_CUSTOMER,
        DEVICE_LIFETIME_YEARS
    )
    ewaste_contract[ct] = ew

# Convert to DataFrame
ewaste_contract_df = pd.DataFrame({
    "Contract_Type": list(ewaste_contract.keys()),
    "Ewaste_kg_per_year": list(ewaste_contract.values())
})

print("\n=== E-waste by Contract Type (Predicted) ===")
print(ewaste_contract_df)

# Plot e-waste by contract type
plt.figure()
sns.barplot(
    x="Contract_Type",
    y="Ewaste_kg_per_year",
    data=ewaste_contract_df,
    palette="crest"
)
plt.title("E-waste Generation by Contract Type (Predicted)")
plt.ylabel("E-waste (kg/year)")
plt.tight_layout()
plt.show()

# Merge churn + e-waste into one summary table
contract_summary_df = contract_churn.merge(
    ewaste_contract_df,
    on="Contract_Type",
    how="inner"
)

print("\n=== Contract-Type Churn + E-waste Summary ===")
print(contract_summary_df.round(4))

# If running in Jupyter, show nicely:
contract_summary_df.round(4)
