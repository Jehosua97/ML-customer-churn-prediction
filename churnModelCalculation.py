import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")

# Make plots a bit bigger
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
# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].mean(), inplace=True)

# Replace "No internet service" / "No phone service" with "No"
df.replace(["No internet service", "No phone service"], "No", inplace=True)

# Encode target
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

# Log transform some numeric features
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
plt.title("Class distribution after SMOTE")
plt.xticks([0, 1], ["No churn", "Churn"])
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. E-waste functions: E = M * N / L

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

# dictionary to store e-waste per model
ewaste_results = {}

# %% [markdown]
# ## 8. Logistic Regression

# %%
print("\n================ LOGISTIC REGRESSION ================")
log_model = LogisticRegression(C=200, max_iter=1000)
log_model.fit(X_train_res, y_train_res)

log_pred = log_model.predict(X_test)

print("\nClassification report (LogReg):")
print(classification_report(y_test, log_pred, target_names=["No churn", "Churn"]))

plot_confusion_matrix(y_test, log_pred, "Confusion Matrix – Logistic Regression")

log_ewaste_kg, log_customers, log_devices = ewaste_from_churn_predictions(log_pred)
ewaste_results["Logistic Regression"] = log_ewaste_kg

print("\nE-waste estimate (LogReg):")
print(f"Predicted churned customers: {log_customers}")
print(f"Associated devices:         {log_devices}")
print(f"E-waste:                    {log_ewaste_kg:.2f} kg/year "
      f"({log_ewaste_kg/1000:.3f} tons/year)")

# %% [markdown]
# ## 9. SVM (Linear)

# %%
print("\n================ SVM (LINEAR) ================")
svm_model = SVC(kernel="linear", C=20)
svm_model.fit(X_train_res, y_train_res)

svm_pred = svm_model.predict(X_test)

print("\nClassification report (SVM):")
print(classification_report(y_test, svm_pred, target_names=["No churn", "Churn"]))

plot_confusion_matrix(y_test, svm_pred, "Confusion Matrix – SVM (Linear)")

svm_ewaste_kg, svm_customers, svm_devices = ewaste_from_churn_predictions(svm_pred)
ewaste_results["SVM (Linear)"] = svm_ewaste_kg

print("\nE-waste estimate (SVM):")
print(f"Predicted churned customers: {svm_customers}")
print(f"Associated devices:         {svm_devices}")
print(f"E-waste:                    {svm_ewaste_kg:.2f} kg/year "
      f"({svm_ewaste_kg/1000:.3f} tons/year)")

# %% [markdown]
# ## 10. XGBoost

# %%
print("\n================ XGBOOST ================")

params = {
    "min_child_weight": [1, 5, 10],
    "gamma": [1.5, 2, 5, 7],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "max_depth": [4, 5, 8],
}

xgb_base = xgb.XGBClassifier(
    learning_rate=0.08,
    n_estimators=100,
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

random_search = RandomizedSearchCV(
    xgb_base,
    param_distributions=params,
    n_iter=5,
    scoring="roc_auc",
    cv=skf.split(X_train_res, y_train_res),
    random_state=0,
    verbose=0,
)

random_search.fit(X_train_res, y_train_res)

xgb_pred = random_search.predict(X_test)

print("\nClassification report (XGBoost):")
print(classification_report(y_test, xgb_pred, target_names=["No churn", "Churn"]))

plot_confusion_matrix(y_test, xgb_pred, "Confusion Matrix – XGBoost")

xgb_ewaste_kg, xgb_customers, xgb_devices = ewaste_from_churn_predictions(xgb_pred)
ewaste_results["XGBoost"] = xgb_ewaste_kg

print("\nE-waste estimate (XGBoost):")
print(f"Predicted churned customers: {xgb_customers}")
print(f"Associated devices:         {xgb_devices}")
print(f"E-waste:                    {xgb_ewaste_kg:.2f} kg/year "
      f"({xgb_ewaste_kg/1000:.3f} tons/year)")

# %% [markdown]
# ## 11. MLP (Neural Network)

# %%
print("\n================ MLP (Neural Network) ================")
mlp_model = MLPClassifier(random_state=1, max_iter=500)
mlp_model.fit(X_train_res, y_train_res)

mlp_pred = mlp_model.predict(X_test)

print("\nClassification report (MLP):")
print(classification_report(y_test, mlp_pred, target_names=["No churn", "Churn"]))

plot_confusion_matrix(y_test, mlp_pred, "Confusion Matrix – MLP")

mlp_ewaste_kg, mlp_customers, mlp_devices = ewaste_from_churn_predictions(mlp_pred)
ewaste_results["MLP"] = mlp_ewaste_kg

print("\nE-waste estimate (MLP):")
print(f"Predicted churned customers: {mlp_customers}")
print(f"Associated devices:         {mlp_devices}")
print(f"E-waste:                    {mlp_ewaste_kg:.2f} kg/year "
      f"({mlp_ewaste_kg/1000:.3f} tons/year)")

# %% [markdown]
# ## 12. Compare E-waste across models (bar chart)

# %%
models = list(ewaste_results.keys())
ewaste_values = [ewaste_results[m] for m in models]

plt.figure(figsize=(7, 4))
sns.barplot(x=models, y=ewaste_values, palette="crest")
plt.ylabel("E-waste (kg/year)")
plt.title("E-waste estimate per model (E = M·N / L)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

print("\nE-waste summary (kg/year):")
for m, v in ewaste_results.items():
    print(f"{m}: {v:.2f} kg/year ({v/1000:.3f} tons/year)")
