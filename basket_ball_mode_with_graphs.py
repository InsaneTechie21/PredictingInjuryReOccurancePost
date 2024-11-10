import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from lifelines import KaplanMeierFitter
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data_path = '/Users/rakshitjoshi/Downloads/BAsketball CSV (new).xlsx'  # Update to your path
data = pd.read_excel(data_path)

# Define target variable and feature set
target_column = 'Preventive_success_of_rehab'
X = data.drop(columns=[target_column, 'Player_ID', 'Date_of_Injury'])
y = data[target_column]

# Encode categorical columns
categorical_features = X.select_dtypes(include='object').columns
for feature in categorical_features:
    encoder = LabelEncoder()
    X[feature] = encoder.fit_transform(X[feature])

# Standardize numerical columns
scaler = StandardScaler()
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predict on test data
y_pred = log_reg.predict(X_test)
y_pred_prob = log_reg.predict_proba(X_test)[:, 1]

# Model evaluation
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 1. Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=['Failure', 'Success'], yticklabels=['Failure', 'Success'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 2. Kaplan-Meier Survival Curve for Rehabilitation
time_durations = data['Rehabilitation_Time_weeks']
events = data['Preventive_success_of_rehab']

kmf = KaplanMeierFitter()
plt.figure(figsize=(8, 6))
kmf.fit(durations=time_durations, event_observed=events)
kmf.plot(ci_show=True)
plt.title("Kaplan-Meier Survival Estimate for Rehab Success")
plt.xlabel("Weeks since Injury")
plt.ylabel("Survival Probability")
plt.show()

# 3. ROC Curve Plotting
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc_score = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# 4. Calibration Curve
plt.figure(figsize=(8, 6))
fraction_positives, mean_preds = calibration_curve(y_test, y_pred_prob, n_bins=10)
plt.plot(mean_preds, fraction_positives, "s-", label="Logistic Regression")
plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Calibration Curve")
plt.legend()
plt.show()

# 5. Histogram of Predicted Probabilities
plt.figure(figsize=(8, 6))
sns.histplot(y_pred_prob, kde=True)
plt.title("Distribution of Predicted Probabilities")
plt.xlabel("Predicted Probability of Rehabilitation Success")
plt.ylabel("Frequency")
plt.show()
