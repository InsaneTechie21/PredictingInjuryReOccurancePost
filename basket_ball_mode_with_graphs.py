import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from lifelines import KaplanMeierFitter
from sklearn.calibration import calibration_curve

# Load the data
file_path = '/Users/rakshitjoshi/Downloads/BAsketball CSV (new).xlsx'  # Replace with your file path
data = pd.read_excel(file_path)

# Select target and features
target = 'Preventive_success_of_rehab'
features = data.drop(columns=['Preventive_success_of_rehab', 'Player_ID', 'Date_of_Injury'])

# Encode categorical variables
categorical_cols = features.select_dtypes(include='object').columns
for col in categorical_cols:
    le = LabelEncoder()
    features[col] = le.fit_transform(features[col])

# Standardize numerical features
scaler = StandardScaler()
numerical_cols = features.select_dtypes(include=['int64', 'float64']).columns
features[numerical_cols] = scaler.fit_transform(features[numerical_cols])

# Prepare target variable and split data
X = features
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy}")
print("Classification Report:\n", classification_rep)

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['Failure', 'Success'], yticklabels=['Failure', 'Success'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 2. Kaplan-Meier Survival Curve for Rehabilitation Success
time_to_event = data['Rehabilitation_Time_weeks']
event_occurred = data['Preventive_success_of_rehab']

kmf = KaplanMeierFitter()
plt.figure(figsize=(8, 6))
kmf.fit(durations=time_to_event, event_observed=event_occurred)
kmf.plot(ci_show=True)
plt.title("Kaplan-Meier Curve for Rehabilitation Success")
plt.xlabel("Weeks since Injury")
plt.ylabel("Survival Probability")
plt.show()

# 3. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
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
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_prob, n_bins=10)
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Logistic Regression")
plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Calibration Curve")
plt.legend()
plt.show()

# 5. Distribution of Predicted Probabilities
plt.figure(figsize=(8, 6))
sns.histplot(y_prob, kde=True)
plt.title("Distribution of Predicted Probabilities for Rehabilitation Success")
plt.xlabel("Predicted Probability of Success")
plt.ylabel("Frequency")
plt.show()
