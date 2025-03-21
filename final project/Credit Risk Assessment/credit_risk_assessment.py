import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/tusharranjanpadhi/Drive/Projects/ML_Modles/Credit Risk Assessment/credit_risk_dataset.csv")

imputer = SimpleImputer(strategy='most_frequent')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
categorical_columns = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
data_encoded = pd.get_dummies(data_imputed, columns=categorical_columns, drop_first=True)

if 'person_age' in data_encoded.columns:
    bins = [0, 25, 45, 65, np.inf]
    labels = [0, 1, 2, 3] 
    data_encoded['AgeGroup'] = pd.cut(data_encoded['person_age'], bins=bins, labels=labels)
    data_encoded['AgeGroup'] = data_encoded['AgeGroup'].astype(int)  

if 'loan_status' in data_encoded.columns:
    label_encoder = LabelEncoder()
    data_encoded['loan_status'] = label_encoder.fit_transform(data_encoded['loan_status'])


X = data_encoded.drop(columns=['person_age', 'loan_status'])
y = data_encoded['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf_classifier = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

feature_importances = rf_classifier.feature_importances_
feature_names = X.columns
sorted_idx = np.argsort(feature_importances)

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align="center")
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.show()

def predict_credit_risk(input_df):
    try:
        # Make prediction
        prediction = rf_classifier.predict(input_df)[0]
        risk_level = "Low Risk" if prediction == 0 else "High Risk"
        return risk_level
    except Exception as e:
        print(f"\nError during prediction: {str(e)}")
        raise

if __name__ == "__main__":
    print("Loading model...")
    # Example usage with command line input
    try:
        print("\nEnter the following information (numbers only, no commas or $ signs):")
        
        person_income = float(input("Annual Income (e.g., 60000): ").replace('$', '').replace(',', ''))
        person_age = int(input("Age (e.g., 35): "))
        person_emp_length = float(input("Employment Length in years (e.g., 8): "))
        loan_amnt = float(input("Loan Amount (e.g., 150000): ").replace('$', '').replace(',', ''))
        loan_int_rate = float(input("Interest Rate (e.g., 7.5): ").replace('%', ''))
        credit_hist_length = float(input("Credit History Length in years (e.g., 10): "))
        
        user_data = {
            'person_income': person_income,
            'person_age': person_age,
            'person_emp_length': person_emp_length,
            'loan_amnt': loan_amnt,
            'loan_int_rate': loan_int_rate,
            'loan_percent_income': (loan_amnt / person_income) * 100,
            'cb_person_cred_hist_length': credit_hist_length
        }
        
        input_df = pd.DataFrame([user_data])
        risk_level = predict_credit_risk(input_df)
        print("\nRisk Assessment Result:")
        print(f"Based on the provided information, the applicant is: {risk_level}")
        
    except ValueError as ve:
        print("\nError: Please enter numbers only, without any symbols ($ , %).")
        print("Example inputs:")
        print("Annual Income: 60000")
        print("Age: 35")
        print("Employment Length: 8")
        print("Loan Amount: 150000")
        print("Interest Rate: 7.5")
        print("Credit History Length: 10")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")