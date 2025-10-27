from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder
import joblib
import pandas as pd

df = pd.read_csv('student_data_cleaned.csv')
cat_cols = ['gender', 'department', 'parental_education']
encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
encoded_cats = encoder.fit_transform(df[cat_cols])
encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(cat_cols))
df_encoded = pd.concat([df.drop(cat_cols, axis=1), encoded_df], axis=1)
X = df_encoded.drop(['student_id', 'dropout'], axis=1)
y = df_encoded['dropout']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
importances = model.feature_importances_
importance_df = pd.DataFrame({'feature': X.columns, 'importance': importances}).sort_values('importance', ascending=False)
print(importance_df.head(10))
joblib.dump(model, 'student_risk_model.joblib')
joblib.dump(encoder, 'encoder.joblib')
df_encoded['risk_score'] = model.predict_proba(X)[:, 1]
df_encoded.to_csv('student_data_with_predictions.csv', index=False)