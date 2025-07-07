import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('Credit Card Defaulter Prediction.csv')
df.columns = df.columns.str.strip()
df['default'] = df['default'].map({'Y': 1, 'N': 0})
df['SEX'] = df['SEX'].map({'F': 0, 'M': 1})
df['EDUCATION'] = LabelEncoder().fit_transform(df['EDUCATION'])
df['MARRIAGE'] = LabelEncoder().fit_transform(df['MARRIAGE'])

X = df[['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'BILL_AMT1', 'PAY_AMT1']]
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

with open('model_rf.pkl', 'wb') as f:
    pickle.dump(model, f)
