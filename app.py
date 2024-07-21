import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
@st.cache
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    data = pd.read_csv(url, names=columns)
    data.replace('?', np.nan, inplace=True)
    data.dropna(inplace=True)
    data = data.astype(float)
    return data

# Load data
data = load_data()

# Preprocess the data
X = data.drop('target', axis=1)
y = data['target']
y = y.apply(lambda x: 1 if x > 0 else 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Streamlit app
st.title("Heart Disease Prediction")

st.sidebar.header("Input Parameters")
def user_input_features():
    age = st.sidebar.slider("Age", int(data.age.min()), int(data.age.max()), int(data.age.mean()))
    sex = st.sidebar.selectbox("Sex", ("Male", "Female"))
    sex = 1 if sex == "Male" else 0
    cp = st.sidebar.slider("Chest Pain Type (cp)", int(data.cp.min()), int(data.cp.max()), int(data.cp.mean()))
    trestbps = st.sidebar.slider("Resting Blood Pressure (trestbps)", int(data.trestbps.min()), int(data.trestbps.max()), int(data.trestbps.mean()))
    chol = st.sidebar.slider("Serum Cholestoral in mg/dl (chol)", int(data.chol.min()), int(data.chol.max()), int(data.chol.mean()))
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", ("True", "False"))
    fbs = 1 if fbs == "True" else 0
    restecg = st.sidebar.slider("Resting Electrocardiographic Results (restecg)", int(data.restecg.min()), int(data.restecg.max()), int(data.restecg.mean()))
    thalach = st.sidebar.slider("Maximum Heart Rate Achieved (thalach)", int(data.thalach.min()), int(data.thalach.max()), int(data.thalach.mean()))
    exang = st.sidebar.selectbox("Exercise Induced Angina (exang)", ("Yes", "No"))
    exang = 1 if exang == "Yes" else 0
    oldpeak = st.sidebar.slider("ST Depression Induced by Exercise (oldpeak)", float(data.oldpeak.min()), float(data.oldpeak.max()), float(data.oldpeak.mean()))
    slope = st.sidebar.slider("Slope of the Peak Exercise ST Segment (slope)", int(data.slope.min()), int(data.slope.max()), int(data.slope.mean()))
    ca = st.sidebar.slider("Number of Major Vessels (ca)", int(data.ca.min()), int(data.ca.max()), int(data.ca.mean()))
    thal = st.sidebar.slider("Thal", int(data.thal.min()), int(data.thal.max()), int(data.thal.mean()))
    
    data_input = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    features = pd.DataFrame(data_input, index=[0])
    return features

input_df = user_input_features()

# Preprocessing user input
input_scaled = scaler.transform(input_df)

# Predictions
prediction = clf.predict(input_scaled)
prediction_proba = clf.predict_proba(input_scaled)

st.subheader("Prediction")
heart_disease = np.array(["Healthy Heart less chance of Heart Disease", "Chances of Heart Disease Visit Doctor"])
st.write(heart_disease[prediction][0])


prediction_proba = 0.7  # Replace with your actual prediction probability

threshold = 0.5  # You can adjust this threshold based on your model's confidence

if prediction_proba >= threshold:
    st.subheader("Prediction Probability: Yes")
    st.write(prediction_proba)
else:
    st.subheader("Prediction Probability: No")
    st.write(prediction_proba)
