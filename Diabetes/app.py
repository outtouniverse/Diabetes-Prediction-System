import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

data = pd.read_csv("diabetes.csv")

#diabetes_dataset.head()

#number of rows
#diabetes_dataset.shape

#diabetes_dataset.describe()

#diabetes_dataset['Outcome'].value_counts()

#diabetes_dataset.groupby('Outcome').mean()

X = data.drop(columns='Outcome', axis=1)
Y = data['Outcome']

scaler = StandardScaler()
scaler.fit(X)
standard_data = scaler.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(standard_data, Y, test_size=0.2, stratify=Y, random_state=2)

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

st.title("Diabetes Prediction")

st.sidebar.title("Input Features")
pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 3)
glucose = st.sidebar.slider("Glucose", 0, 200, 117)
blood_pressure = st.sidebar.slider("Blood Pressure", 0, 122, 72)
skin_thickness = st.sidebar.slider("Skin Thickness", 0, 99, 23)
insulin = st.sidebar.slider("Insulin", 0, 846, 30)
bmi = st.sidebar.slider("BMI", 0.0, 67.1, 32.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.078, 2.42, 0.3725)
age = st.sidebar.slider("Age", 21, 81, 29)


"""
X_train_predict=classifier.predict(X_train)
train_accuracy=accuracy_score(X_train_predict,Y_train)
print(train_accuracy)

X_test_predict=classifier.predict(X_test)
test_accuracy=accuracy_score(X_test_predict,Y_test)
print(test_accuracy)

"""

input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age)
input_data_asarray = np.asarray(input_data)
input_data_reshape = input_data_asarray.reshape(1, -1)

std_data = scaler.transform(input_data_reshape)

if st.button("Predict"):
    predict = classifier.predict(std_data)
    if predict[0] == 1:
        st.write("<h3 style='text-align: center; color: red;'>The person is likely to have diabetes.</h3>", unsafe_allow_html=True)
    else:
        st.write("<h3 style='text-align: center; color: green;'>The person is not likely to have diabetes.</h3>", unsafe_allow_html=True)