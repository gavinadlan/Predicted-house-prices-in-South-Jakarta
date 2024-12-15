import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
data = pd.read_excel("DATA RUMAH.xlsx")
image = Image.open("cluster.jpg")
st.title("Welcome to the House Price Prediction App")
st.image(image, use_column_width=True)

# Checking the data
st.write("This is an application for knowing how much range of house prices you choose using machine learning. Let's try and see!")
check_data = st.checkbox("See the simple data")
if check_data:
    st.write(data.head())
st.write("Now let's find out how much the prices when we choosing some parameters.")

# Input the numbers
lb = st.slider("Luas Bangunan (LB) dalam m²:", int(data['LB'].min()), int(data['LB'].max()), int(data['LB'].mean()))
lt = st.slider("Luas Tanah (LT) dalam m²:", int(data['LT'].min()), int(data['LT'].max()), int(data['LT'].mean()))
kt = st.slider("Jumlah Kamar Tidur (KT):", int(data['KT'].min()), int(data['KT'].max()), int(data['KT'].mean()))
km = st.slider("Jumlah Kamar Mandi (KM):", int(data['KM'].min()), int(data['KM'].max()), int(data['KM'].mean()))
grs = st.slider("Jumlah Garasi (GRS):", int(data['GRS'].min()), int(data['GRS'].max()), int(data['GRS'].mean()))

# Splitting your data
X = data[['LB', 'LT', 'KT', 'KM', 'GRS']]
y = data['HARGA']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

# Modelling step
# Linear Regression model
model = LinearRegression()
# Fitting and predict your model
model.fit(X_train, y_train)
errors = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
predictions = model.predict([[lb, lt, kt, km, grs]])[0]
akurasi = r2_score(y_test, model.predict(X_test))

# Checking prediction house price
if st.button("Run me!"):
    st.header("Your house prices prediction is IDR {:,.0f}".format(predictions))
    st.subheader("Your range of prediction is IDR {:,.0f} - IDR {:,.0f}".format(predictions-errors, predictions+errors))
    st.subheader("Akurasi : {:.2f}".format(akurasi))
