# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from pandas import DataFrame
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import seaborn as sns
# import pickle #
import joblib

st.header("Klasifikasi Sel Kanker Ganas menggunakan metode logistic regresion")

selected = option_menu(
    menu_title=None,  # wajib ada
    options=["Dataset", "Prepocessing", "Model"],
    icons=["book", "cast", "envelope"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#ffffff", },
        "icons": {"font-size": "14px"},
        "nav-link": {"font-size": "15px",
                     "text-align": "center",
                     "margin": "0px",
                     "--hover-color": "#eee",
                     },
    }
)


def load_data():
    pd_crs = pd.read_csv("Cancer_Data.csv")
    return pd_crs


# Hanya akan di run sekali
pd_crs = load_data()

if selected == "Dataset":
    st.write('''#### Dataset''')
    st.write(pd_crs)
    st.write("""
    Data yang dianamisis adalah data tentang sel kanker ganas dan sel kanker jinak.
    """)

    st.write('''#### Fitur-fitur pada dataset''')
    st.write(
        "Pada dataset ini terdiri sebanyak 569 data dengan 32 fitur. Adapun fitur-fiturnya yaitu:")
    st.info('''
    1. id (Id dari pasien)
    2. diagnosis ((Kanker Jinak-B)(Kanker Ganas-M))
    ''')

    st.write('\n')
    st.write('\n')

if selected == "Prepocessing":

    st.write("#### Data sebelum preprosessing")
    st.write(pd_crs.iloc[:20])

    # Menghapus kolom yang kosong
pd_crs.drop('Unnamed: 32', axis=1, inplace=True)

# Mengecek data redundan
redundant_data = pd_crs.duplicated()

jumlah_redundan = redundant_data.sum()

if jumlah_redundan > 0:
    print("jumlah data redundan:", jumlah_redundan)
    redundant_rows = pd_crs[redundant_data]
    print("Baris-baris data redundan:")
    print(redundant_rows)
else:
    print("Tidak ada data redundan dalam dataset.")

# mengecek tanda baca
kolom_pemeriksaan = 'diagnosis'

tanda_baca = pd_crs[kolom_pemeriksaan].str.contains(r'[^\w\s]').any()

if tanda_baca:
    print(f"Tanda baca ditemukan dalam kolom {kolom_pemeriksaan}.")
else:
    print(f"Tidak ada tanda baca dalam kolom {kolom_pemeriksaan}.")


data = pd_crs.drop('id', axis=1)

# pemisahan fitur x dan y
x = data.drop('diagnosis', axis=1)
y = data[['diagnosis']]

# split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42)


# Label Encode
# LE = LabelEncoder()
label_encoder = preprocessing.LabelEncoder()
data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])
data['diagnosis'].unique()

st.write("#### Data setelah normalisasi")
st.write(data.iloc[:20])


if selected == "Model":

    model = joblib.load('linear_regression.joblib')
    st.title(
        "Data mining prediksi klasifikasi Kanker ganas menggunakan metode Logistic Regression")

    radius_mean = st.number_input("input nilai radius mean")
    texture_mean = st.number_input("input nilai texture mean")
    perimeter_mean = st.number_input("input nilai perimeter mean")
    area_mean = st.number_input("input nilai area mean")
    smoothness_mean = st.number_input("input nilai smoothness mean")
    compactness_mean = st.number_input("input nilai compactness mean")
    concavity_mean = st.number_input("input nilai concavity mean")
    concave_points_mean = st.number_input("input nilai concave points mean")
    symmetry_mean = st.number_input("input nilai symmetry mean")
    fractal_dimension_mean = st.number_input("input nilai fractal dimension mean")
    radius_se = st.number_input("input nilai radius se")
    texture_se = st.number_input("input nilai texture se")
    perimeter_se = st.number_input("input nilai perimeter se")
    area_se = st.number_input("input nilai area se")
    smoothness_se = st.number_input("input nilai smoothness se")
    compactness_se = st.number_input("input nilai compactness se")
    concavity_se = st.number_input("input nilai concavity se")
    concave_points_se = st.number_input("input nilai concave_points_se")
    symmetry_se = st.number_input("input nilai symmetry_se")
    fractal_dimension_se = st.number_input("input nilai fractal_dimension_se")
    radius_worst = st.number_input("input nilai radius_worst")
    texture_worst = st.number_input("input nilai texture_worst")
    perimeter_worst = st.number_input("input nilai perimeter_worst")
    area_worst = st.number_input("input nilai area_worst")
    smoothness_worst = st.number_input("input nilai smoothness_worst")
    compactness_worst = st.number_input("input nilai compactness_worst")
    concavity_worst = st.number_input("input nilai concavity_worst")
    concave_points_worst = st.number_input("input nilai concave_points_worst")
    symmetry_worst = st.number_input("input nilai symmetry_worst")
    fractal_dimension_worst = st.number_input("input nilai fractal_dimension_worst")

    # code prediksi
    kanker_diagnosis = ''

    # tombol untuk prediksi
    if st.button('Test Prediksi Kanker'):

        predict_input = pd.DataFrame({
            'radius_mean': [radius_mean],
            'texture_mean': [texture_mean],
            'perimeter_mean': [perimeter_mean],
            'area_mean': [area_mean],
            'smoothness_mean': [smoothness_mean],
            'compactness_mean': [compactness_mean],
            'concavity_mean': [concavity_mean],
            'concave_points_mean': [concave_points_mean],
            'symmetry_mean': [symmetry_mean],
            'fractal_dimension_mean': [fractal_dimension_mean],
            'radius_se': [radius_se],
            'texture_se': [texture_se],
            'perimeter_se': [perimeter_se],
            'area_se': [area_se],
            'smoothness_se': [smoothness_se],
            'compactness_se': [compactness_se],
            'concavity_se': [concavity_se],
            'concave_points_se': [concave_points_se],
            'symmetry_se': [symmetry_se],
            'fractal_dimension_se': [fractal_dimension_se],
            'radius_worst': [radius_worst],
            'texture_worst': [texture_worst],
            'perimeter_worst': [perimeter_worst],
            'area_worst': [area_worst],
            'smoothness_worst': [smoothness_worst],
            'compactness_worst': [compactness_worst],
            'concavity_worst': [concavity_worst],
            'concave_points_worst': [concave_points_worst],
            'symmetry_worst': [symmetry_worst],
            'fractal_dimension_worst': [fractal_dimension_worst],
        })
        # kanker_prediksi = model.predict([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
        #                                         radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
        #                                        radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst ]])
        kanker_prediksi = model.predict(predict_input)

        if kanker_prediksi == 'M':
            st.error("Pasien Terindikasi Kanker Ganas")
        elif kanker_prediksi == 'B':
            st.success("Pasien Terindikasi Kanker Jinak")
