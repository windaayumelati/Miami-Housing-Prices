import pickle
import streamlit as st

model = pickle.load(open('Miami.sav', 'rb'))

st.title('Estimasi Harga House Miami')

LATITUDE = st.number_input('Masukan Nilai Bujur Geografis Lokasi Properti')
LONGITUDE = st.number_input('Masukan Nilai Lintang Geografis Lokasi Properti')
PARCELNO = st.number_input('Masukan identifikasi unik untuk setiap properti')
LND_SQFOOT = st.number_input('Masukan Luas Tanah Dalam Kaki Persegi (square feet)')
TOT_LVG_AREA = st.number_input('Masukan Luas Lantai Properti Dalam Kaki Persegi (square feet)')
SPEC_FEAT_VAL = st.number_input('Masukan Nilai Fitur Khusus Pada Properti ($)')
RAIL_DIST = st.number_input('Masukkan Jarak Ke Jalur Rel Terdekat (Feet)')
OCEAN_DIST = st.number_input('Masukkan Jarak Ke Laut (Feet)')
WATER_DIST = st.number_input('Masukkan Jarak Ke Badan Air Terdekat (Feet)')
CNTR_DIST = st.number_input('Masukkan Jarak Ke Distrik Bisnis Pusat Miami (Feet)')
SUBCNTR_DI = st.number_input('Masukkan Jarak Ke Subcenter Terdekat (Feet)')
HWY_DIST = st.number_input('Masukkan Jarak ke Jalan Raya Terdekat (Indikator Kebisingan)(Feet)')
age = st.number_input('Masukkan Usia Struktur/Bangunan')
avno60plus = st.number_input('Masukkan Tingkat Kebisingan Pesawat')
month_sold = st.number_input('Masukkan Bulan Penjualan Properti:', min_value=1, max_value=12, step=1)
structure_quality = st.number_input('Masukan Kualitas struktur')

predict = ''

if st.button('Estimasi Harga House MIAMI'):
    features = [[LATITUDE, LONGITUDE, PARCELNO, LND_SQFOOT, TOT_LVG_AREA, SPEC_FEAT_VAL, RAIL_DIST, OCEAN_DIST,
                 WATER_DIST, CNTR_DIST, SUBCNTR_DI, HWY_DIST, age, avno60plus, month_sold, structure_quality]]
    predict = model.predict(features)
    st.write('Estimasi Harga House MIAMI (USD): ', predict[0])
    st.write('Estimasi Harga House MIAMI Dalam (IDR): ', predict*19000)
    
