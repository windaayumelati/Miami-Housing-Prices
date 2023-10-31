import pickle
import streamlit as st

model = pickle.load(open('Miami.sav', 'rb'))

st.title('Estimasi Harga House Miami')

LATITUDE = st.number_input('Masukan Harga Tanah')
LONGITUDE = st.number_input('Masukan Luas Tanah')
PARCELNO = st.number_input('Masukan Harga Jual')
LND_SQFOOT = st.number_input('Masukan Harga Pasar')
TOT_LVG_AREA = st.number_input('Masukan nomer Rumah')
SPEC_FEAT_VAL = st.number_input('Masukan Nomer telpon rumah')
RAIL_DIST = st.number_input('Masukkan jarak ke jalur rel terdekat')
OCEAN_DIST = st.number_input('Masukkan jarak ke laut')
WATER_DIST = st.number_input('Masukkan jarak ke badan air terdekat')
CNTR_DIST = st.number_input('Masukkan jarak ke distrik bisnis pusat Miami')
SUBCNTR_DI = st.number_input('Masukkan jarak ke subcenter terdekat')
HWY_DIST = st.number_input('Masukkan jarak ke jalan raya terdekat (indikator kebisingan)')
age = st.number_input('Masukkan usia struktur')
avno60plus = st.number_input('Masukkan tingkat kebisingan pesawat')
month_sold = st.number_input('Msukkan bulan penjualan')
structure_quality = st.number_input('Masukan Kualitas struktur')

predict = ''

if st.button('Estimasi Harga House MIAMI'):
    features = [[LATITUDE, LONGITUDE, PARCELNO, LND_SQFOOT, TOT_LVG_AREA, SPEC_FEAT_VAL, RAIL_DIST, OCEAN_DIST,
                 WATER_DIST, CNTR_DIST, SUBCNTR_DI, HWY_DIST, age, avno60plus, month_sold, structure_quality]]
    predict = model.predict(features)
    st.write('Estimasi Harga House MIAMI: ', predict[0])
    st.write('Estimasi Harga House MIAMI Dalam IDR (Juta): ', predict*19000)
    
