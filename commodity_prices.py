import pickle
import streamlit as st

# Memuat model yang telah disimpan
model = pickle.load(open('commodity_prices.sav', 'rb'))

# Judul aplikasi Streamlit
st.title('Estimasi Tahun Berdasarkan Harga Komoditas')

# Kolom input untuk harga komoditas
Cocoa = st.number_input('Harga Cocoa/Cocoa Price ($/Kg)')
Coffee = st.number_input('Harga Kopi/Coffee Price ($/Kg)')
Tea = st.number_input('Harga Teh/Tea Price ($/Kg)')
Rice = st.number_input('Harga Beras/Rice Price ($/Kg)')
Beef = st.number_input('Harga Daging Sapi/Beef Price ($/mt)')
Sugar = st.number_input('Harga Gula/Sugar Price ($/mt)')
Coconut_Oil = st.number_input('Harga Minyak Kelapa/Coconut Oil Price ($/Kg)')
Gold = st.number_input('Harga Emas/Gold Price ($/Kg)')

# Inisialisasi variabel untuk menyimpan hasil prediksi
year_prediction = ''

# Tombol untuk melakukan prediksi
if st.button('Estimasi Tahun'):
      # Menggunakan model untuk melakukan prediksi
    input_data = [[Cocoa, Coffee, Tea, Rice, Beef, Sugar, Coconut_Oil, Gold]]
    year_prediction = int(model.predict(input_data)[0])  # Mengonversi menjadi integer dan mengakses elemen pertama
    
    # Menampilkan tahun perkiraan pada halaman Streamlit
    st.write('Tahun : ', year_prediction)



