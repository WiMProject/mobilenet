import numpy as np
import zipfile
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import load_img
import streamlit as st

class cfg:
    IMAGE_SIZE = 224

# Fungsi untuk memuat model
def load_model():
    model_path = 'final_model.keras'
    model = tf.keras.models.load_model(model_path)
    return model

# Fungsi prediksi
def predict(image, model):
    labels = ['NORMAL', 'TUBERCULOSIS', 'PNEUMONIA', 'COVID19']
    image = np.array(image)
    image = image / image.max()  # Normalisasi
    image = image.reshape(-1, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3)  # Ubah bentuk
    probabilities = model.predict(image).reshape(-1)  # Dapatkan probabilitas
    pred = labels[np.argmax(probabilities)]  # Kelas dengan probabilitas tertinggi
    return pred, {x: y for x, y in zip(labels, probabilities)}  # Kembalikan hasil

# Judul Aplikasi
st.title('Image Classification with Deep Learning Model')
st.markdown(
    """<style>body {font-family: Arial, sans-serif; background-color: #f9f9f9;} </style>""",
    unsafe_allow_html=True,
)
st.write("Upload X-ray images to classify them into different categories.")

# Pilihan upload
upload_option = st.radio("Select Upload Option", ('Upload Images', 'Upload ZIP of Images'))

model = load_model()

if upload_option == 'Upload Images':
    # Upload gambar satu atau banyak
    uploaded_images = st.file_uploader("Upload one or multiple images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

    if uploaded_images is not None:
        for uploaded_image in uploaded_images:
            # Muat gambar
            image = load_img(uploaded_image, target_size=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))

            # Lakukan prediksi
            pred, probabilities = predict(image, model)

            # Ambil label dan nilai probabilitas
            x = list(probabilities.keys())
            y = list(probabilities.values())

            # Tampilkan gambar dan probabilitas
            st.subheader(f"Prediction for: {uploaded_image.name}")  # Menampilkan nama gambar

            # Mengubah probabilitas ke persentase untuk ditampilkan
            probabilities_percentage = [round(prob * 100, 2) for prob in y]

            fig, ax = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 2]})

            # Subplot untuk gambar
            ax[0].imshow(image)
            ax[0].axis('off')
            ax[0].set_title('Uploaded Image', fontsize=14, pad=15)

            # Subplot untuk bar chart
            sns.barplot(x=probabilities_percentage, y=x, palette='cool', ax=ax[1])
            ax[1].set_title('Predicted Class Probabilities', fontsize=14, pad=15)
            ax[1].set_xlabel('Probability (%)', fontsize=12)
            ax[1].set_xlim([0, 100])  # Set limit x-axis 0 to 100
            for i, v in enumerate(probabilities_percentage):
                ax[1].text(v + 2, i, f"{v}%", color='black', va='center', fontsize=10)

            plt.tight_layout(w_pad=5.0)  # Menambahkan jarak antara subplot
            st.pyplot(fig)

            # Tampilkan hasil prediksi
            predicted_percentage = probabilities_percentage[np.argmax(y)]  # Ambil persentase prediksi tertinggi
            st.success(f'**Predicted Class:** {pred} ({predicted_percentage}%)')  # Kelas yang diprediksi dengan persentase

elif upload_option == 'Upload ZIP of Images':
    # Upload File ZIP
    uploaded_file = st.file_uploader("Upload a ZIP file containing images", type=['zip'])

    if uploaded_file is not None:
        # Ekstrak file ZIP
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall('extracted_images')

        # Ambil semua file gambar dari direktori yang diekstrak
        image_files = [f for f in os.listdir('extracted_images') if f.endswith(('jpg', 'jpeg', 'png'))]

        st.subheader('Uploaded Images:')
        # Buat kontainer untuk menampilkan gambar
        for image_file in image_files:
            # Muat gambar
            image_path = os.path.join('extracted_images', image_file)
            image = load_img(image_path, target_size=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))

            # Lakukan prediksi
            pred, probabilities = predict(image, model)

            # Ambil label dan nilai probabilitas
            x = list(probabilities.keys())
            y = list(probabilities.values())

            # Tampilkan gambar dan probabilitas
            st.image(image, caption=image_file, use_column_width=True, channels="RGB")

            # Mengubah probabilitas ke persentase untuk ditampilkan
            probabilities_percentage = [round(prob * 100, 2) for prob in y]

            fig, ax = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 2]})

            # Subplot untuk gambar
            ax[0].imshow(image)
            ax[0].axis('off')
            ax[0].set_title('Uploaded Image', fontsize=14, pad=15)

            # Subplot untuk bar chart
            sns.barplot(x=probabilities_percentage, y=x, palette='cool', ax=ax[1])
            ax[1].set_title('Predicted Class Probabilities', fontsize=14, pad=15)
            ax[1].set_xlabel('Probability (%)', fontsize=12)
            ax[1].set_xlim([0, 100])  # Set limit x-axis 0 to 100
            for i, v in enumerate(probabilities_percentage):
                ax[1].text(v + 2, i, f"{v}%", color='black', va='center', fontsize=10)

            plt.tight_layout(w_pad=5.0)  # Menambahkan jarak antara subplot
            st.pyplot(fig)

            # Tampilkan hasil prediksi
            predicted_percentage = probabilities_percentage[np.argmax(y)]  # Ambil persentase prediksi tertinggi
            st.success(f'**Predicted Class:** {pred} ({predicted_percentage}%)')  # Kelas yang diprediksi dengan persentase

        # Hapus gambar yang diekstrak setelah ditampilkan
        for image_file in image_files:
            os.remove(os.path.join('extracted_images', image_file))
        os.rmdir('extracted_images')
