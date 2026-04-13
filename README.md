# Smart Mining: Real-Time Mineral Classification System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview
Repositori ini berisi kode sumber, model, dan dokumentasi untuk proyek Deep Learning and Its Application (AoL 1).

Tujuan utama proyek ini adalah untuk mengklasifikasikan mineral guna penyortiran bijih industri secara real-time. Dalam lingkungan pertambangan, penyortiran mineral bermutu tinggi dari batuan limbah pada ban berjalan (conveyor belt) yang bergerak cepat sangat penting untuk efisiensi energi. Proyek ini membandingkan dua arsitektur deep learning untuk menemukan keseimbangan terbaik antara akurasi prediktif dan kecepatan inferensi (latensi).

### Target Aplikasi
- Mineral Berharga: Bornite, Chrysocolla, Malachite
- Batuan Limbah/Lainnya: Quartz, Pyrite, Biotite, Muscovite

---

## Struktur Repositori
Untuk memastikan modularitas kode dan reproduksibilitas, repositori diatur sebagai berikut:

    mineral-sorting-deeplearning-SEANK/
    │
    ├── data/               # Direktori kosong untuk ekstraksi dataset Kaggle
    ├── models/             # Direktori untuk weight model yang disimpan (format .keras)
    ├── notebooks/          
    │   └── AoL_Mineral_Classification.ipynb  # Pipeline utama (Training & Inferensi)
    ├── logs/               # Log pelacakan eksperimen TensorBoard
    ├── README.md           # Dokumentasi proyek
    └── requirements.txt    # Dependensi lingkungan Python

---

## Dataset & Preprocessing

- Sumber: Kaggle - Minerals Identification Classification (https://www.kaggle.com/datasets/youcefattallah97/minerals-identification-classification)
- Total Gambar: ~5.626 gambar dalam 7 kelas mineral.
- Pembagian Data: 70% Training | 15% Validasi | 15% Test. Pembagian ketat ini memastikan evaluasi akhir dilakukan secara eksklusif pada data yang belum pernah dilihat (unseen data) untuk mengukur kinerja dunia nyata secara objektif.
- Penanganan Ketidakseimbangan Data: Dataset mengalami ketidakseimbangan kelas yang parah (misalnya, 180 gambar Quartz vs. 55 gambar Bornite dalam set pengujian). Untuk mencegah bias kelas mayoritas, Class Weighting diterapkan selama fase pelatihan, memberikan bobot penalti yang lebih tinggi kepada kelas minoritas.
- Augmentasi: Augmentasi visual dinamis (rotasi, zoom, kecerahan, kontras) diterapkan untuk mensimulasikan lingkungan pertambangan yang kasar dan berdebu.

---

## Arsitektur Model

### 1. EfficientNet-V2 (Baseline Model)
- Konsep: Menggunakan Transfer Learning (pre-trained pada ImageNet).
- Justifikasi: Bertindak sebagai baseline ambang atas untuk akurasi. Kemampuan ekstraksi fitur yang mendalam membuatnya sangat tangguh terhadap noise visual, meskipun dengan biaya komputasi yang sedikit lebih tinggi.

### 2. Dual-Stream CNN (Proposed Model)
- Konsep: Jaringan kustom ringan yang dilatih dari awal (from scratch).
- Justifikasi: Dirancang khusus untuk meniru kebutuhan industri dengan membagi input menjadi dua aliran paralel: aliran RGB (untuk ekstraksi warna) dan aliran Grayscale (untuk ekstraksi tekstur fisik). Ini mengurangi jumlah parameter secara signifikan untuk mencapai latensi ultra-rendah.

---

## Evaluasi & Hasil

Kedua model dievaluasi pada 15% dataset pengujian yang belum pernah dilihat. Evaluasi menggunakan metrik Weighted Average untuk memperhitungkan ketidakseimbangan kelas.

| Metrik (Weighted) | EfficientNet-V2 (Baseline) | Dual-Stream CNN (Proposed) |
| ----------------- | -------------------------- | -------------------------- |
| Akurasi           | 76.49% (Tinggi)            | 42.81% (Moderat)           |
| Presisi           | 0.7909                     | 0.4740                     |
| Recall            | 0.7649                     | 0.4281                     |
| F1-Score          | 0.7719                     | 0.4291                     |
| Latensi/Gambar    | 4.74 ms                    | 3.89 ms (Lebih Cepat)      |

### Analisis Kritis (Trade-off)
Terdapat trade-off yang jelas antara akurasi prediktif dan kecepatan inferensi.
Meskipun Dual-Stream CNN berhasil mencapai inferensi yang lebih cepat (3.89 ms), arsitekturnya yang dangkal yang dilatih dari awal kesulitan melakukan generalisasi pada augmentasi ekstrem, sehingga menghasilkan akurasi 42.81%. Sebaliknya, EfficientNet-V2 memanfaatkan transfer learning untuk mencapai akurasi 76.49% yang sangat andal dengan tetap mempertahankan latensi yang mumpuni untuk real-time (4.74 ms, setara dengan >200 FPS). Untuk penerapan industri segera, EfficientNet-V2 adalah pilihan yang lebih unggul.

---

## Setup Lingkungan & Instalasi

Untuk menjalankan proyek ini secara lokal atau di server cloud, ikuti langkah-langkah berikut:

1. Clone repositori:
    git clone https://github.com/rednxt/mineral-sorting-deeplearning-SEANK.git
    cd mineral-sorting-deeplearning-SEANK

2. Instal Dependensi:
    Pastikan Anda telah menginstal Python 3.9+. Instal versi yang diperlukan menggunakan file requirements:
    pip install -r requirements.txt

3. Setup Kaggle API (Diperlukan untuk Dataset):
    Pastikan Anda memiliki file kaggle.json dari akun Kaggle Anda.
    mkdir -p ~/.kaggle
    cp kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json

---

## Cara Menjalankan Kode

Seluruh pipeline ujung-ke-ujung disediakan dalam Jupyter Notebook modular. Google Colab (GPU T4) sangat direkomendasikan.

### 1. Prosedur Pelatihan
1. Buka notebooks/AoL_Mineral_Classification.ipynb
2. Jalankan sel Setup & Import untuk mengunduh dan membersihkan dataset secara otomatis.
3. Eksekusi sel Model Building dan Training. Proses ini menggunakan EarlyStopping (patience=7) untuk mencegah overfitting.
4. Log eksperimen disimpan secara otomatis ke direktori logs/fit/.

### 2. Prosedur Inferensi (Menguji Data Baru)
Untuk memverifikasi model menggunakan gambar tunggal, jalankan blok Inferensi di akhir notebook. Skrip akan mengambil batch dari test_ds yang terisolasi:

    # Skrip akan menampilkan grid Matplotlib yang menunjukkan gambar,
    # kelas yang diprediksi, persentase kepercayaan, dan label asli.
    # Teks berwarna BIRU untuk prediksi benar dan MERAH untuk kesalahan klasifikasi.
    visualize_test_inference(model_eff, test_ds, class_names)

---

## Artefak & Hasil Akhir

Karena batasan ukuran file GitHub, weight model .keras dan demonstrasi video di-host secara eksternal.

* EfficientNet-V2 Weights: [Masukkan Link Google Drive Di Sini]
* Dual-Stream CNN Weights: [Masukkan Link Google Drive Di Sini]
* Video Demo Teknis: [Masukkan Link YouTube/GDrive Di Sini]
* Log Eksperimen: Log TensorBoard tersedia di dalam direktori logs/ di repositori ini.

---

## Penulis
* Sean K - [ID Mahasiswa Anda] - [Universitas/Mata Kuliah Anda]
