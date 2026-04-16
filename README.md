# Smart Mining: Real-Time Mineral Classification System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google_Colab-GPU_T4-F9AB00?logo=googlecolab&logoColor=white)

> **Proyek AoL 1 - Deep Learning and Its Application** <br>
> Mengkomparasikan arsitektur *Transfer Learning* (EfficientNet-V2) dengan arsitektur *Custom* (Dual-Stream CNN) untuk otomasi penyortiran mineral (*Ore Sorting*) di industri pertambangan.


## Overview
Repositori ini berisi kode sumber, model, dan dokumentasi untuk proyek Deep Learning and Its Application (AoL 1).

Tujuan utama dari proyek ini adalah untuk mengklasifikasikan mineral untuk penyortiran bijih industri secara _real-time_. Dalam lingkungan pertambangan, penyortiran mineral bermutu tinggi dari batuan limbah pada ban berjalan (conveyor belt) yang bergerak cepat sangat penting untuk efisiensi energi. Proyek ini membandingkan dua arsitektur deep learning untuk menemukan keseimbangan terbaik antara akurasi prediktif dan kecepatan inferensi (latensi).

### Target Aplikasi
Sistem ini bertujuan untuk _conveyor-based_ penyortiran mineral di lingkungan environment. 
Mengelompokkan:
- Mineral Berharga: Bornite, Chrysocolla, Malachite
- Batuan Limbah/Lainnya: Quartz, Pyrite, Biotite, Muscovite

---

## Struktur Repositori
Untuk memastikan modularitas kode dan reproduksibilitas, repositori diatur sebagai berikut:

    mineral-sorting-deeplearning-SEANK/
    │
    ├── data/               # Direktori kosong untuk ekstraksi dataset Kaggle
    ├── models/             # Direktori untuk weight model yang disimpan (format .keras) dan link GDrive
    ├── notebooks/          
    │   └── Project_DL_Ore_Sorting_Sean_K.ipynb  # Pipeline utama (Training & Inferensi)
    ├── logs/               # Log pelacakan eksperimen TensorBoard
    ├── README.md           # Dokumentasi proyek
    └── requirements.txt    # Dependensi lingkungan Python

---

## Dataset & Preprocessing

- Sumber: Kaggle - Minerals Identification Classification (https://www.kaggle.com/datasets/youcefattallah97/minerals-identification-classification)
- Total Gambar: ~5.626 gambar dalam 7 kelas mineral.

- Pembagian Data: 70% Training | 15% Validasi | 15% Test.
Pembagian ini memastikan evaluasi akhir dilakukan secara eksklusif pada data yang belum pernah dilihat (unseen data) untuk mengukur kinerja dunia nyata secara objektif.
- Penanganan Ketidakseimbangan Data: Dataset mengalami ketidakseimbangan kelas yang limauan parah (misalnya, 180 gambar Quartz vs. 55 gambar Bornite dalam set pengujian). Untuk mencegah bias kelas mayoritas, Class Weighting diterapkan selama fase pelatihan, memberikan bobot penalti yang lebih tinggi kepada kelas minoritas.
- Augmentasi: Augmentasi visual dinamis (rotasi, zoom, kecerahan, kontras) diterapkan untuk mensimulasikan lingkungan pertambangan yang kasar dan berdebu.

---

## Arsitektur Model

### 1. EfficientNet-V2 (Baseline Model)
- Konsep: Menggunakan Transfer Learning (pre-trained pada ImageNet).
- Justifikasi: Bertindak sebagai baseline ambang atas untuk akurasi. Kemampuan ekstraksi fitur yang mendalam membuat model ini sangat tangguh terhadap noise visual, meskipun dengan biaya komputasi yang sedikit lebih tinggi. Berbeda dengan model arsitektur dalam pada umumnya, iterasi "V2" secara spesifik menggabungkan lapisan konvolusi awal menjadi operasi Fused-MBConv guna meminimalkan hambatan baca-tulis memori (I/O bottleneck) pada akselerator perangkat keras (GPU). Inovasi ini memungkinkan model mencapai akurasi maksimal sekaligus mempertahankan kecepatan komputasi (latensi) yang sangat responsif untuk kebutuhan industri real-time.

### 2. Dual-Stream CNN (Proposed Model)
- Konsep: Jaringan kustom ringan yang dilatih dari awal (from scratch).
- Justifikasi: Dirancang khusus untuk meniru kebutuhan industri dengan membagi input menjadi dua aliran paralel: aliran RGB (untuk ekstraksi warna) dan aliran Grayscale (untuk ekstraksi tekstur fisik). Ini mengurangi jumlah parameter secara signifikan untuk mencapai latensi ultra-rendah.

---

## Evaluasi & Hasil

Kedua model dievaluasi pada 15% dataset testing yang belum pernah dilihat. Evaluasi menggunakan metrik Weighted Average untuk memperhitungkan ketidakseimbangan kelas.

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
```
    git clone https://github.com/rednxt/mineral-sorting-deeplearning-SEANK.git
    cd mineral-sorting-deeplearning-SEANK
```
3. Instal Dependensi:
    Pastikan sudah menginstal Python 3.9+. Instal versi yang diperlukan menggunakan file requirements:
```
     pip install -r requirements.txt
```
5. Setup Kaggle API (Diperlukan untuk Dataset):
    Pastikan sudah memiliki file kaggle.json dari akun Kaggle.
```
    mkdir -p ~/.kaggle
    cp kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
```
---

## Cara Menjalankan Kode

Seluruh pipeline end-to-end disediakan dalam Jupyter Notebook modular. Google Colab (GPU T4) sangat direkomendasikan.

### Instruksi Opsional (Menjalankan di Google Colab):
Jika tidak ingin mengunggah file ```.ipynb``` secara utuh ke dalam Colab, berikut adalah cara opsional yaitu membuat Notebook kosong baru di Google Colab, lalu menyalin (copas) kode dari file ```Project_DL_Ore_Sorting_Sean_K.ipynb``` secara manual, blok per blok, ke dalam sel (cell) komputasi yang baru.

### 1. Prosedur Pelatihan
1. Buka ```notebooks/Project_DL_Ore_Sorting_Sean_K.ipynb```
2. Jalankan sel Setup dan Import untuk mengunduh dan membersihkan dataset secara otomatis.
3. Eksekusi sel Model Building dan Training. Proses ini menggunakan ```EarlyStopping``` (patience=7) untuk mencegah overfitting.
4. Log eksperimen disimpan secara otomatis ke direktori ```logs/fit/.```

### 2. Prosedur Inferensi (Menguji Data Baru)
Untuk memverifikasi kemampuan model dalam mengenali data yang tidak pernah dilihat selama proses training, proyek ini menyediakan skrip inferensi visual. Pengujian ini membuktikan bahwa model memahami pola visual mineral, bukan sekadar menghafal (overfitting).

Jalankan blok Inferensi di akhir notebook. Skrip akan mengambil batch sampel dari ```test_ds```:
```
# Skrip mengambil 1 batch sampel dan mengeksekusi prediksi probabilitas.
# Menampilkan grid Matplotlib (2x3) dengan persentase Confidence Score.
# Judul berwarna BIRU menandakan prediksi BENAR, dan MERAH menandakan prediksi SALAH.

try:
    print("Inference menggunakan EfficientNet-V2 (Data Test Unseen):")
    visualize_test_inference(model_eff, test_ds, class_names)

    print("\nInference menggunakan Dual-Stream CNN (Data Test Unseen):")
    visualize_test_inference(model_dual, test_ds, class_names)
except NameError as e:
    print(f"Error: {e}. Pastikan model_eff dan model_dual sudah didefinisikan.")
  ```  
---

## Model dan Logs

Karena batasan ukuran file GitHub, weight model ```.keras``` dan demonstrasi video di-host secara eksternal.

* EfficientNet-V2: https://drive.google.com/file/d/1Xt9I2CrlPnvCI3xkXqeayrO3CHakutlJ/view?usp=sharing
* Dual-Stream CNN: https://drive.google.com/file/d/1kORbzcWstbheZj9HTH0lb_MYxilDD2I7/view?usp=sharing 
* Log Eksperimen: Log TensorBoard tersedia di dalam direktori logs/ di repositori ini.

---

## Author
Sean Kenneth Tommy Keleyan
2702751694
MIT

