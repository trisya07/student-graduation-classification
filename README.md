# Laporan Proyek Machine Learning - Trisya Nurmayanti

## Domain Proyek

Perguruan tinggi bertanggung jawab dalam mencetak lulusan yang memiliki kualitas, yang dapat dinilai dari tingkat kelulusan mahasiswa [[1](https://www.researchgate.net/publication/362537239_PREDIKSI_KELULUSAN_MAHASISWA_DENGAN_METODE_NAIVE_BAYES)]. Selain itu, kelulusan tepat waktu adalah indikator utama kesuksesan mahasiswa dalam mendapatkan gelar sarjana [[2](http://e-journal.stmiklombok.ac.id/index.php/misi/article/view/875)]. Namun pada kenyataannya, mahasiswa tidak selalu menyelesaikannya dalam waktu empat tahun [[3](https://jom.fti.budiluhur.ac.id/index.php/SKANIKA/article/view/2976)]. Menurut BAN-PT (2019), kelulusan mahasiswa merupakan salah satu instrumen untuk menentukan akreditasi suatu universitas [[4](https://ejurnal.itats.ac.id/snestik/article/view/4388)]. Kelulusan tepat waktu merupakan salah satu indikator penting dalam dunia pendidikan tinggi, yang mencerminkan efektivitas dari proses pembelajaran yang diterima oleh mahasiswa. Di sisi lain, pemahaman yang lebih baik mengenai faktor-faktor yang memengaruhi kelulusan dapat membantu pihak kampus untuk memberikan intervensi yang lebih tepat sasaran, seperti pemberian dukungan ekstra bagi mahasiswa yang berpotensi mengalami keterlambatan dalam kelulusan.
Selain itu, dengan adanya prediksi ini, pihak kampus juga dapat merancang kebijakan yang lebih efisien dalam mengelola mahasiswa dan memberikan dukungan yang sesuai berdasarkan kebutuhan masing-masing mahasiswa. Oleh karena itu, proyek ini menggunakan algoritma machine learning untuk membangun model prediktif yang dapat memberikan prediksi kelulusan mahasiswa berdasarkan data historis.
Penelitian ini bertujuan untuk mengembangkan pemodelan prediksi kelulusan tepat waktu bagi mahasiswa dengan menggunakan algoritma random forest. Mengingat bahwa kelulusan tepat waktu merupakan indikator utama kesuksesan dalam meraih gelar sarjana dan juga berpengaruh pada akreditasi perguruan tinggi, penelitian ini berfokus pada pemanfaatan data akademik dan demografis untuk membantu perguruan tinggi dalam mendukung mahasiswa menyelesaikan studi dengan lebih baik.

## Referensi
- [[1]](https://journal.likmi.ac.id/index.php/media-informatika/article/view/85) Prediksi Kelulusan Mahasiswa Program Pendidikan Multi Profesi 1 Tahun dengan Metode Naïve Bayes 
- [[2]](https://ejournal.uniramalang.ac.id/index.php/g-tech/article/view/1850) Penerapan Algoritma Decision Tree dalam Klasifikasi Data Prediksi Kelulusan Mahasiswa
- [[3]](https://prosiding.stekom.ac.id/index.php/SEMNASTEKMU/article/view/170) Klasifikasi Terhadap Prediksi Kelulusan Mahasiswa Dengan Menggunakan Metode Support Vector Machine (SVM)

## Business Understanding

Dalam proyek ini, tujuan utama adalah memprediksi apakah mahasiswa akan lulus tepat waktu atau tidak berdasarkan data yang tersedia, seperti nilai Indeks Prestasi Semester (IPS), program studi, jenis kelamin, status pegawai, dan tahun kelahiran. Prediksi kelulusan tepat waktu ini memiliki dampak besar terhadap strategi akademik yang diambil oleh universitas, serta bisa membantu meningkatkan kualitas layanan pendidikan dan memberikan intervensi yang tepat sasaran kepada mahasiswa yang berpotensi terlambat dalam kelulusan.
Sebagai contoh, dengan menggunakan model prediksi ini, pihak kampus dapat merancang kebijakan atau program bantuan khusus bagi mahasiswa yang berisiko terlambat lulus. Selain itu, informasi yang diperoleh juga bisa digunakan untuk merencanakan kebijakan pendaftaran atau pengelolaan beban akademik yang lebih efisien. Oleh karena itu, tujuan dari proyek ini adalah untuk mengembangkan sistem prediksi yang akurat guna membantu pihak universitas dalam pengambilan keputusan yang lebih baik terkait kelulusan mahasiswa.

### Problem Statements

- Bagaimana cara memprediksi mahasiswa yang akan lulus tepat waktu berdasarkan data yang tersedia, seperti nilai IPS, program studi, jenis kelamin, dan status pegawai?
- Faktor-faktor apa saja yang paling mempengaruhi ketepatan waktu kelulusan mahasiswa?
- Bagaimana cara membangun model yang dapat memprediksi kelulusan tepat waktu dan tidak tepat waktu?

### Goals

- Menggunakan algoritma machine learning untuk membangun model prediksi yang mampu memproyeksikan kelulusan mahasiswa tepat waktu atau tidak.
- Mengidentifikasi faktor-faktor yang berpengaruh terhadap ketepatan waktu kelulusan mahasiswa dengan teknik feature importance.
- Menghasilkan model yang dapat memberikan prediksi yang akurat dan dapat diandalkan melalui teknik optimasi hyperparameter dan validasi model.

### Solution statements
- Model yang digunakan adalah Random Forest Classifier karena kemampuannya menangani data campuran (numerik dan kategorikal), ketangguhan terhadap overfitting, serta menyediakan interpretasi melalui fitur penting (feature importance).
- Ketidakseimbangan data diselesaikan dengan SMOTE (Synthetic Minority Over-sampling Technique) yang membantu memperkuat representasi kelas minoritas ('Tidak Lulus Tepat Waktu').
- Model ditingkatkan dengan hyperparameter tuning menggunakan Optuna, untuk mencari konfigurasi terbaik dalam hal akurasi dan keseimbangan prediksi antar kelas.
- Data diproses dengan feature engineering, termasuk mengubah nilai IPS menjadi fitur biner berbasis median dan mengelompokkan tahun_lahir menjadi kategori usia.
- Evaluasi dilakukan dengan metrik akurasi, precision, recall, dan F1-score untuk dapat dianalisa lebih lanjut

## Data Understanding

### Informasi Dataset

Dataset yang digunakan dalam proyek ini adalah dataset mahasiswa dari angkatan 2015-2020 yang berasal dari data internal kampus yaitu Universitas Buana Perjuangan karawang, namun untuk memenuhi ketentuan submission saya upload ke github pribadi saya. dataset yang didapatkan terdiri **[data transkip nilai](https://raw.githubusercontent.com/trisya07/student-graduation-classification/main/transkip_nilai_fix.csv)** dan **[data lulusan](https://raw.githubusercontent.com/trisya07/student-graduation-classification/main/ms_lulusan_fix.csv)**. Pada data transkip nilai berisi detail nilai yang diperoleh para mahasiswa untuk mata kuliah yang diambil. Sedangkan pada data lulusan berisi informasi demografis dan akademik mahasiswa seperti tanggal masuk dan lulus, serta predikat
kelulusan. Dataset ini dapat digunakan untuk melakukan klasifikasi mahasiswa dapat lulus tepat waktu atau tidak.

### Deskripsi Variabel

**1. Data Transkip Nilai**

Dataset transkrip nilai terdiri dari 256.299 baris dan 11 kolom, di mana setiap baris merepresentasikan satu entri nilai mata kuliah yang diambil oleh mahasiswa selama masa studi. Dataset ini digunakan untuk menggambarkan performa akademik mahasiswa berdasarkan nilai yang diperoleh tiap semester. Berikut adalah deskripsi dari masing-masing variabel yang terdapat dalam dataset:

| Nama Kolom       | Deskripsi                                                                                                                                        |
|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| `id`             | ID unik untuk setiap baris entri transkrip                                                                                                       |
| `nim`            | Nomor Induk Mahasiswa                                                                                                                            |
| `kode_mk`        | Kode mata kuliah yang diambil                                                                                                                    |
| `nama_mk`        | Nama mata kuliah                                                                                                                                 |
| `nama_mk_indo`   | Nama mata kuliah dalam Bahasa Indonesia                                                                                                          |
| `nama_mk_ing`    | Nama mata kuliah dalam Bahasa Inggris                                                                                                            |
| `nilai_grade`    | Nilai akhir dalam skala 0–4 yang merupakan hasil konversi dari nilai_total. Fitur ini digunakan untuk menghitung IPS untuk analisis lebih lanjut |
| `nilai_total`    | Merupakan nilai akhir mahasiswa dalam skala 0–100                                                                                                |
| `semester`       | Semester ketika mata kuliah diambil dari semester 0-9                                                                                            |
| `sks_mk`         | Jumlah SKS dari mata kuliah tersebut. Fitur ini digunakan untuk menghitung IPS untuk analisis lebih lanjut                                       |
| `grade`          | Huruf mutu ('B+', 'A-', 'B-', 'A', 'B', 'C', 'C+', 'D') yang merupakan representasi kategori nilai berdasarkan `nilai_grade`                     |

**2. Data Lulusan**

Dataset ini terdiri dari 4.542 baris dan 9 kolom, di mana setiap baris merepresentasikan satu mahasiswa yang telah menyelesaikan studinya. Data ini berisi informasi demografis dan status kelulusan mahasiswa, yang digunakan untuk mendukung analisis prediktif terkait ketepatan waktu kelulusan. Berikut adalah deskripsi dari masing-masing variabel yang terdapat dalam dataset:

| Kolom            | Deskripsi                                                                                                                      |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------|
| `nim`            | Nomor Induk Mahasiswa, sebagai identitas unik dan penghubung ke data transkrip.                                                |
| `prodi`          | Kode program studi tempat mahasiswa terdaftar (26201, 48201, 55201, 57201, 61201, 62201, 73201, 74201, 86206, 87205, 21201)    |
| `predikat`       | Predikat kelulusan yang diperoleh, seperti 'Pujian', 'Sangat Memuaskan', 'Memuaskan', dan '-'                                  |
| `tanggal_lulus`  | Tanggal resmi kelulusan mahasiswa. Fitur ini digunakan menghitung untuk menentukan kelas lulus tepat waktu atau tidak          |
| `tgl_masuk`      | Tanggal mahasiswa pertama kali masuk kuliah Fitur ini digunakan menghitung untuk menentukan kelas lulus tepat waktu atau tidak |
| `status_masuk`   | Status masuk mahasiswa, `0` untuk reguler dan `1` untuk pindahan.                                                              |
| `jenis_kelamin`  | Jenis kelamin mahasiswa, `0` untuk perempuan dan `1` untuk laki-laki.                                                          |
| `tahun_lahir`    | Tahun kelahiran mahasiswa.                                                                                                     |
| `status_pegawai` | Status apakah mahasiswa bekerja selama kuliah, `0` untuk tidak bekerja, `1` untuk bekerja, serta '2' outlier                   |


### Exploratory Data Analysis (EDA)
   - Metode `info()` digunakan untuk menampilkan informasi umum tentang struktur DataFrame, termasuk jumlah entri, tipe data kolom, dan nilai non-null.
     - Data Transkip
       
       <img width="221" alt="info transkip" src="https://github.com/user-attachments/assets/0702420c-5a0d-480e-abf8-0425ae06f936" />

       Secara umum, kondisi data pada data transkip cukup baik dengan sebagian kecil nilai kosong pada kolom nama_mk dan nama_mk_ing sehingga perlu dianalisis lebih lanjut. Sedangkan kolom lainnya lengkap. Tipe Data pada setiap fitur pun sudah sesuai.

     - Data Lulusan

       <img width="286" alt="info lulusan" src="https://github.com/user-attachments/assets/211ad8b9-17c7-47e4-8981-2b47e582583a" />

       Pada dataset lulusan, terdapat ketidaksesuaian tipe data pada kolom tanggal_lulus dan tgl_masuk, di mana kedua kolom tersebut masih bertipe object. Padahal, secara semantik, kolom-kolom ini merepresentasikan informasi tanggal dan seharusnya dikonversi ke dalam tipe data datetime agar dapat diolah dengan benar dalam analisis waktu, terutama untuk digunakan sebagai penentuan labeling pada proyek ini. Selain isu tipe data tersebut, tidak ditemukan adanya nilai kosong (missing value) pada dataset ini, sehingga kondisi datanya secara umum tergolong baik.
       
- Metode describe() digunakan untuk melihat ringkasan statistik dari seluruh kolom dalam dataset
  - Data Transkip
    
    <img width="669" alt="describe transkip" src="https://github.com/user-attachments/assets/546f3ba7-8ed8-43a5-906e-6489f165d13e" />

    Berdasarkan hasil eksplorasi statistik deskriptif terhadap dataset transkrip nilai, ditemukan bahwa nilai minimum pada kolom semester adalah 0. Secara umum, perkuliahan dimulai dari semester 1, sehingga keberadaan nilai 0 pada kolom ini perlu dianalisis lebih lanjut untuk memastikan validitas datanya. Di luar hal tersebut, distribusi nilai pada kolom-kolom lainnya terlihat normal dan tidak menunjukkan adanya kejanggalan yang berarti.
  
  - Data Lulusan
 
    <img width="856" alt="describe lulusan" src="https://github.com/user-attachments/assets/bc7058be-f7c7-4594-b1f9-da2d01c30f0d" />

    Hasil analisis statistik deskriptif pada dataset lulusan menunjukkan bahwa nilai minimum pada kolom tahun_lahir adalah 0. Hal ini tidak masuk akal karena tahun lahir seharusnya bernilai di atas 1900-an. Oleh karena itu, data dengan nilai 0 pada kolom tahun_lahir perlu ditinjau ulang atau dibersihkan. Sementara itu, distribusi pada kolom lainnya terlihat wajar.

- digunakan fungsi `isnull().sum()` untuk mengetahui jumlah nilai yang hilang (missing value) pada setiap kolom dalam dataset. Fungsi ini bekerja dengan memeriksa setiap elemen dalam dataset dan mengembalikan total nilai kosong (null) untuk masing-masing kolom, sehingga memudahkan identifikasi kolom mana yang memerlukan penanganan lebih lanjut.
  - Data Transkip
    
    <img width="118" alt="null transkip" src="https://github.com/user-attachments/assets/f043a109-6d5e-4c58-be4f-eb427a444e1b" />

    Hasil pengecekan menunjukkan bahwa terdapat beberapa nilai yang hilang pada data transkrip. Kolom `nama_mk` dan `nama_mk_ing` masing-masing memiliki 6 data yang hilang, sedangkan kolom `nama_mk_indo` memiliki 1 data yang hilang. Nilai hilang ini perlu dianalisis lebih lanjut untuk menentukan apakah perlu diisi, dihapus, atau diabaikan tergantung pada pengaruhnya terhadap proses pemodelan.

  - Data Lulusan

    <img width="110" alt="null lulusan" src="https://github.com/user-attachments/assets/a755437d-8a6b-4394-8f9b-0e94a7c1ebab" />

    Seperti yang telah disebutkan sebelumnya pada pengecekan dengan`info()', data lulusan tidak menunjukkan adanya indikasi nilai yang hilang. Hasil pengecekan menggunakan fungsi isnull().sum() juga mengonfirmasi bahwa seluruh kolom pada dataset ini terisi lengkap, tanpa adanya missing value. Hal ini menunjukkan bahwa data lulusan sudah cukup bersih dan siap digunakan untuk tahap analisis lebih lanjut.
         
- menggunakan fungsi `duplicated().sum()` untuk melakukan pengecekan duplikasi data. Fungsi ini bekerja dengan memeriksa setiap baris dalam dataset dan mengidentifikasi apakah ada baris yang persis sama dengan baris lainnya. Jika ditemukan baris yang identik (duplikat), maka akan dihitung jumlah totalnya. Hasil dari fungsi ini memberikan gambaran apakah ada data yang tercatat lebih dari sekali, yang bisa memengaruhi hasil analisis atau model yang dibangun.

  <img width="183" alt="duplikat" src="https://github.com/user-attachments/assets/3078fdab-81e5-4e40-b2ea-4a46032be029" />

  Hasil pengecekan duplikasi data menunjukkan bahwa tidak ada baris yang terduplikasi dalam kedua dataset, baik pada data transkrip maupun data lulusan. Artinya, setiap baris dalam dataset adalah unik dan tidak ada entri yang tercatat lebih dari sekali. Hal ini penting karena memastikan bahwa analisis atau model yang dibangun tidak terganggu oleh data yang berulang.

- Visualisasi beberapa kolom untuk memahami distribusi variabel-variabel penting dalam dataset, digunakan bar plot untuk menggambarkan frekuensi dari beberapa kategori dalam data lulusan dan data transkrip. Setiap bar plot menggambarkan distribusi dari satu fitur, yang memungkinkan kita untuk melihat bagaimana data tersebar dan mencari pola-pola yang mungkin ada.
  
  ![eda](https://github.com/user-attachments/assets/5b4573e3-0b4f-462b-9227-c46895714526)
  
  Berdasarkan Visualisasi di atas dapat disimpulkan bahwa:
  - Status Pegawai
    Mayoritas mahasiswa dalam dataset berasal dari kategori bukan pegawai. Hanya sebagian kecil yang berstatus sebagai pegawai atau memiliki status lain. Hal ini menunjukkan bahwa sebagian besar mahasiswa kemungkinan memiliki waktu belajar yang lebih fleksibel. Namun, mahasiswa yang bekerja berpotensi memiliki keterbatasan waktu yang bisa berdampak pada proses belajar mereka. Oleh karena itu, status sebagai pegawai dapat menjadi salah satu faktor yang memengaruhi ketepatan waktu kelulusan.
  - Jenis Kelamin
    Distribusi jenis kelamin menunjukkan bahwa mahasiswa dengan label 0 (kemungkinan laki-laki) lebih banyak dibandingkan label 1 (kemungkinan perempuan). Jika terdapat perbedaan dalam pola belajar atau prestasi akademik antara jenis kelamin tertentu, maka jenis kelamin bisa menjadi salah satu faktor yang turut memengaruhi ketepatan waktu kelulusan mahasiswa.
  - Status Masuk
    Sebagian besar mahasiswa masuk melalui jalur yang diberi label 0, dengan sangat sedikit mahasiswa yang berasal dari jalur masuk label 1. Ini menunjukkan adanya dominasi jalur masuk tertentu yang mungkin memiliki sistem seleksi atau kesiapan akademik yang berbeda. Status masuk dapat menjadi indikator awal kesiapan mahasiswa dalam menjalani perkuliahan, yang pada akhirnya dapat berpengaruh terhadap kelulusan.
  - Program Studi (Prodi)
    Terdapat ketimpangan distribusi jumlah mahasiswa antar program studi. Beberapa prodi memiliki jumlah mahasiswa yang jauh lebih banyak dibandingkan yang lain. Hal ini bisa mencerminkan perbedaan minat, kapasitas penerimaan, atau bahkan tingkat kesulitan akademik di masing-masing prodi. Program studi dengan tingkat kesulitan yang tinggi atau kurikulum yang padat bisa berdampak pada lama studi mahasiswa, sehingga berpengaruh pada kelulusan tepat waktu.
  - Grade
    Distribusi nilai menunjukkan bahwa mayoritas mahasiswa mendapatkan nilai tinggi seperti A, A-, dan B+, dengan proporsi yang lebih kecil pada nilai rendah seperti C dan D. Nilai ini merupakan representasi langsung dari performa akademik mahasiswa. Mahasiswa dengan nilai yang baik kemungkinan besar memiliki pemahaman materi yang lebih kuat dan beban remedial yang lebih sedikit, sehingga lebih mungkin menyelesaikan studi tepat waktu.

## Data Preparation
Pada tahap ini, dilakukan serangkaian proses pembersihan dan penyesuaian data sebelum dianalisis lebih lanjut. Langkah-langkah yang diterapkan dijelaskan secara berurutan sesuai dengan notebook.
### 1. Menangani Missing Value pada Kolom `nama_mk`

Langkah pertama yang dilakukan adalah menangani nilai kosong (missing value) pada kolom `nama_mk` dalam data transkrip. Ditemukan bahwa terdapat nilai `null` pada kolom tersebut. Untuk dilakukan pengecekan berdasarkan `kode_mk` matakuliah tersebut yaitu `FM1190035` untuk melihat apakah ada data `nama_mk` yang terisi atau tidak. Dari Hasilnya ternyata ada data `nama_mk` yang bernilai valid.

<img width="377" alt="tangani null nama_mk" src="https://github.com/user-attachments/assets/88349987-dbda-41ab-89e1-0e21be1dfa0b" />

Hasilnya menunjukkan bahwa kode_mk tersebut memiliki nilai nama_mk pada entri lain, sehingga dapat digunakan sebagai referensi. Maka, dilakukan pengisian nilai yang kosong dengan nama mata kuliah yang sesuai:

```
df_transkip.loc[df_transkip['kode_mk'] == 'FM1190035', 'nama_mk'] = "FARMAKOTERAPI INFEKSI, MATA, PERNAFASAN, TULANG DAN SENDI"
```
Tahapan ini penting dilakukan agar informasi terkait nama mata kuliah tetap lengkap dan akurat, terutama jika kolom ini akan digunakan dalam analisis selanjutnya atau untuk identifikasi data.

### 2. Menghapus Kolom yang Tidak Digunakan
Dua kolom tambahan yaitu nama_mk_indo dan nama_mk_ing dihapus dari dataset. Hal ini dilakukan karena kolom-kolom tersebut tidak digunakan dalam analisis, dan penghapusannya bertujuan untuk menyederhanakan struktur data serta mengurangi redundansi. Hasil dari pengahapusan kolom dapat dilihat pada gambar berikut. 

<img width="238" alt="hasil hps kolom nama mk indo ing" src="https://github.com/user-attachments/assets/e65e8c72-2cae-4889-9270-504ec8e02a91" />

Berdasarkan gambar di atas, terlihat bahwa kolom nama_mk_indo dan nama_mk_ing telah dihapus, sehingga dataset transkrip saat ini memiliki 9 kolom yang tersisa. Hal ini dilakukan untuk menyederhanakan data dan menghindari redundansi informasi.

### 3. Menghapus data `tahun_lahir` yang bernilai `0`

Berdasarkan hasil statistik deskriptif sebelumnya terlihat bahwa kolom tahun_lahir ada yang bernilai `0` tentunya ini tidak benar, sehingga perlu ditangani. Pada proyek ini penanganan dilakukan dengan menghapusnya agar data menajadi relevan dan optimal. 

### 4. Konversi Kolom Tanggal ke Format Datetime

Langkah selanjutnya adalah mengonversi kolom `tanggal_lulus` dan `tgl_masuk` ke dalam format datetime. Hal ini dilakukan agar data tanggal dapat diproses lebih lanjut, misalnya untuk menghitung lama studi, membuat fitur waktu, atau melakukan filtering berdasarkan periode. Konversi dilakukan menggunakan kode berikut:

```
# Konversi kolom 'tanggal_lulus' dan 'tgl_masuk' ke format datetime
df_lulusan['tanggal_lulus'] = pd.to_datetime(df_lulusan['tanggal_lulus'], format='%Y-%m-%d')
df_lulusan['tgl_masuk'] = pd.to_datetime(df_lulusan['tgl_masuk'], format='%Y-%m-%d')
```
Tahapan ini penting karena jika kolom tanggal tetap dalam bentuk string, maka akan menyulitkan dalam melakukan analisis berbasis waktu, seperti menghitung durasi studi. Hasil konversi tipe data dapat dilihat pada gambar berikut.

<img width="271" alt="hasil ganti tipedata diatetime" src="https://github.com/user-attachments/assets/8f2adfae-7f0e-49bf-9fd2-ef5c28a32fdb" />

### 5. Menangani Nilai Tidak Valid pada Kolom `semester`

Pada eksplorasi data, ditemukan bahwa beberapa baris memiliki nilai semester 0, padahal seharusnya semester dimulai dari 1. Untuk menangani hal ini, langkah awal dilakukan identifikasi seluruh kode mata kuliah yang tercatat berada di semester 0. Setelah itu, dilakukan pengecekkan apakah kode mata kuliah tersebut juga muncul di semester lain. Gambar dibawah ini merupakan kode pengecekan kolom `semester` berdasarkan `kode_mk` apakah terdapat di semester lain dengan menampilkan 3 elemen pertama `kode_mk`.

```
kode_mk_semester_0 = df_transkip[df_transkip['semester'] == 0]['kode_mk'].unique()
kode_mk_semester_0
for kode_mk in kode_mk_semester_0[:3]:
    print(f"{kode_mk}: {df_transkip[df_transkip['kode_mk'] == kode_mk]['semester'].unique()}")
```
<img width="98" alt="Hasil identifikasi semester 0" src="https://github.com/user-attachments/assets/50848f9f-fbc5-44fc-ab82-58d619f4299a" />

Hasil di atas menunjukan bahwa terdapat `kode_mk` yang sama di semester lain. Sehingga nilai semester 0 diganti menggunakan nilai yang paling sering muncul (modus) dengan langkah berikut.
```python
    semester_terbanyak = df_transkip[df_transkip['kode_mk'] == kode_mk]['semester'].mode()[0]
    df_transkip.loc[df_transkip['kode_mk'] == kode_mk, 'semester'] = semester_terbanyak
```

Namun Hasil dari penggantian nilai tersebut masih menyisakan kolom `semester` yang bernilai `0` seperti pada gambar dibawah ini

<img width="475" alt="Data Hasil penanganan semester 0" src="https://github.com/user-attachments/assets/7365c411-a24c-4403-8868-fe2defa5a45e" />

Sehingga data tersebut dihapus dari dataset karena dianggap tidak valid dan dapat mengganggu hasil analisis. Langkah ini bertujuan menjaga kualitas dan integritas data agar informasi urutan studi mahasiswa tetap akurat dan logis. Dataset setelah penghapusan tersebut adalah 256.293 data.

### 6. Menghitung dan Menyusun Nilai IPS per Semester

Langkah selanjutnya adalah menghitung nilai IPS (Indeks Prestasi Semester) setiap mahasiswa berdasarkan transkrip nilainya. Perhitungan dilakukan dengan mengalikan bobot nilai (nilai_grade) dengan jumlah SKS mata kuliah, kemudian dibagi total SKS pada semester tersebut. Hasilnya kemudian dibulatkan ke dua desimal.

<img width="178" alt="RUMUS IPS" src="https://github.com/user-attachments/assets/00d4f653-1e4a-43d2-aada-106374aa520a" />

Nilai IPS yang telah dihitung kemudian diubah dari format baris ke format kolom, sehingga setiap mahasiswa memiliki kolom "IPS SMT1", "IPS SMT2", dan seterusnya. Transformasi ini memudahkan dalam analisis pola prestasi dari semester ke semester. Setelah itu, data IPS yang telah dirapikan digabungkan (merge) dengan data lulusan berdasarkan nim agar bisa dianalisis bersama dengan atribut kelulusan, seperti waktu tempuh studi dan status kelulusan.
```
# menghitung ips
ips_df = df_transkip.groupby(['nim', 'semester']).apply(
    lambda x: round((x['nilai_grade'] * x['sks_mk']).sum() / x['sks_mk'].sum(), 2)
).reset_index(name='IPS')

# Mengubah format dari baris ke kolom
ips_df = ips_df.pivot(index='nim', columns='semester', values='IPS').reset_index()
ips_df.columns = ['nim'] + [f'IPS SMT{col}' for col in ips_df.columns[1:]]

# Gabungkan dengan data ms_lulusan berdasarkan NIM
merged_df = pd.merge(ips_df, df_lulusan, on='nim', how='left')
```
Hasil perhitungan IPS adalah sebagai berikut.

<img width="305" alt="hasil hitung IPS" src="https://github.com/user-attachments/assets/ed4f285b-ed11-4908-9197-24f5284f14bd" />

### 7. Menangani Duplikasi Nilai IPS Semester 9 
Beberapa mahasiswa memiliki nilai pada kolom `IPS SMT9`, namun nilainya identik dengan semester sebelumnya atau bahkan kosong. Untuk menjaga konsistensi data dan menghindari duplikasi informasi, dilakukan penghapusan kolom ini setelah dipastikan bahwa nilainya sudah di-backup ke kolom "IPS SMT8" jika dibutuhkan.

```
merged_df['IPS SMT9'] = merged_df.apply(lambda row: row['IPS SMT9'] if pd.isna(row['IPS SMT9']) else row['IPS SMT8'], axis=1)
merged_df.drop(columns=['IPS SMT9'], inplace=True)
```

### 8. Menghitung Lama Studi Mahasiswa dan labeling
Durasi studi mahasiswa dihitung dengan selisih antara `tgl_masuk` dan `tanggal_lulus`. Hasil perhitungan disajikan dalam format deskriptif seperti "4 Tahun 2 Bulan" dan disimpan dalam kolom baru bernama Lama Kuliah. Tujuannya adalah untuk memahami distribusi lama studi mahasiswa. Selain itu, durasi ini juga dikonversi menjadi angka desimal dalam satuan tahun (misalnya 4.2 tahun) agar bisa digunakan dalam perhitungan atau klasifikasi. Berdasarkan durasi tersebut, ditambahkan juga kolom baru bernama Lulus tepat waktu/tidak dengan nilai 1 jika mahasiswa lulus dalam rentang 3.5 hingga kurang dari 5 tahun, dan 0 jika tidak. Langkah ini penting untuk menentukan target label dalam pemodelan klasifikasi.

```
# Menghitung durasi studi masing-masing mahasiswa dalam tahun dan bulan
def calculate_study_duration(row):
    start_date = pd.to_datetime(row['tgl_masuk'])
    end_date = pd.to_datetime(row['tanggal_lulus'])
    duration = end_date - start_date

    years = duration.days // 365
    months = (duration.days % 365) // 30
    return f"{years} Tahun {months} Bulan"

# Tambahkan kolom 'Lama Kuliah'
merged_df['Lama Kuliah'] = merged_df.apply(calculate_study_duration, axis=1)

# Fungsi untuk mengonversi durasi kuliah ke tahun dengan desimal bulan
def parse_study_duration(duration_str):
    years, months = duration_str.split(' Tahun ')
    years = int(years)
    months = int(months.split(' Bulan')[0])
    return years + months / 12

# Tambahkan kolom 'Tahun Kuliah'
merged_df['Tahun Kuliah'] = merged_df['Lama Kuliah'].apply(parse_study_duration)

# Menambahkan kolom 'Lulus tepat waktu/tidak'
def durasi_kuliah(years):
    return 1 if years >= 3.5 and years < 5 else 0
merged_df['Lulus tepat waktu/tidak'] = merged_df['Tahun Kuliah'].apply(durasi_kuliah)
```
Hasilnya dapat dilihat pada Gambar berikut.

<img width="343" alt="hasil labeling" src="https://github.com/user-attachments/assets/c07e47da-e46b-421b-96f7-ae110aaa7ad7" />

### 9 Mengidentifikasi Nilai Ekstrem

Pendeteksian nilai ekstrem (outlier) dilakukan untuk mengidentifikasi data numerik yang menyimpang jauh dari sebaran umumnya. Proses ini dilakukan dengan menghitung rata-rata dan standar deviasi dari setiap kolom numerik, kemudian menentukan batas bawah dan batas atas menggunakan rumus mean ± 3 kali standar deviasi. Nilai-nilai yang berada di luar batas tersebut dianggap sebagai outlier. Visualisasi menggunakan boxplot dan scatterplot juga dilakukan untuk membantu melihat distribusi data secara lebih jelas. Tahapan ini penting agar model yang dibangun tidak terpengaruh oleh nilai-nilai yang menyimpang dan menghasilkan prediksi yang lebih akurat.

![outlier](https://github.com/user-attachments/assets/ba555027-46ed-4d91-b954-514d0ecbaca9)

### 10 Menghapus Mahasiswa Pindahan pada kolom `status_masuk`

Setelah proses pendeteksian nilai ekstrem, langkah pertama penanganan outlier adalah melakukan pembersihan data berdasarkan atribut `status_masuk`. Baris data dengan nilai status_masuk sebesar 1 dihapus dari dataset karena penelitian ini difokuskan pada mahasiswa reguler. Mahasiswa dengan status tersebut merupakan mahasiswa pindahan yang memiliki karakteristik akademik dan perjalanan studi yang berbeda, sehingga dikeluarkan untuk menjaga konsistensi analisis dan hasil model.

### 11 Memperbaiki Nilai Tidak Konsisten pada status_pegawai
Berdasarkan eksplorasi data sebelumnya, ditemukan bahwa kolom `status_pegawai` memiliki nilai yang tidak sesuai, yaitu 2, padahal seharusnya hanya terdiri dari 0 (belum bekerja) dan 1 (sudah bekerja). Setelah ditelusuri, terdapat dua baris data dengan nilai tersebut, dan mahasiswa tersebut diketahui memiliki tahun lahir 1994 dan 1997.

<img width="275" alt="STATUSPEGAWAI" src="https://github.com/user-attachments/assets/4a70a2f9-8895-4ca1-a4ca-73b5ef449e27" />

Berdasarkan asumsi bahwa mahasiswa tersebut kemungkinan telah bekerja atau mengalami gap year, maka nilai 2 tersebut diubah menjadi 1 agar sesuai dengan skema kategorisasi yang digunakan.

### 12 Melakukan Binning pada Nilai IPS  
selanjutnya untuk penanganan outlier kolom `IPS` proses binning dilakukan terhadap nilai IPS dari semester 1 hingga semester 8 dengan tujuan menyederhanakan representasi data numerik menjadi kategori. Metode binning yang digunakan adalah membagi nilai berdasarkan median: jika nilai IPS kurang dari atau sama dengan median maka dikategorikan sebagai `0`, dan jika lebih dari median dikategorikan sebagai `1`. Pendekatan ini membantu model untuk lebih mudah mengenali pola tanpa dipengaruhi oleh skala angka yang beragam.
```
# Daftar kolom IPS
ips_columns = ['IPS SMT1', 'IPS SMT2', 'IPS SMT3', 'IPS SMT4', 'IPS SMT5', 'IPS SMT6', 'IPS SMT7', 'IPS SMT8']

# Lakukan binning (≤ median → 0, > median → 1)
for col in ips_columns:
    median_value = merged_df[col].median()
    merged_df[col] = merged_df[col].apply(lambda x: 0 if x <= median_value else 1)
```
Hasilnya pada gambar berikut.

<img width="240" alt="hasil binning" src="https://github.com/user-attachments/assets/32caa39e-66cd-4efc-bb4e-a39614024878" />

### 13 Membuat Kolom Kategori Usia berdasarkan `tahun_lahir` dan lakukan Encoding 

Untuk memberikan konteks tambahan terhadap atribut `tahun_lahir`, dilakukan pengelompokan usia ke dalam tiga kategori, yaitu `Tua`, `Dewasa`, dan `Muda` berdasarkan rentang tahun lahir. Hal ini bertujuan untuk mempermudah analisis dan pemodelan dengan merepresentasikan usia dalam bentuk kategori. Setelah dikategorikan, dilakukan proses label encoding agar dapat digunakan dalam algoritma pembelajaran mesin. Kolom `kelompok_usia` kemudian disusun ulang agar posisinya berada tepat setelah kolom `tahun_lahir`.
```python
    if tahun < 1990:
        return 'Tua'
    elif tahun < 2000:
        return 'Dewasa'
    else:
        return 'Muda'
```
Hasilnya adalah sebagai berikut.

<img width="343" alt="hasil labeling" src="https://github.com/user-attachments/assets/c861b396-2f47-471e-9b3e-f701d88c3a6a" />

### 14 Pemilihan Fitur dan Split Data

sebelum pemilihan fitur, visualisasi heatmap ini dibuat untuk melihat hubungan korelasi antar fitur numerik dalam dataset. Langkah pertama adalah menyeleksi kolom-kolom dengan tipe data numerik menggunakan select_dtypes. Setelah itu, digunakan fungsi corr() untuk menghitung matriks korelasi antar kolom numerik tersebut. Hasil korelasi divisualisasikan menggunakan seaborn.heatmap dengan parameter annot=True agar nilai korelasi ditampilkan langsung pada setiap sel. Warna pada heatmap mencerminkan tingkat kekuatan korelasi, dengan skema warna coolwarm yang memudahkan identifikasi hubungan positif maupun negatif. Visualisasi ini berguna untuk memahami hubungan antar variabel dan membantu dalam proses seleksi fitur atau deteksi multikolinearitas sebelum pemodelan.

![korelasi fitur](https://github.com/user-attachments/assets/4fb253b0-d9be-4e82-932d-897cc7ccf89f)

Setelah melakukan visualisasi korelasi antar fitur, langkah selanjutnya adalah melakukan pemilihan fitur (feature selection) untuk menentukan atribut mana saja yang relevan dalam membangun model prediksi. Dalam hal ini, fitur-fitur yang tidak relevan seperti kolom target (Lulus tepat waktu/tidak), `nim`, `tgl_masuk` dan `tanggal_lulus`, `predikat`, `Lama Kuliah`, dan `IPS SMT8` dihapus dari variabel input (X). Data kemudian dibagi menjadi dua bagian, yaitu 80% data latih (training set) dan 20% data uji (testing set) secara acak, dengan pengaturan stratifikasi pada label untuk menjaga proporsi kelas tetap seimbang. 

```
X = merged_df.drop(columns=['Lulus tepat waktu/tidak', 'nim', 'status_masuk', 'tgl_masuk', 'tanggal_lulus', 'Lama Kuliah', 'predikat', 'IPS SMT8', 'tahun_lahir'])
y = merged_df['Lulus tepat waktu/tidak']

# Pembagian data 80% untuk pelatihan dan 20% untuk pengujian secara acak
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```
selanjutnya dilakukan visualisasi pada label data training untuk melihat apakah ditribusi label seimbang atau tidak. Hasil distribusi dapat dilihat pada gambar berikut.

![distribusi label tidak seimbang](https://github.com/user-attachments/assets/85bfc6a4-61f4-429a-b066-353182bd1061)

berdasarkan visualisasi tersebut terlihat label `0` atau tidak tepat wkatu sangat rendah dan tidak seimbang untuk itu perlu dilakukan penanganan agar pemodelan tetap optimal.

### 15. Penerapan Synthetic Minority Over-sampling Technique (SMOTE)

Karena data target tidak seimbang—dengan jumlah mahasiswa yang lulus tepat waktu jauh lebih banyak dibandingkan yang tidak—maka dilakukan oversampling menggunakan metode SMOTE (Synthetic Minority Over-sampling Technique). SMOTE bekerja dengan cara membuat sampel sintetis dari kelas minoritas untuk menyeimbangkan distribusi kelas. Hal ini bertujuan agar model tidak bias terhadap kelas mayoritas dan dapat belajar mengenali pola dari kedua kelas secara seimbang. Setelah diterapkan, jumlah data untuk masing-masing kelas menjadi seimbang, sebagaimana ditunjukkan oleh hasil value_counts() pada label target.
```
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
```

Hasil setelah dilakukan SMOTE adalah sebagai berikut.

![hasil smote](https://github.com/user-attachments/assets/c66c2469-ff9a-4366-962a-faa2e22f8ec7)

## Modeling
- Tanpa Hyperparameter Tuninng
  Pada tahap ini, kami menggunakan model Random Forest Classifier untuk memprediksi apakah seorang mahasiswa lulus tepat waktu atau tidak. Random Forest merupakan metode ensemble learning berbasis bagging, yang membangun sejumlah pohon keputusan (decision trees) selama pelatihan, lalu menggabungkan hasil dari masing-masing pohon untuk menghasilkan prediksi akhir melalui proses voting. Dalam proyek ini, model Random Forest diinisialisasi menggunakan parameter default, kecuali random_state yang diatur ke 42 untuk memastikan reprodusibilitas hasil. Parameter default ini berarti jumlah pohon (n_estimators) adalah 100, kedalaman maksimum pohon tidak dibatasi (None), dan parameter lainnya seperti min_samples_split, min_samples_leaf, serta max_features juga mengikuti nilai default dari library scikit-learn. Model kemudian dilatih menggunakan data latih hasil resampling dengan SMOTE untuk mengatasi ketidakseimbangan kelas.

- Pemodelan menggunakan Hyperparameter Tuning dengan Optuna
  Setelah mendapatkan baseline model dengan parameter default, dilakukan proses hyperparameter tuning untuk meningkatkan performa model. Proses ini bertujuan mencari kombinasi parameter terbaik yang dapat memaksimalkan akurasi prediksi. Untuk itu digunakan Optuna, yaitu library optimasi otomatis berbasis Bayesian Optimization yang efisien dalam melakukan pencarian ruang parameter.
  Beberapa parameter penting yang disetel meliputi: jumlah pohon (n_estimators), kedalaman maksimum pohon (max_depth), jumlah minimal sampel untuk pemisahan node (min_samples_split), jumlah minimal sampel pada daun (min_samples_leaf), serta metode pemilihan fitur (max_features) antara 'sqrt' atau 'log2'.
  ```
  params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
    }
  ```
  Fungsi objektif (objective) akan dievaluasi sebanyak 30 kali uji coba (n_trials=30), dan dari seluruh hasil percobaan, Optuna akan memilih kombinasi parameter terbaik yang menghasilkan akurasi tertinggi. Model akhir kemudian dibangun kembali menggunakan parameter terbaik tersebut dan dilatih ulang dengan data latih.

## Evaluation
### Metrik Evaluasi yang Digunakan
1. Akurasi : Mengukur proporsi prediksi yang benar terhadap seluruh data.
2. Precision: Mengukur seberapa tepat prediksi model pada masing-masing kelas. 
3. Recall: Mengukur seberapa banyak kasus positif yang berhasil ditangkap model. 
4. F1-score: Harmonik rata-rata dari precision dan recall.
   
<img width="403" alt="rumus eval" src="https://github.com/user-attachments/assets/1d695dc2-68da-4d13-95a0-4eb53a7c3a39" />


### Hasil Evaluasi
Berikut adalah perbandingan hasil model sebelum dan sesudah fine tuning menggunakan metrik akurasi, precision, recall, dan F1-score:

| Kelas | Metrik      | Sebelum Tuning | Setelah Tuning |
|-------|-------------|----------------|----------------|
| 0     | Precision   | 0.28           | 0.30           |
| 0     | Recall      | 0.54           | 0.54           |
| 0     | F1-score    | 0.37           | 0.38           |
| 1     | Precision   | 0.92           | 0.92           |
| 1     | Recall      | 0.79           | 0.81           |
| 1     | F1-score    | 0.85           | 0.86           |
| -     | Accuracy    | 0.76           | 0.77           |

###  Evaluasi terhadap Data Baru

Untuk menguji kemampuan generalisasi model, dilakukan prediksi terhadap data baru mahasiswa dengan karakteristik sebagai berikut:

| Fitur            | Nilai  |
|------------------|--------|
| IPS SMT1         | 3.70   |
| IPS SMT2         | 3.70   |
| IPS SMT3         | 3.57   |
| IPS SMT4         | 3.89   |
| IPS SMT5         | 3.71   |
| IPS SMT6         | 3.79   |
| IPS SMT7         | 3.82   |
| Prodi            | 61201  |
| Jenis Kelamin    | 0 (Perempuan) |
| Status Pegawai   | 0 (Bukan Pegawai) |
| Tahun Lahir      | 2003 (Kategori: Muda) |

Model memprediksi mahasiswa tersebut akan **Tepat Waktu** dalam menyelesaikan studinya.

<img width="398" alt="pengujian" src="https://github.com/user-attachments/assets/9b6f5418-5523-4666-8e04-532a7516f49f" />

### Interpretasi Model: Feature Importance

Berikut adalah kontribusi masing-masing fitur terhadap prediksi model berdasarkan nilai *feature importance* dari Random Forest:

![fitur penting](https://github.com/user-attachments/assets/05ad4cd7-72b4-4ae2-9a7c-18cfcc3e1da1)

Dari visualisasi tersebut, dapat disimpulkan bahwa fitur `prodi`, `kelompok_usia`, dan beberapa nilai IPS semester akhir (seperti SMT7 dan SMT4) memiliki pengaruh terbesar terhadap keputusan model.

### Kesimpulan Evaluasi
Model Random Forest menunjukkan performa yang cukup baik dalam mengklasifikasikan mahasiswa yang lulus tepat waktu dan tidak tepat waktu. Setelah dilakukan fine-tuning menggunakan Optuna, terjadi peningkatan akurasi dari **0.76** menjadi **0.77** pada data pengujian. Metrik lain seperti precision, recall, dan F1-score juga mengalami sedikit peningkatan, khususnya pada kelas minoritas (tidak tepat waktu), yang sebelumnya sulit diprediksi.
Model juga diuji terhadap satu data baru dan berhasil memprediksi bahwa mahasiswa tersebut akan **lulus tepat waktu**, sesuai dengan ekspektasi berdasarkan input nilai akademik dan demografis. Visualisasi feature importance mengungkapkan bahwa **prodi**, **kelompok usia**, dan **IPS semester 7** merupakan faktor yang paling berpengaruh terhadap prediksi model. Hal ini dapat menjadi masukan bagi pihak kampus untuk fokus pada indikator-indikator tersebut dalam upaya peningkatan ketepatan waktu kelulusan mahasiswa.

**---Ini adalah bagian akhir laporan---**
