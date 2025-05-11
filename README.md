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
Dalam proyek ini, data yang digunakan berasal dari Universitas Buana Perjuangan Karawang dan terdiri dari dua sumber utama: data transkrip akademik dan data kelulusan mahasiswa. Data kelulusan berisi informasi demografis mahasiswa dan data transkip berisi data akademik. Dataset ini digunakan untuk membangun model klasifikasi guna memprediksi apakah seorang mahasiswa akan lulus tepat waktu atau tidak.

### 1. **Dataset Transkrip Akademik**

Dataset transkrip akademik berisi informasi nilai mahasiswa per mata kuliah selama masa studi mereka. Variabel-variabel yang ada pada dataset ini adalah sebagai berikut:

- **id**: Identifikasi unik untuk setiap entri transkrip.
- **nim**: Nomor Induk Mahasiswa yang menjadi kunci untuk menghubungkan data ini dengan dataset lainnya.
- **kode_mk**: Kode mata kuliah yang diambil oleh mahasiswa.
- **nama_mk**: Nama mata kuliah.
- **nama_mk_indo**: Nama mata kuliah dalam bahasa Indonesia.
- **nama_mk_ing**: Nama mata kuliah dalam bahasa Inggris.
- **nilai_grade**: Nilai akhir yang diberikan pada mahasiswa untuk mata kuliah tersebut.
- **nilai_total**: Total nilai yang diperoleh mahasiswa untuk mata kuliah tersebut.
- **semester**: Semester dimana mata kuliah diambil.
- **sks_mk**: Jumlah SKS untuk mata kuliah tersebut.
- **grade**: Keterangan mengenai grade yang diperoleh mahasiswa ('B+', 'A-', 'B-', 'A', 'B', 'C', 'C+', 'D')

### 2. **Dataset Data Lulusan**

Dataset data lulusan berisi informasi mengenai status kelulusan mahasiswa, termasuk apakah mereka lulus tepat waktu atau tidak. Variabel-variabel yang ada pada dataset ini adalah sebagai berikut:

- **nim**: Nomor Induk Mahasiswa yang menjadi kunci untuk menghubungkan data lulusan dengan data transkrip.
- **prodi**: Kode Program studi atau jurusan tempat mahasiswa belajar (26201, 48201, 55201, 57201, 61201, 62201, 73201, 74201, 86206,
       87205, 21201).
- **predikat**: Predikat kelulusan ('Pujian', 'Sangat Memuaskan', 'Memuaskan', '-').
- **tanggal_lulus**: Tanggal mahasiswa dinyatakan lulus.
- **tgl_masuk**: Tanggal mahasiswa mulai kuliah di universitas.
- **status_masuk**: Status masuk mahasiswa ('0': mahasiswa reguler, '1': mahasiswa pindahan).
- **jenis_kelamin**: Jenis kelamin mahasiswa ('0': Perempuan, '1': Laki-Laki).
- **tahun_lahir**: Tahun kelahiran mahasiswa.
- **status_pegawai**: Status apakah mahasiswa bekerja atau tidak selama kuliah ('0': Tidak bekerja, '1': Bekerja).

### Exploratory Data Analysis (EDA)
**1. Pemeriksaan Tipe Data**
   - Metode `info()` digunakan untuk menampilkan informasi umum tentang struktur DataFrame, termasuk jumlah entri, tipe data kolom, dan nilai non-null.
     
     ![Deskripsi](https://drive.google.com/uc?export=view&id=1YXXZwXwiB7XuDzvsUtCQRBLqNCZAsqYW) ![Deskripsi](https://drive.google.com/uc?export=view&id=11PBtpbrqNaAZbKvi4JeKPrGiuXY0DS3y)
   - Metode describe() digunakan untuk melihat ringkasan statistik dari seluruh kolom dalam dataset
     ![Deskripsi](https://drive.google.com/uc?export=view&id=1I2jrlwnJAidF9_Twyyp-9qFLa9lT_aqP)
     ![Deskripsi](https://drive.google.com/uc?export=view&id=1cpVMHKOfFFQXJ_n0itxkQVAzDckRL4Yj)
   - pengecekan nilai missing value dengan `isnull().sum()`
     
     ![Deskripsi](https://drive.google.com/uc?export=view&id=1ojdlpst1aSf0gZCRnTihu-IMS1Xjs2_2)![Deskripsi](https://drive.google.com/uc?export=view&id=1Hboito59yjyBHbjkffc64CwsSBV6rveJ)     
  - Cek duplikasi Data menggunakan `duplicated().sum()`

     ![Deskripsi](https://drive.google.com/uc?export=view&id=1qeBPVwJCpVF85Uw8LYSzN9uHbidXWTvp)
    
**Kesimpulan**
Terdapat nilai yang hilang pada data transkrip, yaitu pada kolom `nama_mk`, `nama_mk_indo`, dan `nama_mk_ing`. Selain itu, terdapat ketidaksesuaian tipe data pada kolom `tanggal_lulus` dan `tgl_masuk`, sehingga perlu dilakukan penyesuaian tipe data pada kedua kolom tersebut. Berdasarkan hasil deskripsi data, terdapat nilai `0` yang seharusnya dimulai dari `1` pada beberapa kolom, yang memerlukan koreksi. Deskripsi data lulusan juga menunjukkan hasil yang tidak normal, yang perlu diperiksa lebih lanjut untuk memastikan akurasi data. Namun, pada data ini tidak ditemukan duplikasi.

**2. Visualisasi distribusi beberapa kolom** 

   ![Distribusi Data Mahasiswa](https://drive.google.com/uc?export=view&id=1CqkNC6WPiI1a_fRV2ZnJGk3QKftVZcqS)

Berdasarkan Visualisasi di atas dapat disimpulkan bahwa
   1. Status Pegawai
Mayoritas mahasiswa dalam dataset berasal dari kategori bukan pegawai. Hanya sebagian kecil yang berstatus sebagai pegawai atau memiliki status lain. Hal ini menunjukkan bahwa sebagian besar mahasiswa kemungkinan memiliki waktu belajar yang lebih fleksibel. Namun, mahasiswa yang bekerja berpotensi memiliki keterbatasan waktu yang bisa berdampak pada proses belajar mereka. Oleh karena itu, status sebagai pegawai dapat menjadi salah satu faktor yang memengaruhi ketepatan waktu kelulusan.

2. Jenis Kelamin
Distribusi jenis kelamin menunjukkan bahwa mahasiswa dengan label 0 (kemungkinan laki-laki) lebih banyak dibandingkan label 1 (kemungkinan perempuan). Jika terdapat perbedaan dalam pola belajar atau prestasi akademik antara jenis kelamin tertentu, maka jenis kelamin bisa menjadi salah satu faktor yang turut memengaruhi ketepatan waktu kelulusan mahasiswa.

3. Status Masuk
Sebagian besar mahasiswa masuk melalui jalur yang diberi label 0, dengan sangat sedikit mahasiswa yang berasal dari jalur masuk label 1. Ini menunjukkan adanya dominasi jalur masuk tertentu yang mungkin memiliki sistem seleksi atau kesiapan akademik yang berbeda. Status masuk dapat menjadi indikator awal kesiapan mahasiswa dalam menjalani perkuliahan, yang pada akhirnya dapat berpengaruh terhadap kelulusan.

4. Program Studi (Prodi)
Terdapat ketimpangan distribusi jumlah mahasiswa antar program studi. Beberapa prodi memiliki jumlah mahasiswa yang jauh lebih banyak dibandingkan yang lain. Hal ini bisa mencerminkan perbedaan minat, kapasitas penerimaan, atau bahkan tingkat kesulitan akademik di masing-masing prodi. Program studi dengan tingkat kesulitan yang tinggi atau kurikulum yang padat bisa berdampak pada lama studi mahasiswa, sehingga berpengaruh pada kelulusan tepat waktu.

5. Grade
Distribusi nilai menunjukkan bahwa mayoritas mahasiswa mendapatkan nilai tinggi seperti A, A-, dan B+, dengan proporsi yang lebih kecil pada nilai rendah seperti C dan D. Nilai ini merupakan representasi langsung dari performa akademik mahasiswa. Mahasiswa dengan nilai yang baik kemungkinan besar memiliki pemahaman materi yang lebih kuat dan beban remedial yang lebih sedikit, sehingga lebih mungkin menyelesaikan studi tepat waktu.

## Data Preparation
Pada tahap ini, dilakukan serangkaian proses pembersihan dan penyesuaian data sebelum dianalisis lebih lanjut. Langkah-langkah yang diterapkan dijelaskan secara berurutan sesuai dengan notebook.
### 1. Menangani Missing Value pada Kolom `nama_mk`

Langkah pertama yang dilakukan adalah menangani nilai kosong (missing value) pada kolom `nama_mk` dalam data transkrip. Ditemukan bahwa terdapat nilai `null` pada kolom tersebut. Untuk dilakukan pengecekan berdasarkan `kode_mk` matakuliah tersebut yaitu `FM1190035` untuk melihat apakah ada data `nama_mk` yang terisi atau tidak. Dari Hasilnya ternyata ada data `nama_mk` yang bernilai valid.
 ![tangani null nama mk](https://drive.google.com/uc?export=view&id=1rluZ-8rQ6bFJVKSk2zmZOvUvdwSme9NH)

Hasilnya menunjukkan bahwa kode_mk tersebut memiliki nilai nama_mk pada entri lain, sehingga dapat digunakan sebagai referensi. Maka, dilakukan pengisian nilai yang kosong dengan nama mata kuliah yang sesuai:

```sh
df_transkip.loc[df_transkip['kode_mk'] == 'FM1190035', 'nama_mk'] = "FARMAKOTERAPI INFEKSI, MATA, PERNAFASAN, TULANG DAN SENDI"
```
Tahapan ini penting dilakukan agar informasi terkait nama mata kuliah tetap lengkap dan akurat, terutama jika kolom ini akan digunakan dalam analisis selanjutnya atau untuk identifikasi data.
### 2. Menghapus Kolom yang Tidak Digunakan
Dua kolom tambahan yaitu nama_mk_indo dan nama_mk_ing dihapus dari dataset. Hal ini dilakukan karena kolom-kolom tersebut tidak digunakan dalam analisis, dan penghapusannya bertujuan untuk menyederhanakan struktur data serta mengurangi redundansi.

### 3. Konversi Kolom Tanggal ke Format Datetime

Langkah selanjutnya adalah mengonversi kolom `tanggal_lulus` dan `tgl_masuk` ke dalam format datetime. Hal ini dilakukan agar data tanggal dapat diproses lebih lanjut, misalnya untuk menghitung lama studi, membuat fitur waktu, atau melakukan filtering berdasarkan periode.

Konversi dilakukan menggunakan kode berikut:

```python
# Konversi kolom 'tanggal_lulus' dan 'tgl_masuk' ke format datetime
df_lulusan['tanggal_lulus'] = pd.to_datetime(df_lulusan['tanggal_lulus'], format='%Y-%m-%d')
df_lulusan['tgl_masuk'] = pd.to_datetime(df_lulusan['tgl_masuk'], format='%Y-%m-%d')
```
Tahapan ini penting karena jika kolom tanggal tetap dalam bentuk string, maka akan menyulitkan dalam melakukan analisis berbasis waktu, seperti menghitung durasi studi.
### 4. Menangani Nilai Tidak Valid pada Kolom `semester`
Pada eksplorasi data, ditemukan bahwa beberapa baris memiliki nilai semester 0, padahal seharusnya semester dimulai dari 1. Untuk menangani hal ini, langkah awal dilakukan identifikasi seluruh kode mata kuliah yang tercatat berada di semester 0. Setelah itu, dicek apakah kode mata kuliah tersebut juga muncul di semester lain. Jika ditemukan semester yang lebih umum untuk kode tersebut, maka nilai semester 0 diganti menggunakan nilai yang paling sering muncul (modus).
```python
    semester_terbanyak = df_transkip[df_transkip['kode_mk'] == kode_mk]['semester'].mode()[0]
    df_transkip.loc[df_transkip['kode_mk'] == kode_mk, 'semester'] = semester_terbanyak
```
Namun, jika semester 0 tidak bisa diperbaiki (tidak muncul di semester lain), maka data tersebut dihapus dari dataset karena dianggap tidak valid dan dapat mengganggu hasil analisis. Langkah ini bertujuan menjaga kualitas dan integritas data agar informasi urutan studi mahasiswa tetap akurat dan logis.
### 5. Menghitung dan Menyusun Nilai IPS per Semester
Langkah selanjutnya adalah menghitung nilai IPS (Indeks Prestasi Semester) setiap mahasiswa berdasarkan transkrip nilainya. Perhitungan dilakukan dengan mengalikan bobot nilai (nilai_grade) dengan jumlah SKS mata kuliah, kemudian dibagi total SKS pada semester tersebut. Hasilnya kemudian dibulatkan ke dua desimal.

![RUMUS CARI IPS](https://drive.google.com/uc?export=view&id=1RUkpOyvGrWVLZV506XuKnkqwRs_aTZ1K)

Nilai IPS yang telah dihitung kemudian diubah dari format baris ke format kolom, sehingga setiap mahasiswa memiliki kolom "IPS SMT1", "IPS SMT2", dan seterusnya. Transformasi ini memudahkan dalam analisis pola prestasi dari semester ke semester. Setelah itu, data IPS yang telah dirapikan digabungkan (merge) dengan data lulusan berdasarkan nim agar bisa dianalisis bersama dengan atribut kelulusan, seperti waktu tempuh studi dan status kelulusan.
### 6. Menangani Duplikasi Nilai IPS Semester 9 
Beberapa mahasiswa memiliki nilai pada kolom `IPS SMT9`, namun nilainya identik dengan semester sebelumnya atau bahkan kosong. Untuk menjaga konsistensi data dan menghindari duplikasi informasi, dilakukan penghapusan kolom ini setelah dipastikan bahwa nilainya sudah di-backup ke kolom "IPS SMT8" jika dibutuhkan.
```sh
merged_df['IPS SMT9'] = merged_df.apply(lambda row: row['IPS SMT9'] if pd.isna(row['IPS SMT9']) else row['IPS SMT8'], axis=1)
merged_df.drop(columns=['IPS SMT9'], inplace=True)
```
### 7. Menghitung Lama Studi Mahasiswa
Durasi studi mahasiswa dihitung dengan selisih antara tanggal masuk (tgl_masuk) dan tanggal lulus (tanggal_lulus). Hasil perhitungan disajikan dalam format deskriptif seperti "4 Tahun 2 Bulan" dan disimpan dalam kolom baru bernama Lama Kuliah. Tujuannya adalah untuk memahami distribusi lama studi mahasiswa. Selain itu, durasi ini juga dikonversi menjadi angka desimal dalam satuan tahun (misalnya 4.2 tahun) agar bisa digunakan dalam perhitungan atau klasifikasi. Berdasarkan durasi tersebut, ditambahkan juga kolom baru bernama Lulus tepat waktu/tidak dengan nilai 1 jika mahasiswa lulus dalam rentang 3.5 hingga kurang dari 5 tahun, dan 0 jika tidak. Langkah ini penting untuk menentukan target label dalam pemodelan klasifikasi.

### 8 Mengecek Nilai Ekstrem
Pendeteksian nilai ekstrem (outlier) dilakukan untuk mengidentifikasi data numerik yang menyimpang jauh dari sebaran umumnya. Proses ini dilakukan dengan menghitung rata-rata dan standar deviasi dari setiap kolom numerik, kemudian menentukan batas bawah dan batas atas menggunakan rumus mean ± 3 kali standar deviasi. Nilai-nilai yang berada di luar batas tersebut dianggap sebagai outlier. Visualisasi menggunakan boxplot dan scatterplot juga dilakukan untuk membantu melihat distribusi data secara lebih jelas. Tahapan ini penting agar model yang dibangun tidak terpengaruh oleh nilai-nilai yang menyimpang dan menghasilkan prediksi yang lebih akurat.

### 9 Menghapus Mahasiswa Pindahan
Setelah proses pendeteksian nilai ekstrem, dilakukan pembersihan data berdasarkan atribut status_masuk. Baris data dengan nilai status_masuk sebesar 1 dihapus dari dataset karena penelitian ini difokuskan pada mahasiswa reguler. Mahasiswa dengan status tersebut merupakan mahasiswa pindahan yang memiliki karakteristik akademik dan perjalanan studi yang berbeda, sehingga dikeluarkan untuk menjaga konsistensi analisis dan hasil model.

### 10 Memperbaiki Nilai Tidak Konsisten pada status_pegawai
Berdasarkan eksplorasi data sebelumnya, ditemukan bahwa kolom `status_pegawai` memiliki nilai yang tidak sesuai, yaitu 2, padahal seharusnya hanya terdiri dari 0 (belum bekerja) dan 1 (sudah bekerja). Setelah ditelusuri, terdapat dua baris data dengan nilai tersebut, dan mahasiswa tersebut diketahui memiliki tahun lahir 1994 dan 1997.

![DATA OUTLIER STATUS PEGAWAI](https://drive.google.com/uc?export=view&id=1VkBXfNw-AMK3PcFg58lLax-qVCNtPPqi)

Berdasarkan asumsi bahwa mahasiswa tersebut kemungkinan telah bekerja atau mengalami gap year, maka nilai 2 tersebut diubah menjadi 1 agar sesuai dengan skema kategorisasi yang digunakan.

### 11 Melakukan Binning pada Nilai IPS  
Proses binning dilakukan terhadap nilai IPS dari semester 1 hingga semester 8 dengan tujuan menyederhanakan representasi data numerik menjadi kategori. Metode binning yang digunakan adalah membagi nilai berdasarkan median: jika nilai IPS kurang dari atau sama dengan median maka dikategorikan sebagai `0`, dan jika lebih dari median dikategorikan sebagai `1`. Pendekatan ini membantu model untuk lebih mudah mengenali pola tanpa dipengaruhi oleh skala angka yang beragam.

### 12 Membuat Kolom Kategori Usia  
Untuk memberikan konteks tambahan terhadap atribut `tahun_lahir`, dilakukan pengelompokan usia ke dalam tiga kategori, yaitu `Tua`, `Dewasa`, dan `Muda` berdasarkan rentang tahun lahir. Hal ini bertujuan untuk mempermudah analisis dan pemodelan dengan merepresentasikan usia dalam bentuk kategori. Setelah dikategorikan, dilakukan proses label encoding agar dapat digunakan dalam algoritma pembelajaran mesin. Kolom `kelompok_usia` kemudian disusun ulang agar posisinya berada tepat setelah kolom `tahun_lahir`.
```python
    if tahun < 1990:
        return 'Tua'
    elif tahun < 2000:
        return 'Dewasa'
    else:
        return 'Muda'
```

## Modeling
Pada tahap ini, kami menggunakan model Random Forest Classifier untuk memprediksi apakah seorang mahasiswa lulus tepat waktu atau tidak.

**Kelebihan Random Forest:**
- Dapat menangani dataset besar dan kompleks.
- Tidak rentan terhadap overfitting, terutama pada data yang besar.
- Menangani missing values dengan baik.

**Kekurangan Random Forest:**
- Pelatihan model lebih memakan waktu dibandingkan dengan model sederhana lainnya.
- Cenderung lebih sulit diinterpretasi dibandingkan dengan model yang lebih sederhana.
  
Model Random Forest Classifier dipilih karena kemampuannya dalam menangani data yang kompleks dan ketidakseimbangan kelas. proses pelatihan dimulai dengan:
- Data dipersiapkan dengan menghapus kolom-kolom yang tidak relevan dan memilih fitur yang akan digunakan untuk prediksi. Kolom target yang digunakan adalah Lulus tepat waktu/tidak.
- Dataset yang digunakan memiliki distribusi kelas yang tidak seimbang, dengan jumlah kelas 0 (tidak lulus tepat waktu) jauh lebih banyak dibandingkan kelas 1 (lulus tepat waktu). Untuk mengatasi ketidakseimbangan ini, digunakan teknik SMOTE (Synthetic Minority Over-sampling Technique) untuk meningkatkan jumlah data pada kelas minoritas sehingga distribusi kelas menjadi lebih seimbang.
- pelatihan model ini dilatih dengan data pelatihan yang sudah di-resample menggunakan SMOTE untuk menyeimbangkan distribusi kelas.
- Untuk meningkatkan performa model, dilakukan tuning hyperparameter menggunakan Optuna. Proses ini melibatkan pencarian parameter terbaik seperti jumlah estimators, kedalaman pohon, dan fitur yang digunakan pada setiap split untuk memastikan model memberikan hasil yang optimal.
  ```sh
  params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
    }
  ```

## Evaluation
### Metrik Evaluasi yang Digunakan
1. Akurasi : Mengukur proporsi prediksi yang benar terhadap seluruh data.
2. Precision: Mengukur seberapa tepat prediksi model pada masing-masing kelas. 
3. Recall: Mengukur seberapa banyak kasus positif yang berhasil ditangkap model. 
4. F1-score: Harmonik rata-rata dari precision dan recall.
   
![rumus evaluasi](https://drive.google.com/uc?export=view&id=1L3N-WAOqV15SEzJQqgoOncqu-2QVG1Qo)

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
| -     | Accuracy    | 0.7611         | 0.7743         |

### 🔍 Evaluasi terhadap Data Baru

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

![pengujian data baru](https://drive.google.com/uc?export=view&id=1uMkyqSHGBRXX4V-FDO-6aJm8o-PSvyao/view?usp=sharing)


### Interpretasi Model: Feature Importance

Berikut adalah kontribusi masing-masing fitur terhadap prediksi model berdasarkan nilai *feature importance* dari Random Forest:

![Feature Importance dari Random Forest](https://drive.google.com/uc?export=view&id=1DoVyMLlDBJ9itpiD5XqXmGIh4vwVEEQV/view?usp=sharing))

Dari visualisasi tersebut, dapat disimpulkan bahwa fitur `prodi`, `kelompok_usia`, dan beberapa nilai IPS semester akhir (seperti SMT7 dan SMT4) memiliki pengaruh terbesar terhadap keputusan model.

### Kesimpulan Evaluasi
Model Random Forest menunjukkan performa yang cukup baik dalam mengklasifikasikan mahasiswa yang lulus tepat waktu dan tidak tepat waktu. Setelah dilakukan fine-tuning menggunakan Optuna, terjadi peningkatan akurasi dari **0.7611** menjadi **0.7743** pada data pengujian. Metrik lain seperti precision, recall, dan F1-score juga mengalami sedikit peningkatan, khususnya pada kelas minoritas (tidak tepat waktu), yang sebelumnya sulit diprediksi.
Model juga diuji terhadap satu data baru dan berhasil memprediksi bahwa mahasiswa tersebut akan **lulus tepat waktu**, sesuai dengan ekspektasi berdasarkan input nilai akademik dan demografis. Visualisasi feature importance mengungkapkan bahwa **prodi**, **kelompok usia**, dan **IPS semester 7** merupakan faktor yang paling berpengaruh terhadap prediksi model. Hal ini dapat menjadi masukan bagi pihak kampus untuk fokus pada indikator-indikator tersebut dalam upaya peningkatan ketepatan waktu kelulusan mahasiswa.

**---Ini adalah bagian akhir laporan---**
