# **D. Konklusi**

Berdasarkan hasil analisis data, Exploratory Data Analysis (EDA), pemodelan, dan evaluasi pada data test, berikut jawaban atas pertanyaan pada tahap Business Understanding, terjemahan metrik, keputusan yang diambil, dan keterbatasan penelitian.

## **D.1 Jawaban Pertanyaan Business Understanding**

**1. Bagaimana pola penjualan dan pembatalan menurut waktu, wilayah, kategori, metode pembayaran, opsi kirim, ongkir, dan diskon?**

Berdasarkan hasil eksplorasi data, transaksi didominasi oleh produk perlengkapan rumah tangga, terutama kategori Celengan sebesar 26% dan Mangkok Sambal / Saus sebesar 17%. Dari sisi geografis, pesanan terkonsentrasi di Jawa Barat, Banten, dan DKI Jakarta dengan total proporsi sekitar 63,5% pada analisis keterwakilan sampel.

Dari sisi pengiriman, sekitar 82–83,5% transaksi menggunakan layanan SPX. Sementara itu, metode pembayaran COD mencakup sekitar 55% transaksi. Analisis ongkos kirim menunjukkan bahwa pesanan dengan perkiraan ongkos kirim di atas batas IQR (Rp38.000; 1.815 pesanan) memiliki tingkat pembatalan sebesar 23,97%, dibandingkan 12,58% pada pesanan normal.

Selain itu, alasan pembatalan terbanyak adalah kategori lainnya atau perubahan keputusan pembeli sebanyak 662 kasus (23,39%), diikuti perubahan pesanan sebanyak 558 kasus (19,72%) dan pembatalan otomatis oleh sistem sebanyak 447 kasus (15,80%).

Dari sisi waktu, tingkat pembatalan cenderung turun: 14,80% pada data latih (Des 2023–Mei 2025), 11,71% pada data validasi (Jun–Agu 2025), dan 10,23% pada data test (Sep–Nov 2025). Temuan ini menunjukkan adanya variasi pola transaksi dan pembatalan berdasarkan karakteristik transaksi serta periode pengamatan. Namun, pola deskriptif tersebut belum membuktikan hubungan sebab-akibat.

**2. Faktor apa yang berasosiasi dengan pembatalan setelah faktor lain dikontrol?**

Analisis awal dilakukan melalui pengujian statistik terhadap atribut pembayaran dan pengiriman, termasuk metode pembayaran, opsi pengiriman, wilayah tujuan, biaya pengiriman, berat pesanan, dan diskon. Hasil eksplorasi menunjukkan adanya perbedaan tingkat pembatalan pada beberapa kelompok karakteristik transaksi, termasuk kelompok pesanan dengan perkiraan ongkos kirim di atas batas IQR.

Namun, pengujian yang tersedia belum memberikan hasil analisis multivariat yang valid untuk menyimpulkan faktor mana yang tetap berasosiasi dengan pembatalan setelah seluruh faktor lainnya dikontrol. Hasil permutation importance pada model terpilih juga menunjukkan bahwa kontribusi fitur dalam model tidak dapat langsung diartikan sebagai hubungan independen atau sebab-akibat. Oleh karena itu, faktor yang memiliki asosiasi independen dengan pembatalan masih memerlukan pengujian lebih lanjut.

**3. Seberapa baik pembatalan dapat diprediksi dari informasi saat pesanan masuk?**

Pemodelan dilakukan menggunakan tiga algoritma, yaitu K-Nearest Neighbors (KNN), Support Vector Machine (SVM), dan Random Forest. Pencarian hyperparameter dilakukan menggunakan RandomizedSearchCV dengan temporal cross-validation pada data train, kemudian model dibandingkan menggunakan data validasi dan dievaluasi pada data test.

Berdasarkan hasil perbandingan, Run_2_SVM dipilih sebagai model terpilih di antara model yang diuji. Pemilihan tersebut mempertimbangkan hasil evaluasi pada data validasi, hasil F1-score pada threshold yang digunakan, serta pertimbangan kalibrasi probabilitas model. Model tersebut menggunakan konfigurasi SVM dengan kernel RBF, C sebesar 0,5, dan gamma auto. Hasil evaluasi pada data test adalah sebagai berikut.

| Metrik | Hasil |
|---|---|
| Threshold | 0,28 |
| F1-Score | 0,14 |
| Precision | 0,32 |
| Recall | 0,09 |
| PR-AUC | 0,17 |
| ROC-AUC | 0,61 |

Berdasarkan confusion matrix, model menghasilkan 29 True Positive, 2.826 True Negative, 62 False Positive, dan 300 False Negative pada data test.

Hasil tersebut menunjukkan bahwa SVM memiliki performa relatif dalam konteks kriteria pemilihan model yang digunakan, tetapi belum memiliki kemampuan yang memadai dalam mendeteksi sebagian besar pesanan batal. Model hanya berhasil mengenali sebagian kecil pembatalan aktual, sehingga belum cukup andal untuk digunakan sebagai sistem deteksi dini secara langsung.

**4. Segmen mana yang layak diprioritaskan untuk penyesuaian kebijakan toko?**

Berdasarkan temuan eksplorasi, beberapa karakteristik transaksi menunjukkan perbedaan tingkat pembatalan, termasuk kelompok pesanan dengan perkiraan ongkos kirim di atas batas IQR. Namun, hasil tersebut merupakan temuan deskriptif dan belum cukup untuk menetapkan segmen yang secara independen memiliki risiko pembatalan lebih tinggi.

Analisis permutation importance pada model terpilih menunjukkan bahwa kategori produk dan metode pembayaran memiliki kontribusi relatif lebih besar dibandingkan fitur lainnya berdasarkan penurunan PR-AUC. Meskipun demikian, kontribusi fitur terhadap model tidak secara langsung menunjukkan bahwa suatu kategori produk atau metode pembayaran memiliki tingkat pembatalan yang lebih tinggi.

Oleh karena itu, penyesuaian kebijakan toko sebaiknya tidak didasarkan semata-mata pada hasil prediksi model atau permutation importance. Evaluasi lebih lanjut terhadap tingkat pembatalan per kategori produk, kanal pembayaran, dan karakteristik pengiriman diperlukan sebelum menetapkan segmen prioritas secara definitif.

Dari sisi operasional, toko dapat mempertimbangkan evaluasi proses transaksi, kejelasan informasi pesanan, transparansi biaya pengiriman, serta mekanisme pembayaran. Namun, rekomendasi kebijakan pada segmen tertentu masih memerlukan validasi lebih lanjut dengan perbandingan tingkat pembatalan per segmen pada jumlah sampel yang memadai.

Dengan demikian, penyesuaian kebijakan toko sebaiknya didasarkan pada kombinasi temuan EDA dan validasi segmen lebih lanjut, bukan semata-mata berdasarkan hasil prediksi model.

## **D.2 Terjemahan Metrik ke Bahasa Sehari-hari**

Pada data test terdapat 3.217 pesanan, dan 329 di antaranya (10,23%) benar-benar batal. Model Run_2_SVM menghasilkan beberapa metrik yang menggambarkan kemampuan model dalam mendeteksi pembatalan.

- **Recall 0,09**: dari setiap 100 pesanan yang akhirnya batal, model berhasil mengenali sekitar 9 pesanan. Sebagian besar pembatalan lainnya masih belum berhasil terdeteksi. Hal ini menunjukkan bahwa model memiliki keterbatasan dalam menemukan seluruh pesanan yang berakhir batal.

- **Precision 0,32**: model menandai 91 pesanan sebagai berisiko batal. Dari jumlah tersebut, 29 pesanan benar-benar batal, sedangkan 62 pesanan sebenarnya selesai. Dengan kata lain, sekitar 32 dari setiap 100 pesanan yang ditandai berisiko batal benar-benar batal, sementara sisanya merupakan salah alarm.

- **F1-Score 0,14**: merupakan ukuran gabungan precision dan recall. Nilai F1 yang masih rendah menunjukkan bahwa keseimbangan antara kemampuan mendeteksi pembatalan dan ketepatan prediksi belum memadai.

- **PR-AUC 0,17**: menggambarkan kemampuan model dalam membedakan dan mengurutkan pesanan berdasarkan risiko pembatalan pada berbagai threshold. Nilai PR-AUC model berada di atas prevalensi kelas batal pada data test, yaitu sekitar 0,10, tetapi kemampuan tersebut masih terbatas. Model memiliki lift PR-AUC sebesar 1,65 dibandingkan prevalensi kelas batal, dengan interval bootstrap 1,44–2,01.

- **ROC-AUC 0,61**: menggambarkan kemampuan model dalam membedakan pesanan batal dan selesai berdasarkan skor prediksi pada berbagai threshold. Nilai ini menunjukkan kemampuan diskriminasi yang relatif lemah, meskipun berada di atas performa tebakan acak sebesar 0,50.

Secara keseluruhan, hasil metrik menunjukkan bahwa model masih memiliki keterbatasan dalam mengenali pembatalan secara tepat. Model dapat memberikan indikasi risiko pada sebagian pesanan, tetapi sebagian besar pembatalan masih belum berhasil dideteksi. Oleh karena itu, hasil prediksi belum cukup kuat untuk dijadikan dasar keputusan operasional secara langsung.

## **D.3 Apakah Target Business Understanding Sudah Tercapai?**

Kriteria keberhasilan pada Business Understanding ditetapkan berdasarkan tiga target evaluasi pada data test, yaitu F1-score yang lebih tinggi daripada baseline terbaik (F1-score 0,186), Recall ≥ 0,70, dan Precision ≥ 0,30.

| Kriteria | Target | Hasil | Status |
|---|---|---|---|
| F1 lebih baik dari baseline terbaik | > 0,186 | 0,14 | Belum tercapai |
| Recall | ≥ 0,70 | 0,09 | Belum tercapai |
| Precision | ≥ 0,30 | 0,32 | Tercapai |

Baseline yang digunakan adalah baseline pertama, yaitu semua pesanan ditebak batal, dengan F1-score sebesar 0,186, dan baseline kedua, yaitu COD diprediksi batal sementara selain COD diprediksi selesai, dengan F1-score sebesar 0,16. Kedua baseline tersebut memperoleh F1-score lebih tinggi dibandingkan model terpilih, sehingga target F1 belum tercapai.

Model hanya memenuhi target precision, yaitu sebesar 0,32, tetapi belum memenuhi target recall sebesar 0,70 dan belum menghasilkan F1-score yang lebih tinggi daripada baseline terbaik.

Evaluasi operating point pada data test menunjukkan bahwa target Recall ≥ 0,70 dan Precision ≥ 0,30 tidak dapat dicapai secara bersamaan berdasarkan threshold yang dievaluasi. Pada target recall minimal 0,70, precision maksimum yang dapat dicapai adalah sekitar 0,13. Sebaliknya, ketika precision dibatasi minimal 0,30, recall maksimum yang dapat dicapai hanya sekitar 0,09.

Evaluasi bootstrap juga menunjukkan bahwa selisih F1-score model terhadap baseline pertama, yaitu baseline semua pesanan ditebak batal, sebesar -0,05 dengan interval kepercayaan [-0,09; -0,00]. Hasil tersebut menunjukkan bahwa baseline pertama memiliki F1-score yang lebih tinggi dibandingkan model terpilih pada evaluasi yang dilakukan. Sementara itu, lift PR-AUC model sebesar 1,65 dengan interval kepercayaan 1,44–2,01 menunjukkan bahwa kemampuan pengurutan risiko pembatalan model berada di atas prevalensi kelas batal.

Secara keseluruhan, target Business Understanding untuk membangun model yang mampu mendeteksi risiko pembatalan secara memadai belum tercapai. Meskipun model dapat memberikan indikasi risiko pada sebagian pesanan, rendahnya recall menunjukkan bahwa sebagian besar pembatalan masih belum dapat dikenali. Selain itu, performa F1 pada data test belum mengungguli baseline terbaik.

Oleh karena itu, model belum dapat dianggap sebagai solusi prediksi yang memenuhi kebutuhan bisnis, tetapi hasil penelitian tetap memberikan gambaran mengenai keterbatasan prediksi pembatalan menggunakan atribut yang tersedia saat pesanan dibuat.

## **D.4 Keputusan**

Run_2_SVM dipilih sebagai model terpilih di antara model yang diuji berdasarkan hasil perbandingan performa pada data validasi dan pertimbangan terhadap kalibrasi probabilitas. Hasil PR-AUC SVM pada data validasi dan test relatif sama dengan Random Forest, sehingga pemilihan SVM tidak didasarkan pada keunggulan diskriminasi secara mutlak.

Pertimbangan tambahan dalam pemilihan SVM adalah hasil evaluasi kalibrasi probabilitas. Berdasarkan evaluasi yang dilakukan, SVM memperoleh Brier Score sebesar 0,09 dan ECE sebesar 0,03, sedangkan Random Forest memperoleh Brier Score sebesar 0,21 dan ECE sebesar 0,34. Hasil tersebut menunjukkan bahwa probabilitas prediksi SVM pada evaluasi yang dilakukan memiliki kesalahan kalibrasi yang lebih rendah dibandingkan Random Forest.

Namun, pemilihan SVM tidak berarti model telah memenuhi target bisnis. F1-score pada data test masih belum mengungguli baseline terbaik dan recall masih rendah, yaitu sebesar 0,09. Oleh karena itu, pemilihan model ini lebih tepat diartikan sebagai pemilihan model berdasarkan kriteria evaluasi dan pertimbangan kalibrasi yang digunakan, bukan sebagai model yang sudah optimal atau siap digunakan secara operasional.

Berdasarkan hasil penelitian, keputusan yang dapat diambil saat ini adalah:

1. **Perbaikan proses transaksi dan transparansi biaya pengiriman.** Temuan EDA menunjukkan adanya perbedaan tingkat pembatalan pada kelompok pesanan berdasarkan karakteristik ongkos kirim. Oleh karena itu, toko dapat mempertimbangkan penyajian informasi biaya pengiriman secara lebih transparan sebelum pembeli menyelesaikan transaksi.

2. **Evaluasi proses transaksi berdasarkan karakteristik pesanan.** Hasil eksplorasi dan evaluasi model dapat digunakan sebagai bahan untuk meninjau proses transaksi, informasi pesanan, dan mekanisme pembayaran. Namun, penetapan segmen prioritas berdasarkan kategori produk, metode pembayaran, atau wilayah masih memerlukan validasi tingkat pembatalan per segmen.

3. **Perbaikan proses pemesanan dan pembayaran.** Banyak pembatalan berkaitan dengan perubahan keputusan pembeli, perubahan pesanan, dan pembatalan otomatis. Toko dapat mengevaluasi kejelasan informasi pesanan, proses checkout, serta mekanisme pembayaran untuk mengurangi potensi pembatalan.

4. **Pengembangan model prediksi lebih lanjut.** Run_2_SVM belum direkomendasikan sebagai dasar keputusan otomatis, seperti menolak, menahan, atau memberikan perlakuan khusus pada pesanan. Rendahnya recall menyebabkan sebagian besar pesanan batal tidak terdeteksi, sementara false positive dapat menyebabkan pesanan yang sebenarnya selesai salah ditandai sebagai berisiko batal. Pengembangan model perlu dilakukan lebih lanjut sebelum digunakan dalam proses operasional.

Dengan demikian, hasil penelitian saat ini lebih layak dimanfaatkan sebagai bahan evaluasi proses transaksi berdasarkan temuan EDA, sementara model prediksi masih berada pada tahap pengembangan dan belum siap diterapkan sebagai sistem deteksi dini yang berdampak langsung pada keputusan terhadap pesanan.

## **D.5 Keterbatasan**

1. **Kelas tidak seimbang dan berubah dari waktu ke waktu.** Proporsi pesanan selesai dan batal sekitar 86,26% dan 13,74% pada data yang digunakan dalam pemodelan. Tingkat pembatalan turun dari 14,80% pada data latih menjadi 10,23% pada data test, sehingga pola di data latih belum tentu sama dengan kondisi terbaru.

2. **Generalisasi terbatas.** Transaksi sangat terkonsentrasi di Pulau Jawa dan kategori produk tertentu pada satu toko, sehingga hasilnya belum tentu berlaku untuk toko lain.

3. **Kemampuan model rendah.** ROC-AUC model terpilih sebesar 0,61 dan recall sebesar 0,09 menunjukkan bahwa kemampuan model dalam mendeteksi pembatalan masih terbatas. Sebanyak 300 pesanan batal pada data test belum berhasil terdeteksi. Profil kelompok False Negative memiliki karakteristik yang relatif dekat dengan seluruh pesanan test, dengan 27,33% tujuan luar Jawa dan median ongkir Rp6.500, dibandingkan 26,17% tujuan luar Jawa dan median ongkir Rp5.000 pada seluruh pesanan test. Hal ini menunjukkan bahwa sebagian pembatalan memiliki karakteristik transaksi yang sulit dibedakan dari pesanan selesai berdasarkan fitur yang tersedia saat pesanan dibuat.

4. **Jumlah data test untuk kelas batal terbatas dan label November belum sepenuhnya terminal.** Terdapat 329 pesanan batal pada data test, dengan 29 pesanan berhasil terdeteksi. Selain itu, periode November 2025 memiliki 881 pesanan terminal, sementara sekitar 250 pesanan yang masih berjalan tidak disertakan dalam evaluasi karena label akhirnya belum tersedia. Kondisi ini membatasi keterwakilan periode November dalam evaluasi dan dapat memengaruhi interpretasi hasil pada periode tersebut. Hasil evaluasi berpotensi berubah apabila data yang digunakan mencakup label final dari seluruh transaksi pada periode yang sama.

5. **Ruang pencarian hyperparameter terbatas.** Pencarian hyperparameter telah dilakukan menggunakan RandomizedSearchCV pada tiga algoritma dengan ruang parameter dan jumlah iterasi yang telah ditentukan. Meskipun pendekatan ini memungkinkan pencarian konfigurasi secara lebih sistematis dibandingkan eksperimen manual, konfigurasi terbaik yang diperoleh masih bergantung pada ruang parameter, jumlah iterasi, dan pembagian cross-validation yang digunakan. Oleh karena itu, konfigurasi yang diperoleh belum tentu merupakan konfigurasi optimal secara menyeluruh.

6. **Analisis faktor dan segmentasi belum tuntas.** Analisis multivariat untuk mengontrol faktor lain dan pemeringkatan segmen prioritas masih memerlukan pengujian lebih lanjut. Hasil permutation importance menunjukkan kontribusi relatif fitur terhadap model, tetapi belum dapat digunakan untuk menyimpulkan asosiasi independen antarvariabel. Oleh karena itu, hasil EDA belum dapat digunakan untuk menetapkan segmen prioritas secara definitif.

### **Rekomendasi Pengembangan Selanjutnya**

Berdasarkan keterbatasan yang ditemukan, beberapa pengembangan yang dapat dilakukan pada penelitian selanjutnya adalah:

1. **Penanganan ketidakseimbangan kelas:** mengeksplorasi strategi penanganan class imbalance dan mengevaluasi dampaknya terhadap recall, precision, dan F1-score secara terkontrol.

2. **Evaluasi temporal:** menggunakan data transaksi dari periode yang lebih baru dan melakukan evaluasi berkala untuk mengantisipasi perubahan karakteristik transaksi serta tingkat pembatalan.

3. **Peningkatan kemampuan model:** memperluas ruang pencarian hyperparameter, mengeksplorasi algoritma alternatif, dan mengevaluasi penyesuaian threshold berdasarkan kebutuhan operasional agar kemampuan model dalam mendeteksi pembatalan dapat ditingkatkan.

4. **Penambahan informasi prediktor:** mengevaluasi kemungkinan penggunaan fitur tambahan yang benar-benar tersedia saat pesanan dibuat, agar model memiliki informasi yang lebih memadai untuk membedakan pesanan batal dan selesai.

5. **Analisis multivariat dan segmentasi:** melakukan pengujian multivariat untuk mengevaluasi asosiasi faktor pembatalan setelah mengontrol faktor lain, serta memvalidasi segmen prioritas melalui perbandingan tingkat pembatalan dengan jumlah sampel yang memadai.

6. **Evaluasi kesiapan operasional:** sebelum model diterapkan, melakukan pengujian lebih lanjut terhadap performa model pada periode yang berbeda, kestabilan threshold, tingkat false positive, dan dampak keputusan berbasis model.