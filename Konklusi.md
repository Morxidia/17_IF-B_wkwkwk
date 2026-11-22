# D. Konklusi

Berdasarkan hasil analisis data, Exploratory Data Analysis (EDA), pemodelan, dan evaluasi pada data test, berikut jawaban atas pertanyaan pada tahap Business Understanding, terjemahan metrik, keputusan yang diambil, dan keterbatasan penelitian.

## D.1 Jawaban Pertanyaan Business Understanding

**1. Bagaimana pola penjualan dan pembatalan menurut waktu, wilayah, kategori, metode pembayaran, opsi kirim, ongkir, dan diskon?**

Berdasarkan hasil eksplorasi data, transaksi didominasi oleh produk perlengkapan rumah tangga, terutama kategori Celengan sebesar 26% dan Mangkok Sambal / Saus sebesar 17%. Dari sisi geografis, pesanan terkonsentrasi di Jawa Barat, Banten, dan DKI Jakarta dengan total proporsi sekitar 63,5% pada analisis keterwakilan sampel.

Dari sisi pengiriman, sekitar 82–83,5% transaksi menggunakan layanan SPX. Sementara itu, metode pembayaran COD mencakup sekitar 55% transaksi. Analisis biaya pengiriman juga menunjukkan bahwa pesanan dengan ongkos kirim yang dibayar pembeli di atas batas atas IQR memiliki tingkat pembatalan sekitar 20,7%, lebih tinggi dibandingkan pesanan normal yang sekitar 11,5%.

Selain itu, alasan pembatalan terbanyak adalah kategori lainnya atau perubahan keputusan pembeli sebanyak 662 kasus (23,39%), diikuti perubahan pesanan sebanyak 558 kasus (19,72%) dan pembatalan otomatis oleh sistem sebanyak 447 kasus (15,80%).

Dari sisi waktu, tingkat pembatalan cenderung turun: 14,80% pada data latih (Des 2023–Mei 2025), 11,71% pada data validasi (Jun–Agu 2025), dan 10,23% pada data test (Sep–Nov 2025). Temuan ini menunjukkan bahwa pembatalan berkaitan dengan karakteristik transaksi, pembayaran, dan pengiriman. Namun, pola deskriptif ini belum membuktikan hubungan sebab-akibat.

**2. Faktor apa yang berasosiasi dengan pembatalan setelah faktor lain dikontrol?**

Analisis awal dilakukan melalui pengujian statistik terhadap atribut pembayaran dan pengiriman, termasuk metode pembayaran, opsi pengiriman, wilayah tujuan, biaya pengiriman, berat pesanan, dan diskon. Hasil eksplorasi menunjukkan bahwa beban ongkos kirim merupakan salah satu aspek yang perlu diperhatikan karena kelompok pesanan dengan ongkos kirim tinggi memiliki tingkat pembatalan lebih besar dibandingkan kelompok normal.

Namun, pengujian yang tersedia belum memberikan hasil analisis multivariat yang valid untuk menyimpulkan faktor mana yang tetap berasosiasi dengan pembatalan setelah seluruh faktor lainnya dikontrol. Oleh karena itu, temuan ini belum dapat diartikan sebagai pengaruh independen suatu variabel terhadap pembatalan.

**3. Seberapa baik pembatalan dapat diprediksi dari informasi saat pesanan masuk?**

Pemodelan dilakukan menggunakan tiga algoritma, yaitu K-Nearest Neighbors (KNN), Support Vector Machine (SVM), dan Random Forest. Pencarian hyperparameter dilakukan menggunakan RandomizedSearchCV dengan temporal cross-validation pada data train, kemudian model dibandingkan menggunakan data validasi dan dievaluasi pada data test.

Berdasarkan hasil perbandingan, **Run_2_SVM** dipilih sebagai model dengan performa relatif terbaik di antara model yang diuji. Model tersebut menggunakan konfigurasi SVM dengan kernel RBF, C sebesar 0,5, dan gamma auto. Hasil evaluasi pada data test adalah sebagai berikut.

| Metrik | Hasil |
|---|---|
| Threshold | 0,28 |
| F1-Score | 0,14 |
| Precision | 0,32 |
| Recall | 0,09 |
| PR-AUC | 0,17 |
| ROC-AUC | 0,61 |

Berdasarkan confusion matrix, model menghasilkan 29 True Positive, 2.826 True Negative, 62 False Positive, dan 300 False Negative pada data test.

Hasil tersebut menunjukkan bahwa model memiliki performa relatif terbaik di antara model yang diuji, tetapi masih memiliki keterbatasan dalam mendeteksi pesanan batal. Model hanya berhasil mengenali sebagian kecil pembatalan aktual, sehingga belum cukup andal untuk digunakan sebagai sistem deteksi dini secara langsung.

**4. Segmen mana yang layak diprioritaskan untuk penyesuaian kebijakan toko?**

Berdasarkan temuan eksplorasi, aspek yang layak menjadi perhatian dalam penyesuaian kebijakan toko adalah beban ongkos kirim, metode pembayaran, dan karakteristik wilayah tujuan. Pesanan dengan biaya pengiriman yang relatif tinggi perlu menjadi perhatian karena kelompok tersebut menunjukkan tingkat pembatalan lebih tinggi dibandingkan pesanan dengan biaya pengiriman normal.

Analisis kesalahan model terpilih juga menunjukkan bahwa sebagian pesanan batal yang berhasil dideteksi memiliki karakteristik tujuan luar Jawa dan ongkir relatif tinggi. Namun, karakteristik tersebut juga ditemukan pada sebagian pesanan selesai yang salah ditandai sebagai pembatalan. Oleh karena itu, pola hasil prediksi belum dapat digunakan untuk menetapkan segmen prioritas secara definitif.

Dari sisi operasional, toko dapat mempertimbangkan evaluasi subsidi ongkos kirim untuk tujuan dengan biaya pengiriman tinggi, transparansi estimasi biaya pengiriman, serta penyempurnaan informasi pesanan sebelum pembeli menyelesaikan transaksi. Namun, prioritas segmen secara spesifik masih memerlukan validasi lebih lanjut dengan perbandingan tingkat pembatalan per segmen pada jumlah sampel yang memadai.

Dengan demikian, penyesuaian kebijakan toko sebaiknya didasarkan pada kombinasi temuan EDA dan validasi segmen lebih lanjut, bukan semata-mata berdasarkan hasil prediksi model.

## D.2 Terjemahan Metrik ke Bahasa Sehari-hari

Pada data test terdapat 3.217 pesanan, dan 329 di antaranya (10,23%) benar-benar batal. Model Run_2_SVM menghasilkan beberapa metrik yang menggambarkan kemampuan model dalam mendeteksi pembatalan.

- **Recall 0,09**: dari setiap 100 pesanan yang akhirnya batal, model berhasil mengenali sekitar 9 pesanan. Sebagian besar pembatalan lainnya masih belum berhasil terdeteksi. Hal ini menunjukkan bahwa model memiliki keterbatasan dalam menemukan seluruh pesanan yang berakhir batal.

- **Precision 0,32**: model menandai 91 pesanan sebagai berisiko batal. Dari jumlah tersebut, 29 pesanan benar-benar batal, sedangkan 62 pesanan sebenarnya selesai. Dengan kata lain, sekitar 32 dari setiap 100 pesanan yang ditandai berisiko batal benar-benar batal, sementara sisanya merupakan salah alarm.

- **F1-Score 0,14**: merupakan ukuran gabungan precision dan recall. Nilai F1 yang masih rendah menunjukkan bahwa keseimbangan antara kemampuan mendeteksi pembatalan dan ketepatan prediksi belum memadai.

- **PR-AUC 0,17**: menggambarkan kemampuan model dalam membedakan dan mengurutkan pesanan berdasarkan risiko pembatalan pada berbagai threshold. Nilai PR-AUC model berada di atas prevalensi kelas batal pada data test, yaitu sekitar 0,10, tetapi kemampuan tersebut masih terbatas.

- **ROC-AUC 0,61**: menggambarkan kemampuan model dalam membedakan pesanan batal dan selesai berdasarkan skor prediksi pada berbagai threshold. Nilai ini menunjukkan kemampuan diskriminasi yang relatif lemah, meskipun berada di atas performa tebakan acak sebesar 0,50.

Secara keseluruhan, hasil metrik menunjukkan bahwa model masih memiliki keterbatasan dalam mengenali pembatalan secara tepat. Model dapat memberikan indikasi risiko pada sebagian pesanan, tetapi sebagian besar pembatalan masih belum berhasil dideteksi. Oleh karena itu, hasil prediksi belum cukup kuat untuk dijadikan dasar keputusan operasional secara langsung.

## D.3 Apakah Target Business Understanding Sudah Tercapai?

Kriteria keberhasilan pada Business Understanding adalah F1-score lebih baik daripada baseline terbaik pada data test, dengan target awal Recall ≥ 0,70 dan Precision ≥ 0,30.

| Kriteria | Target | Hasil | Status |
|---|---|---|---|
| F1 lebih baik dari baseline terbaik | > 0,186 | 0,14 | Belum tercapai |
| Recall | ≥ 0,70 | 0,09 | Belum tercapai |
| Precision | ≥ 0,30 | 0,32 | Tercapai |

Baseline yang digunakan adalah baseline pertama, yaitu semua pesanan ditebak batal, dengan F1-score sebesar 0,186, dan baseline kedua, yaitu COD diprediksi batal sementara selain COD diprediksi selesai, dengan F1-score sebesar 0,16. Kedua baseline tersebut memperoleh F1-score lebih tinggi dibandingkan model terpilih, sehingga target F1 belum tercapai.

Model hanya memenuhi target precision, yaitu sebesar 0,32, tetapi belum memenuhi target recall sebesar 0,70 dan belum menghasilkan F1-score yang lebih tinggi daripada baseline terbaik.

Secara keseluruhan, target Business Understanding untuk membangun model yang mampu mendeteksi risiko pembatalan secara memadai belum tercapai. Meskipun model dapat memberikan indikasi risiko pada sebagian pesanan, rendahnya recall menunjukkan bahwa sebagian besar pembatalan masih belum dapat dikenali. Selain itu, performa F1 pada data test belum mengungguli baseline terbaik.

Oleh karena itu, model belum dapat dianggap sebagai solusi prediksi yang memenuhi kebutuhan bisnis, tetapi hasil penelitian tetap memberikan gambaran mengenai keterbatasan prediksi pembatalan menggunakan atribut yang tersedia saat pesanan dibuat.

## D.4 Keputusan

Run_2_SVM dipilih sebagai model terbaik di antara model yang diuji berdasarkan hasil perbandingan pada data validasi dan pertimbangan terhadap hasil evaluasi pada data test. Model tersebut memperoleh F1-score validasi sebesar 0,14 dan PR-AUC validasi sebesar 0,20, serta memiliki performa relatif konsisten antara validasi dan test pada F1-score.

Namun, pemilihan Run_2_SVM tidak berarti model telah memenuhi target bisnis. F1-score pada data test masih belum mengungguli baseline terbaik dan recall masih rendah, yaitu sebesar 0,09. Oleh karena itu, pemilihan model ini lebih tepat diartikan sebagai pemilihan model dengan performa relatif terbaik di antara model yang diuji setelah tuning, bukan sebagai model yang sudah optimal atau siap digunakan secara operasional.

Berdasarkan hasil penelitian, keputusan yang dapat diambil saat ini adalah:

1. **Perbaikan proses transaksi dan transparansi biaya pengiriman.** Temuan EDA menunjukkan bahwa pesanan dengan ongkos kirim tinggi memiliki tingkat pembatalan yang lebih besar dibandingkan pesanan dengan ongkos kirim normal. Oleh karena itu, toko dapat mempertimbangkan penyajian informasi biaya pengiriman secara lebih transparan sebelum pembeli menyelesaikan transaksi.

2. **Evaluasi kebijakan pengiriman untuk wilayah dengan biaya pengiriman tinggi.** Pesanan ke luar Pulau Jawa dengan ongkos kirim tinggi terlihat pada sebagian pesanan batal yang berhasil dideteksi model. Temuan ini dapat dijadikan dasar untuk mengevaluasi pilihan layanan pengiriman atau kemungkinan subsidi ongkos kirim, tetapi perlu divalidasi lebih lanjut melalui analisis tingkat pembatalan per segmen.

3. **Perbaikan proses pemesanan dan pembayaran.** Banyak pembatalan berkaitan dengan perubahan keputusan pembeli, perubahan pesanan, dan pembatalan otomatis. Toko dapat mengevaluasi kejelasan informasi pesanan, proses checkout, serta mekanisme pembayaran untuk mengurangi potensi pembatalan.

4. **Pengembangan model prediksi lebih lanjut.** Run_2_SVM belum direkomendasikan sebagai dasar keputusan otomatis, seperti menolak, menahan, atau memberikan perlakuan khusus pada pesanan. Rendahnya recall menyebabkan sebagian besar pesanan batal tidak terdeteksi, sementara false positive dapat menyebabkan pesanan yang sebenarnya selesai salah ditandai sebagai berisiko batal. Pengembangan model perlu dilakukan lebih lanjut sebelum digunakan dalam proses operasional.

Dengan demikian, hasil penelitian saat ini lebih layak dimanfaatkan sebagai bahan evaluasi proses transaksi dan pengiriman berdasarkan temuan EDA, sementara model prediksi masih berada pada tahap pengembangan dan belum siap diterapkan sebagai sistem deteksi dini yang berdampak langsung pada keputusan terhadap pesanan.

## D.5 Keterbatasan

1. **Kelas tidak seimbang dan berubah dari waktu ke waktu.** Proporsi pesanan selesai dan batal sekitar 86,26% dan 13,74%. Tingkat pembatalan turun dari 14,80% pada data latih menjadi 10,23% pada data test, sehingga pola di data latih belum tentu sama dengan kondisi terbaru.

2. **Generalisasi terbatas.** Transaksi sangat terkonsentrasi di Pulau Jawa dan kategori produk tertentu pada satu toko, sehingga hasilnya belum tentu berlaku untuk toko lain.

3. **Kemampuan model rendah.** ROC-AUC model terpilih sebesar 0,61 dan recall sebesar 0,09 menunjukkan bahwa kemampuan model dalam mendeteksi pembatalan masih terbatas. Sebanyak 300 pesanan batal pada data test belum berhasil terdeteksi. Profil kelompok False Negative memiliki karakteristik yang relatif dekat dengan seluruh pesanan test, dengan 27,33% tujuan luar Jawa dan median ongkir Rp6.500, dibandingkan 26,17% tujuan luar Jawa dan median ongkir Rp5.000 pada seluruh pesanan test. Hal ini menunjukkan bahwa sebagian pembatalan memiliki karakteristik transaksi yang sulit dibedakan dari pesanan selesai berdasarkan fitur yang tersedia saat pesanan dibuat.

4. **Jumlah data test untuk kelas batal terbatas.** Terdapat 329 pesanan batal pada data test, dengan 29 pesanan berhasil terdeteksi. Jumlah tersebut membuat hasil evaluasi berpotensi berubah jika periode test yang digunakan berbeda.

5. **Ruang pencarian hyperparameter terbatas.** Pencarian hyperparameter telah dilakukan menggunakan RandomizedSearchCV pada tiga algoritma dengan ruang parameter dan jumlah iterasi yang telah ditentukan. Meskipun pendekatan ini memungkinkan pencarian konfigurasi secara lebih sistematis dibandingkan eksperimen manual, konfigurasi terbaik yang diperoleh masih bergantung pada ruang parameter, jumlah iterasi, dan pembagian cross-validation yang digunakan. Oleh karena itu, konfigurasi yang diperoleh belum tentu merupakan konfigurasi optimal secara menyeluruh.

6. **Analisis faktor dan segmen belum tuntas.** Analisis multivariat untuk mengontrol faktor lain dan pemeringkatan segmen prioritas masih memerlukan pengujian lebih lanjut. Oleh karena itu, hasil EDA belum dapat digunakan untuk menyimpulkan hubungan independen antarvariabel maupun menetapkan segmen prioritas secara definitif.

### Rekomendasi Pengembangan Selanjutnya

Berdasarkan keterbatasan yang ditemukan, beberapa pengembangan yang dapat dilakukan pada penelitian selanjutnya adalah:

1. **Penanganan ketidakseimbangan kelas:** mengeksplorasi strategi penanganan class imbalance dan mengevaluasi dampaknya terhadap recall, precision, dan F1-score secara terkontrol.

2. **Evaluasi temporal:** menggunakan data transaksi dari periode yang lebih baru dan melakukan evaluasi berkala untuk mengantisipasi perubahan karakteristik transaksi serta tingkat pembatalan.

3. **Peningkatan kemampuan model:** memperluas ruang pencarian hyperparameter, mengeksplorasi algoritma alternatif, dan mengevaluasi penyesuaian threshold berdasarkan kebutuhan operasional agar kemampuan model dalam mendeteksi pembatalan dapat ditingkatkan.

4. **Penambahan informasi prediktor:** mengevaluasi kemungkinan penggunaan fitur tambahan yang benar-benar tersedia saat pesanan dibuat, agar model memiliki informasi yang lebih memadai untuk membedakan pesanan batal dan selesai.

5. **Analisis multivariat dan segmentasi:** melakukan pengujian multivariat untuk mengevaluasi asosiasi faktor pembatalan setelah mengontrol faktor lain, serta memvalidasi segmen prioritas melalui perbandingan tingkat pembatalan dengan jumlah sampel yang memadai.

6. **Evaluasi kesiapan operasional:** sebelum model diterapkan, melakukan pengujian lebih lanjut terhadap performa model pada periode yang berbeda, kestabilan threshold, tingkat false positive, dan dampak keputusan berbasis model.