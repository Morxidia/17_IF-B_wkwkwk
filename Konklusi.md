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

Pemodelan dilakukan dengan tiga algoritma, yaitu K-Nearest Neighbors (KNN), Support Vector Machine (SVM), dan Random Forest, dalam lima konfigurasi eksperimen. Model terbaik adalah **SVM Linear (Run_4_SVM)** dengan hasil pada data test sebagai berikut.

| Metrik | Hasil |
|---|---|
| Threshold | 0,25 |
| F1-Score | 0,1315 |
| Precision | 0,2887 |
| Recall | 0,0851 |
| PR-AUC | 0,1679 |
| ROC-AUC | 0,6062 |

Confusion matrix menunjukkan bahwa dari 329 pesanan yang benar-benar batal, model hanya mendeteksi 28 pesanan, sedangkan 301 pesanan batal lainnya terlewat. Selain itu, terdapat 69 pesanan yang akhirnya selesai tetapi salah diprediksi batal. Jadi, pembatalan hanya dapat diprediksi secara terbatas: model memiliki kemampuan diskriminasi yang relatif lemah dan belum cukup andal untuk digunakan sebagai sistem deteksi dini.

**4. Segmen mana yang layak diprioritaskan untuk penyesuaian kebijakan toko?**

Berdasarkan temuan eksplorasi, aspek yang layak menjadi perhatian dalam penyesuaian kebijakan toko adalah beban ongkos kirim, metode pembayaran, dan karakteristik wilayah tujuan. Pesanan dengan biaya pengiriman yang relatif tinggi perlu menjadi perhatian karena kelompok tersebut menunjukkan tingkat pembatalan lebih tinggi dibandingkan pesanan dengan biaya pengiriman normal.

Analisis kesalahan model memperkuat hal ini: salah satu pola yang berhasil ditangkap model adalah pesanan ke luar Pulau Jawa dengan ongkir tinggi. Pada pesanan batal yang terdeteksi, 86% tujuannya luar Jawa dan median ongkirnya Rp61.000, dibandingkan Rp5.000 pada seluruh pesanan test. Alasan pembatalan yang berkaitan dengan perubahan pesanan dan pembatalan otomatis juga menunjukkan perlunya evaluasi pada proses pemesanan dan pembayaran.

Dari sisi operasional, toko dapat mempertimbangkan evaluasi subsidi ongkos kirim untuk tujuan luar Jawa, transparansi estimasi biaya pengiriman, serta penyempurnaan informasi pesanan sebelum pembeli menyelesaikan transaksi. Namun, prioritas segmen secara spesifik masih memerlukan validasi lebih lanjut dengan perbandingan tingkat pembatalan per segmen pada jumlah sampel yang memadai.

## D.2 Terjemahan Metrik ke Bahasa Sehari-hari

Pada data test terdapat 3.217 pesanan, dan 329 di antaranya (10,23%) benar-benar batal. Model Run_4_SVM menghasilkan beberapa metrik yang menggambarkan kemampuan model dalam mendeteksi pembatalan.

- **Recall 0,0851**: dari setiap 100 pesanan yang akhirnya batal, model hanya berhasil mengenali sekitar 9 pesanan. Sekitar 91 pembatalan lainnya tidak berhasil terdeteksi. Hal ini menunjukkan bahwa sebagian besar pembatalan masih terlewat oleh model.

- **Precision 0,2887**: model menandai 97 pesanan sebagai berisiko batal. Dari jumlah tersebut, 28 pesanan benar-benar batal, sedangkan 69 pesanan sebenarnya selesai. Dengan kata lain, dari setiap 100 pesanan yang ditandai berisiko batal, sekitar 29 benar-benar batal dan sekitar 71 merupakan salah alarm.

- **F1-Score 0,1315**: merupakan ukuran gabungan precision dan recall. Nilai F1 yang rendah menunjukkan bahwa kemampuan model dalam mendeteksi pembatalan dan menjaga ketepatan prediksi masih belum memadai.

- **PR-AUC 0,1679**: menunjukkan kemampuan model dalam membedakan dan mengurutkan pesanan berdasarkan risiko pembatalan pada berbagai threshold. Nilainya berada di atas baseline prevalensi kelas batal pada data test, yaitu sekitar 0,1023, tetapi peningkatan tersebut masih terbatas.

- **ROC-AUC 0,6062**: jika satu pesanan batal dan satu pesanan selesai dipilih secara acak, model memiliki kecenderungan memberikan skor risiko lebih tinggi pada pesanan batal sekitar 60,6% dari pasangan tersebut. Nilai ini masih menunjukkan kemampuan diskriminasi yang relatif lemah.

Secara keseluruhan, hasil metrik menunjukkan bahwa model masih memiliki keterbatasan dalam mengenali pembatalan secara tepat. Model dapat memberikan indikasi risiko pada sebagian pesanan, tetapi sebagian besar pembatalan masih belum berhasil dideteksi. Oleh karena itu, hasil prediksi belum cukup kuat untuk dijadikan dasar keputusan operasional secara langsung.

## D.3 Apakah Target Business Understanding Sudah Tercapai?

Kriteria berhasil pada Business Understanding adalah F1-score lebih baik daripada baseline pada data test, dengan target awal Recall ≥ 0,70 dan Precision ≥ 0,30.

| Kriteria | Target | Hasil | Status |
|---|---|---|---|
| F1 lebih baik dari baseline terbaik | > 0,186 | 0,1315 | Belum tercapai |
| Recall | ≥ 0,70 | 0,0851 | Belum tercapai |
| Precision | ≥ 0,30 | 0,2887 | Belum tercapai (hampir) |

Baseline yang dipakai adalah "semua pesanan ditebak batal" (F1 0,186) dan "COD = batal" (F1 0,164). Keduanya mendapat F1 lebih tinggi daripada model, sehingga target belum tercapai. Model hanya unggul dalam precision (0,2887 dibandingkan 0,102 dan 0,095 pada baseline), tetapi recall-nya jauh lebih rendah.

Secara keseluruhan, target Business Understanding untuk membangun model yang mampu mendeteksi risiko pembatalan secara memadai belum tercapai. Meskipun model dapat memberikan indikasi risiko pada sebagian pesanan, rendahnya recall menunjukkan bahwa sebagian besar pembatalan masih belum dapat dikenali. Selain itu, performa F1 pada data test belum mengungguli baseline terbaik. Oleh karena itu, model belum dapat dianggap sebagai solusi prediksi yang memenuhi kebutuhan bisnis, tetapi hasil penelitian ini tetap memberikan gambaran mengenai keterbatasan prediksi pembatalan menggunakan atribut yang tersedia saat pesanan dibuat.

## D.4 Keputusan

SVM Linear (Run_4_SVM) dipilih sebagai model terbaik di antara konfigurasi eksperimen yang telah diuji berdasarkan hasil evaluasi yang diperoleh, dengan mempertimbangkan performa relatif model pembanding dan hasil pada data test. KNN (k=25) menunjukkan indikasi overfitting yang kuat karena performa pada data train jauh lebih tinggi dibandingkan pada data test, yaitu F1-score 0,969 pada data train tetapi hanya 0,107 pada data test. Random Forest juga mengalami penurunan performa ketika dievaluasi pada data test, sedangkan SVM RBF memiliki performa test yang lebih rendah dibandingkan Run_4_SVM, dengan F1-score 0,046.

Namun, pemilihan Run_4_SVM tidak berarti model telah memenuhi target bisnis. F1-score pada data test masih belum mengungguli baseline terbaik dan recall masih sangat rendah. Oleh karena itu, pemilihan model ini lebih tepat diartikan sebagai pemilihan model dengan performa relatif terbaik di antara eksperimen yang dilakukan, bukan sebagai model yang sudah optimal atau siap digunakan secara operasional.

Berdasarkan hasil penelitian, keputusan yang dapat diambil saat ini adalah:

1. **Perbaikan proses transaksi dan transparansi biaya pengiriman.** Temuan EDA menunjukkan bahwa pesanan dengan ongkos kirim tinggi memiliki tingkat pembatalan yang lebih besar dibandingkan pesanan dengan ongkos kirim normal. Oleh karena itu, toko dapat mempertimbangkan penyajian informasi biaya pengiriman secara lebih transparan sebelum pembeli menyelesaikan transaksi.

2. **Evaluasi kebijakan pengiriman untuk wilayah dengan biaya pengiriman tinggi.** Pesanan ke luar Pulau Jawa dengan ongkos kirim tinggi terlihat pada sebagian pesanan batal yang berhasil dideteksi model. Temuan ini dapat dijadikan dasar untuk mengevaluasi pilihan layanan pengiriman atau kemungkinan subsidi ongkos kirim, tetapi perlu divalidasi lebih lanjut melalui analisis tingkat pembatalan per segmen.

3. **Perbaikan proses pemesanan dan pembayaran.** Banyak pembatalan berkaitan dengan perubahan keputusan pembeli, perubahan pesanan, dan pembatalan otomatis. Toko dapat mengevaluasi kejelasan informasi pesanan, proses checkout, serta mekanisme pembayaran untuk mengurangi potensi pembatalan.

4. **Pengembangan model prediksi lebih lanjut.** Run_4_SVM belum direkomendasikan sebagai dasar keputusan otomatis, seperti menolak, menahan, atau memberikan perlakuan khusus pada pesanan. Rendahnya recall menyebabkan sebagian besar pesanan batal tidak terdeteksi, sementara false positive dapat menyebabkan pesanan yang sebenarnya selesai salah ditandai sebagai berisiko batal. Pengembangan model perlu dilakukan lebih lanjut sebelum digunakan dalam proses operasional.

Dengan demikian, hasil penelitian saat ini lebih layak dimanfaatkan sebagai bahan evaluasi proses transaksi dan pengiriman berdasarkan temuan EDA, sementara model prediksi masih berada pada tahap pengembangan dan belum siap diterapkan sebagai sistem deteksi dini yang berdampak langsung pada keputusan terhadap pesanan.

## D.5 Keterbatasan

1. **Kelas tidak seimbang dan berubah dari waktu ke waktu.** Proporsi pesanan selesai dan batal sekitar 86,26% dan 13,74%. Tingkat pembatalan turun dari 14,80% pada data latih menjadi 10,23% pada data test, sehingga pola di data latih belum tentu sama dengan kondisi terbaru.

2. **Generalisasi terbatas.** Transaksi sangat terkonsentrasi di Pulau Jawa dan kategori produk tertentu pada satu toko, sehingga hasilnya belum tentu berlaku untuk toko lain.

3. **Kemampuan model rendah.** ROC-AUC hanya 0,6062 dan recall 0,0851. Pesanan batal yang terlewat sebanyak 301 pesanan memiliki ciri yang hampir sama dengan pesanan selesai, dengan median ongkir Rp6.500 dan 28% tujuan luar Jawa, dibandingkan median ongkir Rp5.000 dan 26% tujuan luar Jawa pada seluruh pesanan test. Penyebab pembatalan terbanyak, seperti perubahan keputusan pembeli, perubahan pesanan, dan pembatalan otomatis, tidak terlihat secara langsung dari informasi saat checkout.

4. **Jumlah data test untuk kelas batal terbatas.** Hanya terdapat 329 pesanan batal pada data test, dengan 28 pesanan berhasil terdeteksi. Jumlah tersebut membuat hasil evaluasi berpotensi berubah jika periode test yang digunakan berbeda.

5. **Eksperimen model terbatas.** Hanya lima konfigurasi yang dicoba secara manual tanpa pencarian hyperparameter yang sistematis. Eksplorasi algoritma dan pengaturan model masih dapat diperluas untuk mencari konfigurasi yang lebih sesuai dengan kebutuhan prediksi pembatalan.

6. **Analisis faktor dan segmen belum tuntas.** Analisis multivariat untuk mengontrol faktor lain dan pemeringkatan segmen prioritas masih memerlukan pengujian lebih lanjut. Oleh karena itu, hasil EDA belum dapat digunakan untuk menyimpulkan hubungan independen antarvariabel maupun menetapkan segmen prioritas secara definitif.

### Rekomendasi Pengembangan Selanjutnya

Berdasarkan keterbatasan yang ditemukan, beberapa pengembangan yang dapat dilakukan pada penelitian selanjutnya adalah:

1. **Penanganan ketidakseimbangan kelas:** mengeksplorasi strategi penanganan class imbalance dan mengevaluasi dampaknya terhadap recall, precision, dan F1-score secara terkontrol.

2. **Evaluasi temporal:** menggunakan data transaksi dari periode yang lebih baru dan melakukan evaluasi berkala untuk mengantisipasi perubahan karakteristik transaksi serta tingkat pembatalan.

3. **Peningkatan kemampuan model:** melakukan tuning hyperparameter yang lebih sistematis, eksplorasi algoritma alternatif, dan penyesuaian threshold berdasarkan kebutuhan operasional.

4. **Penambahan informasi prediktor:** mengevaluasi kemungkinan penggunaan fitur tambahan yang benar-benar tersedia saat pesanan dibuat, agar model memiliki informasi yang lebih memadai untuk membedakan pesanan batal dan selesai.

5. **Analisis multivariat dan segmentasi:** melakukan pengujian multivariat untuk mengevaluasi asosiasi faktor pembatalan setelah mengontrol faktor lain, serta memvalidasi segmen prioritas melalui perbandingan tingkat pembatalan dengan jumlah sampel yang memadai.

6. **Evaluasi kesiapan operasional:** sebelum model diterapkan, melakukan pengujian lebih lanjut terhadap performa model pada periode yang berbeda, kestabilan threshold, tingkat false positive, dan dampak keputusan berbasis model.