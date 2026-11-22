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

Confusion matrix menunjukkan bahwa dari 329 pesanan yang benar-benar batal, model hanya mendeteksi 28 pesanan, sedangkan 301 pesanan batal lainnya terlewat. Selain itu, ada 69 pesanan yang akhirnya selesai tetapi salah diprediksi batal. Jadi, pembatalan **hanya dapat diprediksi secara terbatas**: model sedikit lebih baik daripada tebakan acak, tetapi belum cukup andal untuk dipakai sebagai sistem deteksi dini.

**4. Segmen mana yang layak diprioritaskan untuk penyesuaian kebijakan toko?**

Berdasarkan temuan eksplorasi, aspek yang layak menjadi perhatian dalam penyesuaian kebijakan toko adalah beban ongkos kirim, metode pembayaran, dan karakteristik wilayah tujuan. Pesanan dengan biaya pengiriman yang relatif tinggi perlu menjadi perhatian karena kelompok tersebut menunjukkan tingkat pembatalan lebih tinggi dibandingkan pesanan dengan biaya pengiriman normal.

Analisis kesalahan model memperkuat hal ini: satu-satunya pola yang berhasil ditangkap model adalah pesanan ke luar Pulau Jawa dengan ongkir tinggi. Pada pesanan batal yang terdeteksi, 86% tujuannya luar Jawa dan median ongkirnya Rp 61.000, dibandingkan Rp 5.000 pada seluruh pesanan test. Alasan pembatalan yang berkaitan dengan perubahan pesanan dan pembatalan otomatis juga menunjukkan perlunya evaluasi pada proses pemesanan dan pembayaran.

Dari sisi operasional, toko dapat mempertimbangkan evaluasi subsidi ongkos kirim untuk tujuan luar Jawa, transparansi estimasi biaya pengiriman, serta penyempurnaan informasi pesanan sebelum pembeli menyelesaikan transaksi. Namun, prioritas segmen secara spesifik masih memerlukan validasi lebih lanjut dengan perbandingan tingkat pembatalan per segmen pada jumlah sampel yang memadai.

## D.2 Terjemahan Metrik ke Bahasa Sehari-hari

Pada data test terdapat 3.217 pesanan, dan 329 di antaranya (10,23%) benar-benar batal.

- **Recall 0,0851**: dari setiap 100 pesanan yang akhirnya batal, model hanya berhasil mengenali sekitar 9. Sekitar 91 pembatalan lolos tanpa peringatan.
- **Precision 0,2887**: model menandai 97 pesanan sebagai berisiko batal. Dari 100 pesanan yang ditandai, hanya sekitar 29 yang benar-benar batal, dan sekitar 71 adalah salah alarm.
- **F1-Score 0,1315**: nilai gabungan precision dan recall. Skala F1 adalah 0 sampai 1, jadi 0,13 berarti keseimbangan keduanya masih sangat rendah.
- **PR-AUC 0,1679**: kemampuan model menempatkan pesanan batal di urutan atas hanya sekitar 1,6 kali lebih baik daripada tebakan acak (acak = 0,1023).
- **ROC-AUC 0,6062**: jika satu pesanan batal dan satu pesanan selesai dipilih acak, model memberi skor risiko lebih tinggi pada pesanan batal sebanyak 61% dari kasus, sedangkan tebakan acak 50%.

## D.3 Apakah Target Business Understanding Sudah Tercapai?

Kriteria berhasil pada Business Understanding adalah F1-score lebih baik daripada baseline pada data test, dengan target awal Recall ≥ 0,70 dan Precision ≥ 0,30.

| Kriteria | Target | Hasil | Status |
|---|---|---|---|
| F1 lebih baik dari baseline terbaik | > 0,186 | 0,131 | Belum tercapai |
| Recall | ≥ 0,70 | 0,085 | Belum tercapai |
| Precision | ≥ 0,30 | 0,289 | Belum tercapai (hampir) |

Baseline yang dipakai adalah "semua pesanan ditebak batal" (F1 0,186) dan "COD = batal" (F1 0,164). Keduanya mendapat F1 lebih tinggi daripada model, sehingga **target belum tercapai**. Model hanya unggul dalam precision (0,289 dibandingkan 0,102 dan 0,095 pada baseline), tetapi recall-nya jauh lebih rendah.

## D.4 Keputusan

SVM Linear dipilih sebagai model terbaik karena tiga alasan: (1) F1 test tertinggi (0,1315); (2) stabil antara data latih dan test, dengan selisih F1 hanya 0,065 (F1 latih 0,197, F1 test 0,131); dan (3) model pesaing bermasalah. KNN (k=25) mendapat F1 0,969 pada data latih tetapi hanya 0,107 pada data test, Random Forest turun dari PR-AUC 0,426 pada data latih menjadi 0,166 pada data test, dan SVM RBF memang stabil tetapi F1 test-nya hanya 0,046.

Model ini dapat dijadikan dasar awal pengembangan sistem prediksi risiko pembatalan, tetapi **belum direkomendasikan** sebagai dasar keputusan otomatis (misalnya menolak atau menahan pesanan) karena recall masih sangat rendah dan F1-nya belum mengalahkan baseline sederhana. Untuk saat ini, temuan EDA tentang ongkir tinggi dan tujuan luar Jawa lebih layak dipakai sebagai bahan kebijakan toko.

## D.5 Keterbatasan

1. **Kelas tidak seimbang dan berubah dari waktu ke waktu.** Proporsi pesanan selesai dan batal sekitar 86,26% dan 13,74%. Tingkat pembatalan turun dari 14,80% (latih) menjadi 10,23% (test), sehingga pola di data latih belum tentu sama dengan kondisi terbaru.
2. **Generalisasi terbatas.** Transaksi sangat terkonsentrasi di Pulau Jawa dan kategori produk tertentu pada satu toko, sehingga hasilnya belum tentu berlaku untuk toko lain.
3. **Kemampuan model rendah.** ROC-AUC hanya 0,61 dan recall 0,085. Pesanan batal yang terlewat (301 pesanan) memiliki ciri yang hampir sama dengan pesanan selesai (median ongkir Rp 6.500, 28% luar Jawa, dibandingkan Rp 5.000 dan 26% pada seluruh pesanan test). Penyebab pembatalan terbanyak, seperti perubahan keputusan pembeli, perubahan pesanan, dan pembatalan otomatis, tidak terlihat dari informasi saat checkout.
4. **Jumlah data test kecil untuk kelas batal.** Hanya 329 pesanan batal pada data test dengan 28 terdeteksi, sehingga metrik bisa berubah cukup besar jika periode test berbeda. Interval kepercayaan (bootstrap) belum dihitung.
5. **Eksperimen model terbatas.** Hanya lima konfigurasi yang dicoba secara manual tanpa pencarian hyperparameter yang sistematis, dan belum ada analisis kepentingan fitur (misalnya SHAP).
6. **Analisis faktor dan segmen belum tuntas.** Analisis multivariat untuk mengontrol faktor lain dan pemeringkatan segmen prioritas masih memerlukan pengujian lebih lanjut.