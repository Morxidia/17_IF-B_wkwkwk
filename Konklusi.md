## Konklusi

Berdasarkan hasil analisis data, Exploratory Data Analysis (EDA), dan pemodelan yang telah dilakukan, diperoleh kesimpulan untuk menjawab pertanyaan pada tahap Business Understanding sebagai berikut.

1. Bagaimana pola penjualan dan pembatalan menurut waktu, wilayah, kategori, metode pembayaran, opsi kirim, ongkir, dan diskon?

Berdasarkan hasil eksplorasi data, transaksi didominasi oleh produk perlengkapan rumah tangga, terutama kategori Celengan sebesar 26% dan Mangkok Sambal / Saus sebesar 17%. Dari sisi geografis, pesanan terkonsentrasi di Jawa Barat, Banten, dan DKI Jakarta dengan total proporsi sekitar 63,5% pada analisis keterwakilan sampel.

Dari sisi pengiriman, sekitar 82–83,5% transaksi menggunakan layanan SPX. Sementara itu, metode pembayaran COD mencakup sekitar 55% transaksi. Analisis biaya pengiriman juga menunjukkan bahwa pesanan dengan ongkos kirim yang dibayar pembeli di atas batas atas IQR memiliki tingkat pembatalan sekitar 20,7%, lebih tinggi dibandingkan pesanan normal yang sekitar 11,5%.

Selain itu, alasan pembatalan terbanyak adalah kategori lainnya atau perubahan keputusan pembeli sebanyak 662 kasus (23,39%), diikuti perubahan pesanan sebanyak 558 kasus (19,72%) dan pembatalan otomatis oleh sistem sebanyak 447 kasus (15,80%).

Temuan tersebut menunjukkan bahwa pembatalan pesanan berkaitan dengan karakteristik transaksi, pembayaran, dan pengiriman. Namun, pola deskriptif ini belum membuktikan hubungan sebab-akibat.

2. Faktor apa yang berasosiasi dengan pembatalan setelah faktor lain dikontrol?

Analisis awal dilakukan melalui pengujian statistik terhadap atribut pembayaran dan pengiriman, termasuk metode pembayaran, opsi pengiriman, wilayah tujuan, biaya pengiriman, berat pesanan, dan diskon.

Hasil eksplorasi menunjukkan bahwa beban ongkos kirim merupakan salah satu aspek yang perlu diperhatikan karena kelompok pesanan dengan ongkos kirim tinggi memiliki tingkat pembatalan lebih besar dibandingkan kelompok normal.

Namun, pengujian yang tersedia belum memberikan hasil analisis multivariat yang valid untuk menyimpulkan faktor mana yang tetap berasosiasi dengan pembatalan setelah seluruh faktor lainnya dikontrol. Oleh karena itu, temuan ini belum dapat diartikan sebagai pengaruh independen suatu variabel terhadap pembatalan.

3. Seberapa baik pembatalan dapat diprediksi dari informasi saat pesanan masuk?

Pemodelan dilakukan menggunakan tiga algoritma, yaitu K-Nearest Neighbors (KNN), Support Vector Machine (SVM), dan Random Forest, dengan total lima konfigurasi eksperimen.

Berdasarkan hasil evaluasi, SVM dengan kernel Linear (Run_4_SVM) memperoleh performa terbaik dengan nilai:
Metrik	Hasil
Threshold	0,25
F1-Score	0,1315
Precision	0,2887
Recall	    0,0851
PR-AUC	    0,1679
ROC-AUC	    0,6068

Hasil Confusion Matrix menunjukkan bahwa model berhasil mengidentifikasi 28 pesanan batal, sedangkan 301 pesanan batal lainnya tidak berhasil terdeteksi. Selain itu, terdapat 69 pesanan selesai yang salah diprediksi sebagai batal.

Nilai Recall sebesar 0,0851 menunjukkan bahwa model hanya mampu mendeteksi sekitar 8,51% dari seluruh pesanan yang benar-benar batal. Dengan demikian, SVM Linear merupakan model terbaik dari lima konfigurasi yang diuji, tetapi kemampuan prediksinya masih rendah dan belum optimal untuk diterapkan sebagai sistem deteksi dini secara langsung.

4. Segmen mana yang layak diprioritaskan untuk penyesuaian kebijakan toko?

Berdasarkan temuan eksplorasi, aspek yang layak menjadi perhatian dalam penyesuaian kebijakan toko adalah beban ongkos kirim, metode pembayaran, dan karakteristik wilayah tujuan.

Pesanan dengan biaya pengiriman yang relatif tinggi perlu menjadi perhatian karena kelompok tersebut menunjukkan tingkat pembatalan lebih tinggi dibandingkan pesanan dengan biaya pengiriman normal. Selain itu, alasan pembatalan yang berkaitan dengan perubahan pesanan dan pembatalan otomatis menunjukkan perlunya evaluasi pada proses pemesanan dan pembayaran.

Dari sisi operasional, toko dapat mempertimbangkan evaluasi subsidi ongkos kirim, transparansi estimasi biaya pengiriman, serta penyempurnaan informasi pesanan sebelum pembeli menyelesaikan transaksi.

Namun, prioritas segmen secara spesifik masih memerlukan validasi lebih lanjut menggunakan perbandingan tingkat pembatalan per segmen dengan jumlah sampel yang memadai.


# Keputusan dan Keterbatasan Penelitian

Berdasarkan keseluruhan hasil penelitian, SVM Linear dipilih sebagai model terbaik berdasarkan nilai F1-Score tertinggi sebesar 0,1315. Model ini dapat dijadikan dasar awal pengembangan sistem prediksi risiko pembatalan pesanan, tetapi belum direkomendasikan sebagai dasar pengambilan keputusan otomatis karena nilai Recall masih rendah.

Penelitian ini memiliki beberapa keterbatasan, yaitu:
1. Dataset memiliki ketidakseimbangan kelas antara pesanan selesai dan batal, dengan proporsi sekitar 86,26% pesanan selesai dan 13,74% pesanan batal.
2. Dataset memiliki konsentrasi transaksi yang tinggi pada wilayah Jawa dan kategori produk tertentu sehingga generalisasi ke toko lain masih terbatas.
3. Kemampuan model dalam mendeteksi pesanan batal masih rendah.
4. Analisis faktor yang mengontrol pengaruh variabel lain dan pemeringkatan segmen prioritas masih memerlukan pengujian lebih lanjut.