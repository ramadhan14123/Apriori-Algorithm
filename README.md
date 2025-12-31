# Apriori Data Mining Pipeline

Proyek ini menjalankan analisis asosiasi menggunakan algoritma Apriori pada dataset transaksi biner (one-hot 0/1) dan menghasilkan:
- Frequent itemsets beserta support dan hitungannya
- Association rules (support, confidence, lift, leverage, conviction) dengan filter ambang
- Perhitungan support untuk seluruh kombinasi item yang muncul (dibatasi dengan `all_combos_max_k` bila diperlukan)
- (Opsional) Evaluasi akurasi rule pada data uji (test split)

Entrypoint utama ada di `Methode.py` (CLI). Logika dipisah dalam paket `apriori_mining/` untuk keterbacaan dan performa.

## Konfigurasi (config.json)
Semua parameter dapat diatur lewat file JSON lalu diberikan ke CLI dengan `--config`. Nilai dari CLI akan meng-override nilai di `config.json` (prioritas: CLI > config.json > default).

Contoh file `config.json`:
```json
{
  "data": "Datasets/Dataset_Preprocessed - Sheet2.csv",
  "min_support": 0.22,
  "min_confidence": 0.77,
  "min_lift": 1.0,
  "all_combos_max_k": null,
  "seed": 42,
  "test_size": 0.2,
  "progress": true,
  "no_eval": false
}
```

Keterangan setiap variabel:

- data
  - Path ke file CSV dataset (relative atau absolute). Dataset harus berupa one-hot 0/1 per item.
  - Baris pertama dianggap header (nama item). Duplikasi header (mis. nama item sama) akan ditangani otomatis.

- min_support (0–1)
  - Ambang minimum support untuk menentukan frequent itemsets.
  - 0.22 berarti 22% dari seluruh transaksi. Semakin kecil nilainya, semakin banyak itemset yang lolos (bisa lebih lambat).

- min_confidence (0–1)
  - Ambang minimum confidence untuk menyaring association rules.
  - 0.77 berarti hanya aturan dengan confidence ≥ 77% yang dipertahankan.

- min_lift (≥ 0)
  - Ambang minimum lift untuk menyaring association rules. Default 1.0 (asosiasi lebih baik dari kebetulan).
  - Naikkan ke 1.2–2.0 untuk aturan yang lebih kuat namun lebih sedikit.

- all_combos_max_k (integer | null)
  - Batas ukuran maksimum itemset saat menghitung semua kombinasi beserta support-nya.
  - null atau tidak diisi: hitung semua ukuran (bisa mahal secara waktu untuk banyak item).
  - Disarankan set ke 3–4 untuk dataset besar agar proses tetap cepat.

- seed (integer)
  - Seed untuk random number generator agar pembagian train/test reproducible.

- test_size (0–1)
  - Proporsi data yang dijadikan test set saat evaluasi rule (jika `no_eval` = false).
  - Contoh 0.2 berarti 20% data untuk test, 80% untuk train.

- progress (boolean)
  - true: tampilkan progress bar/log (tqdm). false: lebih senyap.

- no_eval (boolean)
  - true: lewati evaluasi rule pada test set (lebih cepat). false: lakukan evaluasi 80/20 (atau sesuai `test_size`).

Catatan format:
- Gunakan nilai desimal untuk persen (0.22 = 22%, 0.77 = 77%). Jangan gunakan tanda % atau 0–100.

## Cara Menjalankan (PowerShell)
Jalankan dengan `config.json` saja (disarankan):

```powershell
python .\main.py --config .\config.json
```

Atau override nilai tertentu via CLI (menggantikan nilai di config):

```powershell
python .\main.py --config .\config.json --min_support 0.15 --min_confidence 0.6
```
 
Opsi tambahan:
- `--output_all_rules` : bila diberikan, program akan menyertakan semua aturan yang mungkin dihasilkan dari frequent itemsets tanpa menyaring berdasarkan `min_confidence` atau `min_lift`. Ini berguna jika Anda ingin melakukan pemetaan/peranking sendiri. Kode penyaringan asli tetap ada (tetap di file sebagai komentar/guard) sehingga tidak hilang.
 
Contoh (tampilkan semua rule tanpa filter):

```powershell
python .\main.py --config .\config.json --output_all_rules
```

Peringatan: menampilkan semua aturan dapat menghasilkan jumlah aturan yang sangat besar (eksponensial terhadap ukuran itemset). Gunakan `all_combos_max_k` atau batasi `min_support` untuk mengendalikan ukuran keluaran.

## Output
File CSV akan disimpan di folder `Outputs/`:
- `frequent_itemsets.csv` — daftar frequent itemsets (k, itemset, support_count, support)
- `association_rules.csv` — aturan asosiasi dengan metrik (support, confidence, lift, leverage, conviction)
- `all_combinations.csv` — semua kombinasi item yang muncul beserta support
- (opsional) `rule_accuracy_test.csv` — hasil evaluasi rule pada test set

### Detail kolom pada setiap CSV

1) frequent_itemsets.csv
- k: ukuran itemset (jumlah item), mis. 2 berarti pasangan.
- itemset: daftar item dipisah titik koma `;` dan sudah diurutkan alfabetis, mis. `Roti;Mentega`.
- support_count: jumlah transaksi yang mengandung seluruh item pada itemset tersebut.
- support: proporsi transaksi yang mengandung itemset tersebut, yaitu support_count / total_transaksi (nilai 0–1).

Contoh interpretasi: baris `2,Roti;Mentega,308,0.308` berarti pasangan {Roti, Mentega} muncul pada 308 transaksi (30.8% dari keseluruhan).

2) association_rules.csv
- antecedent: bagian kiri aturan (X) dipisah `;`, mis. `Roti;Selai`.
- consequent: bagian kanan aturan (Y) dipisah `;`, mis. `Mentega`.
- support: support untuk X ∪ Y (gabungan kedua sisi), proporsi transaksi yang memuat semua item X dan Y.
- confidence: P(Y|X) = support(X ∪ Y) / support(X), peluang Y muncul ketika X muncul.
- lift: confidence / support(Y). Nilai > 1 menunjukkan asosiasi lebih kuat dari kebetulan.
- leverage: support(X ∪ Y) − support(X) × support(Y). Selisih terhadap ekspektasi kebetulan (positif = lebih sering dari kebetulan).
- conviction: (1 − support(Y)) / (1 − confidence). Lebih besar menandakan implikasi X ⇒ Y lebih kuat.

Contoh interpretasi: `Roti;Selai → Mentega` dengan support 0.272 dan confidence 0.9067 artinya 27.2% transaksi memuat ketiganya, dan ketika `Roti;Selai` muncul, 90.67% di antaranya juga memuat `Mentega`. Lift 2.05 berarti 2× lebih sering dari kebetulan.

3) all_combinations.csv
- k: ukuran kombinasi.
- itemset: kombinasi item (dipisah `;`).
- support_count: jumlah transaksi yang memuat kombinasi tersebut.
- support: proporsi kemunculan kombinasi.

Catatan: Berbeda dengan frequent_itemsets.csv yang hanya memuat itemset yang lolos ambang `min_support`, file ini mencakup semua kombinasi ber-support > 0 (dibatasi oleh `all_combos_max_k` bila diatur). Gunakan untuk eksplorasi lengkap pola yang ada.

4) rule_accuracy_test.csv (jika evaluasi diaktifkan)
- antecedent, consequent, support, confidence, lift, leverage, conviction: metrik aturan yang dihitung di data train.
- test_support: support(X ∪ Y) pada test set.
- test_confidence: confidence P(Y|X) pada test set, mengukur generalisasi aturan di data yang tidak terlihat saat training.

Catatan: Pembagian train/test ditentukan oleh `test_size` (default 0.2) dan `seed`. Metrik pada kolom tanpa prefix "test_" berasal dari model/rule yang dilatih pada data train.

## Tips & Dampak Parameter
- Turunkan `min_support` untuk menemukan pola jarang, tapi waktu komputasi bisa meningkat.
- Naikkan `min_confidence`/`min_lift` untuk aturan lebih kuat, jumlah aturan berkurang.
- Batasi `all_combos_max_k` (mis. 3–4) pada dataset besar agar perhitungan kombinasi tidak meledak eksponensial.
- Set `progress` ke false untuk run yang lebih senyap (mis. saat otomatisasi batch).

## Struktur Proyek Singkat
- `main.py` — Menjalankan semua program
- `apriori_mining/`
  - `data.py` — pembacaan data & konteks cepat (NumPy/pandas)
  - `apriori.py` — algoritma Apriori (frequent itemsets)
  - `combos.py` — perhitungan semua kombinasi
  - `rules.py` — pembentukan association rules & metrik
  - `eval.py` — evaluasi rule pada test set
  - `io_utils.py` — ekspor CSV
  - `config.py` — skema & loader konfigurasi
