# AI Credit Scoring — Layer 1: P2P Lending & Multifinance

Prototipe sistem AI Credit Scoring untuk menilai kelayakan kredit nasabah **thin-file** (peminjam tanpa rekening formal atau slip gaji) di Indonesia, menggunakan 18 sinyal digital alternatif: perilaku e-wallet, income gig economy, konsistensi tagihan listrik, dan riwayat platform.

Dibangun untuk mata kuliah **Inovasi dan Entrepreneur Kecerdasan Artifisial (IEKA)** — Universitas Gadjah Mada.

---

## Fitur

| Halaman | Fungsi |
|---------|--------|
| 🏠 Overview | KPI portofolio, distribusi skor, default rate per segmen |
| 🎯 Keputusan Kredit | Form multi-step + alur visual disetujui/ditolak, SHAP bars, detail pembiayaan, narasi nasabah & audit OJK |
| 📋 Monitor Pengajuan | Riwayat pengajuan, filter multi-dimensi, ekspor CSV |
| 📈 Analitik Portofolio | NPL per segmen, P2P vs multifinance, scatter kota |
| ⚙️ Performa Model | AUC-ROC, feature importance, dokumentasi 18 fitur |

## Tech Stack

- **Model:** XGBoost + SHAP (AUC-ROC 0.812)
- **Frontend:** Streamlit
- **Data:** Pandas + CSV (dataset sintetis 2.000 nasabah)
- **Regulasi:** POJK 29/2024 (Innovative Credit Scoring)

---

## Struktur Repository

```
.
├── app.py                       # File utama aplikasi
├── requirements.txt             # Daftar dependency
└── data/
    ├── 01_nasabah_layer1.csv
    ├── 02_perilaku_digital.csv
    ├── 03_riwayat_pinjaman.csv
    └── 04_pengajuan_layer1.csv
```

> **Penting:** folder `data/` harus berada di root repository. Kode membaca file dengan path relatif `data/...`, dan Streamlit Community Cloud menjalankan app dari root repo.

---

## Menjalankan Secara Lokal

```bash
# 1. Install dependency
pip install -r requirements.txt

# 2. Jalankan app
streamlit run app.py
```

App akan terbuka otomatis di browser pada `http://localhost:8501`.

---

## Deploy Gratis ke Streamlit Community Cloud

1. Push seluruh isi folder ini ke repository GitHub (publik).
2. Buka [share.streamlit.io](https://share.streamlit.io) dan login dengan akun GitHub.
3. Klik **Create app**, pilih repository, branch `main`, dan file utama `app.py`.
4. Klik **Deploy**. App akan live dalam beberapa menit di URL `nama-app.streamlit.app`.

**Catatan deploy:**
- Repository harus publik untuk tier gratis (atau hubungkan akses repo privat).
- Limit resource ~1 GB RAM — cukup untuk dataset 2.000 baris.
- App bisa masuk mode *sleep* jika lama tidak diakses. Buka app beberapa menit sebelum demo agar sudah aktif.

---

## Catatan Penting

Sistem AI memberikan **rekomendasi pendukung**, bukan keputusan kredit final. Credit analyst tetap memiliki otoritas penuh. Setiap override tercatat otomatis dalam audit trail untuk pelaporan OJK.

Dataset yang digunakan adalah **data sintetis** untuk keperluan prototipe — bukan data nasabah nyata. Performa model (AUC 0.812) perlu divalidasi dengan data nyata melalui shadow mode sebelum digunakan secara komersial.

---

*Frans Alwan Purba · 25/563545/PPA/07116 · IEKA 2025 · Universitas Gadjah Mada*
