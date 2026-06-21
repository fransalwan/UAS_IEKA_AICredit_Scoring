"""
Dataset Generator — Layer 1: P2P Lending & Multifinance
Fokus: nasabah gig economy, UMKM, buruh — thin-file segment
"""
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random, os

fake = Faker('id_ID')
np.random.seed(42)
random.seed(42)

OUT = "data"
os.makedirs(OUT, exist_ok=True)

N = 2000  # lebih besar karena volume P2P jauh lebih tinggi

# Platform Layer 1
PLATFORMS = [
    ("PLT001", "Amartha Fintech",       "P2P Lending",   "UMKM mikro"),
    ("PLT002", "Modal Rakyat",          "P2P Lending",   "Pedagang & warung"),
    ("PLT003", "Adira Finance",         "Multifinance",  "Motor & elektronik"),
    ("PLT004", "BFI Finance",           "Multifinance",  "Motor & UMKM"),
    ("PLT005", "KoinWorks SME",         "P2P Lending",   "UKM & usaha kecil"),
]

# Segmen thin-file yang dominan di Layer 1
SEGMEN = [
    ("Ojek online / driver",     0.18),
    ("Pedagang pasar / warung",  0.16),
    ("Buruh pabrik",             0.14),
    ("Kurir / ekspedisi",        0.10),
    ("Freelancer digital",       0.09),
    ("Petani / nelayan",         0.08),
    ("Karyawan kontrak",         0.10),
    ("Penjahit / konveksi",      0.07),
    ("Pemilik UMKM kecil",       0.08),
]

KOTA = [
    "Jakarta Selatan","Jakarta Barat","Jakarta Timur","Depok","Bekasi",
    "Tangerang","Bandung","Surabaya","Semarang","Medan",
    "Makassar","Yogyakarta","Palembang","Pekanbaru","Bogor",
]

PRODUK = [
    ("P2P Lending",  ["Modal usaha","Renovasi","Konsumsi","Pendidikan","Kesehatan"]),
    ("Multifinance", ["Motor baru","Motor bekas","Elektronik","Mesin usaha","Refinancing"]),
]

def wc(opts):
    return random.choices([o[0] for o in opts], weights=[o[1] for o in opts], k=1)[0]

def rdate(a, b):
    return a + timedelta(days=random.randint(0, (b-a).days))

# ── 1. nasabah_layer1.csv ─────────────────────────────────────────────────
print("Generating nasabah_layer1.csv ...")
rows = []
for i in range(N):
    plt = random.choice(PLATFORMS)
    segmen = wc(SEGMEN)
    tipe = plt[2]
    usia = random.randint(19, 52)
    kota = random.choice(KOTA)
    lama_platform = random.randint(1, 36)  # bulan

    # Digital behaviour — lebih tinggi karena P2P/multifinance lebih urban
    punya_smartphone  = 1 if random.random() > 0.04 else 0
    punya_ewallet     = 1 if random.random() > 0.25 else 0
    punya_rekening    = 1 if random.random() > 0.40 else 0
    punya_kartu_kredit= 1 if random.random() > 0.75 else 0
    punya_bpjs        = 1 if random.random() > 0.35 else 0

    # Gig spesifik
    is_gig = segmen in ["Ojek online / driver","Kurir / ekspedisi","Freelancer digital"]
    punya_app_gig = 1 if (is_gig and random.random() > 0.15) else 0
    rating_gig    = round(random.uniform(3.5, 5.0), 1) if punya_app_gig else 0
    jam_kerja_gig = random.randint(4, 14) if punya_app_gig else 0

    tanggungan = random.choices([0,1,2,3,4,5], weights=[.10,.15,.25,.25,.15,.10])[0]

    rows.append({
        "nasabah_id"         : f"L1NSB{i+1:05d}",
        "platform_id"        : plt[0],
        "platform_nama"      : plt[1],
        "platform_tipe"      : plt[2],
        "platform_segmen"    : plt[3],
        "segmen_pekerjaan"   : segmen,
        "usia"               : usia,
        "kota"               : kota,
        "jumlah_tanggungan"  : tanggungan,
        "lama_di_platform_bln": lama_platform,
        "punya_smartphone"   : punya_smartphone,
        "punya_ewallet"      : punya_ewallet,
        "punya_rekening_bank": punya_rekening,
        "punya_kartu_kredit" : punya_kartu_kredit,
        "punya_bpjs"         : punya_bpjs,
        "is_gig_worker"      : int(is_gig),
        "punya_app_gig"      : punya_app_gig,
        "rating_gig_app"     : rating_gig,
        "jam_kerja_per_hari" : jam_kerja_gig,
        "tgl_bergabung"      : rdate(
            datetime(2022,1,1),
            datetime(2024,12,31) - timedelta(days=lama_platform*30)
        ).strftime("%Y-%m-%d"),
    })

df_nsb = pd.DataFrame(rows)
df_nsb.to_csv(f"{OUT}/01_nasabah_layer1.csv", index=False)
print(f"  ✓ {len(df_nsb)} baris")

# ── 2. perilaku_digital.csv ──────────────────────────────────────────────
print("Generating perilaku_digital.csv ...")
drows = []
for _, r in df_nsb.iterrows():
    is_gig = r["is_gig_worker"]

    # E-wallet activity (24 bulan)
    if r["punya_ewallet"]:
        ew_aktif_bln    = random.randint(6, 24)
        ew_trx_per_bln  = random.randint(5, 80)
        ew_avg_nominal  = random.randint(30, 800) * 1000
        ew_top_up_rutin = 1 if random.random() > 0.3 else 0
        ew_spike_sebelum= 1 if random.random() < 0.10 else 0  # red flag
        ew_saldo_avg    = random.randint(10, 500) * 1000
    else:
        ew_aktif_bln=ew_trx_per_bln=ew_avg_nominal=ew_top_up_rutin=ew_spike_sebelum=ew_saldo_avg=0

    # Gig income dari app (Gojek/Grab/Shopee)
    if r["punya_app_gig"]:
        gig_income_avg_bln = random.randint(1500, 8000) * 1000
        gig_order_per_bln  = random.randint(50, 400)
        gig_konsisten_6bln = 1 if random.random() > 0.2 else 0
        gig_cancel_rate    = round(random.uniform(0.01, 0.15), 2)
    else:
        gig_income_avg_bln=gig_order_per_bln=gig_konsisten_6bln=0
        gig_cancel_rate=0.0

    # Tagihan listrik
    listrik_tepat = random.randint(14, 24)
    listrik_pct   = round(listrik_tepat / 24 * 100, 1)

    # Belanja e-commerce (Tokopedia/Shopee)
    if r["punya_smartphone"]:
        ecom_trx_per_bln   = random.randint(0, 20)
        ecom_avg_nominal   = random.randint(50, 500) * 1000
        ecom_return_rate   = round(random.uniform(0, 0.15), 2)
        punya_toko_online  = 1 if (random.random() > 0.6 and r["segmen_pekerjaan"] == "Pemilik UMKM kecil") else 0
        omzet_toko_online  = random.randint(1, 20) * 1_000_000 if punya_toko_online else 0
    else:
        ecom_trx_per_bln=ecom_avg_nominal=0
        ecom_return_rate=0.0
        punya_toko_online=omzet_toko_online=0

    # BPJS
    bpjs_rutin = 1 if (r["punya_bpjs"] and random.random() > 0.25) else 0

    # Pulsa & data
    pulsa_per_bln = random.randint(20, 150) * 1000
    pulsa_konsisten = 1 if random.random() > 0.3 else 0

    # Riwayat di platform (on-time rate)
    platform_ontime_rate = round(random.uniform(0.55, 1.0), 2)
    platform_late_days_avg = int((1 - platform_ontime_rate) * 45)

    drows.append({
        "nasabah_id"            : r["nasabah_id"],
        # E-wallet
        "ew_aktif_bulan"        : ew_aktif_bln,
        "ew_transaksi_per_bulan": ew_trx_per_bln,
        "ew_avg_nominal"        : ew_avg_nominal,
        "ew_top_up_rutin"       : ew_top_up_rutin,
        "ew_spike_sebelum_ajuan": ew_spike_sebelum,
        "ew_saldo_rata_rata"    : ew_saldo_avg,
        # Gig income
        "gig_income_avg_bln"    : gig_income_avg_bln,
        "gig_order_per_bln"     : gig_order_per_bln,
        "gig_konsisten_6bln"    : gig_konsisten_6bln,
        "gig_cancel_rate"       : gig_cancel_rate,
        # Tagihan
        "listrik_tepat_waktu"   : listrik_tepat,
        "listrik_pct_tepat"     : listrik_pct,
        "bpjs_rutin"            : bpjs_rutin,
        # E-commerce
        "ecom_trx_per_bln"      : ecom_trx_per_bln,
        "ecom_avg_nominal"      : ecom_avg_nominal,
        "ecom_return_rate"      : ecom_return_rate,
        "punya_toko_online"     : punya_toko_online,
        "omzet_toko_online"     : omzet_toko_online,
        # Pulsa
        "pulsa_bulanan"         : pulsa_per_bln,
        "pulsa_konsisten_12bln" : pulsa_konsisten,
        # Platform behaviour
        "platform_ontime_rate"  : platform_ontime_rate,
        "platform_late_days_avg": platform_late_days_avg,
    })

df_dig = pd.DataFrame(drows)
df_dig.to_csv(f"{OUT}/02_perilaku_digital.csv", index=False)
print(f"  ✓ {len(df_dig)} baris")

# ── 3. riwayat_pinjaman.csv ───────────────────────────────────────────────
print("Generating riwayat_pinjaman.csv ...")
pinjrows = []
nsb_kredit = {}

for _, r in df_nsb.iterrows():
    n_pinj = random.choices([0,1,2,3,4], weights=[.20,.30,.28,.14,.08])[0]
    hist = []
    tipe = r["platform_tipe"]

    for k in range(n_pinj):
        kid = f"L1KRD{len(pinjrows)+1:06d}"
        tgl_cair = rdate(datetime(2022,1,1), datetime(2024,6,1))

        if tipe == "P2P Lending":
            pokok = random.choice([500,1000,2000,3000,5000,7000,10000]) * 1000
            tenor = random.choice([1,3,6,9,12])
            bunga = round(random.uniform(1.5, 3.5), 2)
        else:  # Multifinance
            pokok = random.choice([3,5,7,10,15,20,25]) * 1_000_000
            tenor = random.choice([12,18,24,36,48])
            bunga = round(random.uniform(1.0, 2.0), 2)

        produk_list = PRODUK[0][1] if tipe=="P2P Lending" else PRODUK[1][1]
        produk = random.choice(produk_list)

        # Status berdasarkan profil
        dig = df_dig[df_dig["nasabah_id"]==r["nasabah_id"]].iloc[0]
        base_lancar = 0.62
        if dig["platform_ontime_rate"] > 0.85: base_lancar += 0.12
        if dig["listrik_pct_tepat"] > 85: base_lancar += 0.06
        if r["lama_di_platform_bln"] > 12: base_lancar += 0.05
        if r["punya_rekening_bank"]: base_lancar += 0.04
        if dig["ew_aktif_bulan"] > 12: base_lancar += 0.04
        if len(hist) > 0 and hist[-1] == "Lancar": base_lancar += 0.07
        if r["jumlah_tanggungan"] >= 4: base_lancar -= 0.07
        if dig["ew_spike_sebelum_ajuan"]: base_lancar -= 0.09

        status = random.choices(
            ["Lancar","DPK","Kurang Lancar","Macet"],
            weights=[max(base_lancar,0.3), 0.15, 0.08, max(0.45-base_lancar,0.05)]
        )[0]
        hist.append(status)

        hari_telat = {
            "Lancar": random.randint(0,3),
            "DPK": random.randint(4,30),
            "Kurang Lancar": random.randint(31,90),
            "Macet": random.randint(91,365),
        }[status]

        lunas = 1 if (tgl_cair + timedelta(days=tenor*30) < datetime(2024,12,1)
                      and status in ["Lancar","DPK"]) else 0

        pinjrows.append({
            "pinjaman_id"       : kid,
            "nasabah_id"        : r["nasabah_id"],
            "platform_id"       : r["platform_id"],
            "tgl_cair"          : tgl_cair.strftime("%Y-%m-%d"),
            "tipe_produk"       : tipe,
            "nama_produk"       : produk,
            "pokok"             : pokok,
            "tenor_bulan"       : tenor,
            "bunga_per_bulan"   : bunga,
            "total_kewajiban"   : round(pokok * (1 + bunga/100 * tenor)),
            "status_kolektibilitas": status,
            "hari_keterlambatan": hari_telat,
            "sudah_lunas"       : lunas,
        })
    nsb_kredit[r["nasabah_id"]] = hist

df_pinj = pd.DataFrame(pinjrows)
df_pinj.to_csv(f"{OUT}/03_riwayat_pinjaman.csv", index=False)
print(f"  ✓ {len(df_pinj)} baris")

# ── 4. pengajuan_layer1.csv — dataset ML utama ───────────────────────────
print("Generating pengajuan_layer1.csv ...")
ajrows = []

for _, r in df_nsb.iterrows():
    aid = f"L1AJN{r['nasabah_id'][5:]}"
    dig = df_dig[df_dig["nasabah_id"]==r["nasabah_id"]].iloc[0]
    hist_pinj = df_pinj[df_pinj["nasabah_id"]==r["nasabah_id"]]

    n_pinj     = len(hist_pinj)
    n_lancar   = len(hist_pinj[hist_pinj["status_kolektibilitas"]=="Lancar"])
    n_macet    = len(hist_pinj[hist_pinj["status_kolektibilitas"]=="Macet"])
    avg_telat  = round(hist_pinj["hari_keterlambatan"].mean(), 1) if n_pinj > 0 else 0

    tipe = r["platform_tipe"]
    if tipe == "P2P Lending":
        jumlah = random.choice([500,1000,2000,3000,5000]) * 1000
        tenor  = random.choice([1,3,6,9,12])
        tujuan = random.choice(["Modal usaha","Konsumsi","Pendidikan","Renovasi"])
    else:
        jumlah = random.choice([3,5,7,10,15,20]) * 1_000_000
        tenor  = random.choice([12,18,24,36])
        tujuan = random.choice(["Motor baru","Motor bekas","Elektronik","Mesin usaha"])

    # ── Skor AI ──────────────────────────────────────────────────────
    skor = 50
    pos, neg = [], []

    if r["lama_di_platform_bln"] >= 12:
        skor += 7; pos.append(f"Aktif di platform {r['lama_di_platform_bln']} bulan (+7)")
    if dig["platform_ontime_rate"] >= 0.90:
        skor += 10; pos.append(f"On-time rate {dig['platform_ontime_rate']:.0%} (+10)")
    elif dig["platform_ontime_rate"] >= 0.75:
        skor += 5; pos.append(f"On-time rate {dig['platform_ontime_rate']:.0%} (+5)")
    if n_lancar > 0:
        add = min(n_lancar * 5, 15)
        skor += add; pos.append(f"{n_lancar} pinjaman lancar (+{add})")
    if n_pinj > 0 and n_macet == 0:
        skor += 7; pos.append("Tidak ada riwayat macet (+7)")
    if dig["listrik_pct_tepat"] >= 90:
        skor += 7; pos.append(f"Listrik {dig['listrik_pct_tepat']}% tepat waktu (+7)")
    elif dig["listrik_pct_tepat"] >= 75:
        skor += 3; pos.append(f"Listrik {dig['listrik_pct_tepat']}% tepat waktu (+3)")
    if dig["ew_aktif_bulan"] >= 12:
        skor += 5; pos.append(f"E-wallet aktif {dig['ew_aktif_bulan']} bulan (+5)")
    if dig["ew_transaksi_per_bulan"] >= 20:
        skor += 3; pos.append(f"Transaksi e-wallet {dig['ew_transaksi_per_bulan']}/bln (+3)")
    if r["punya_app_gig"] and dig["gig_konsisten_6bln"]:
        skor += 8; pos.append(f"Income gig konsisten 6 bulan (+8)")
    if r["punya_app_gig"] and r["rating_gig_app"] >= 4.5:
        skor += 5; pos.append(f"Rating gig {r['rating_gig_app']} (+5)")
    if r["punya_rekening_bank"]:
        skor += 4; pos.append("Punya rekening bank (+4)")
    if dig["bpjs_rutin"]:
        skor += 3; pos.append("BPJS dibayar rutin (+3)")
    if dig["punya_toko_online"] and dig["omzet_toko_online"] > 3_000_000:
        skor += 5; pos.append(f"Toko online omzet Rp {dig['omzet_toko_online']/1e6:.0f} jt (+5)")

    if n_macet > 0:
        kurang = min(n_macet * 12, 24)
        skor -= kurang; neg.append(f"{n_macet} pinjaman macet (-{kurang})")
    if avg_telat > 30:
        skor -= 10; neg.append(f"Rata-rata telat {avg_telat:.0f} hari (-10)")
    if r["jumlah_tanggungan"] >= 4:
        skor -= 5; neg.append(f"{r['jumlah_tanggungan']} tanggungan (-5)")
    if dig["ew_spike_sebelum_ajuan"]:
        skor -= 8; neg.append("Lonjakan top-up sebelum pengajuan (-8) ⚠️")
    if dig["gig_cancel_rate"] > 0.10:
        skor -= 4; neg.append(f"Cancel rate gig {dig['gig_cancel_rate']:.0%} (-4)")

    skor = max(10, min(95, skor + random.randint(-4, 4)))
    kat = "Rendah" if skor >= 70 else "Menengah" if skor >= 50 else "Tinggi"
    rek = "Setujui" if skor >= 50 else "Tolak"
    prob_default = max(0.02, min(0.88, (100-skor)/100 * 0.58))
    label_default = 1 if random.random() < prob_default else 0

    ajrows.append({
        "pengajuan_id"          : aid,
        "nasabah_id"            : r["nasabah_id"],
        "platform_id"           : r["platform_id"],
        "platform_tipe"         : tipe,
        "tgl_pengajuan"         : rdate(datetime(2024,1,1), datetime(2024,12,31)).strftime("%Y-%m-%d"),
        "tipe_produk"           : tipe,
        "tujuan"                : tujuan,
        "jumlah_diajukan"       : jumlah,
        "tenor_bln"             : tenor,
        # Fitur agregat
        "lama_platform_bln"     : r["lama_di_platform_bln"],
        "platform_ontime_rate"  : dig["platform_ontime_rate"],
        "n_pinjaman_sebelumnya" : n_pinj,
        "n_lancar"              : n_lancar,
        "n_macet"               : n_macet,
        "avg_telat_hari"        : avg_telat,
        "listrik_pct_tepat"     : dig["listrik_pct_tepat"],
        "ew_aktif_bln"          : dig["ew_aktif_bulan"],
        "ew_trx_per_bln"        : dig["ew_transaksi_per_bulan"],
        "ew_spike"              : dig["ew_spike_sebelum_ajuan"],
        "is_gig"                : r["is_gig_worker"],
        "gig_income_avg"        : dig["gig_income_avg_bln"],
        "gig_konsisten"         : dig["gig_konsisten_6bln"],
        "gig_cancel_rate"       : dig["gig_cancel_rate"],
        "punya_rekening"        : r["punya_rekening_bank"],
        "bpjs_rutin"            : dig["bpjs_rutin"],
        "punya_toko_online"     : dig["punya_toko_online"],
        "omzet_toko_online"     : dig["omzet_toko_online"],
        # Output
        "skor_ai"               : skor,
        "kategori_risiko"       : kat,
        "rekomendasi"           : rek,
        "prob_default"          : round(prob_default, 3),
        "label_default"         : label_default,
        "faktor_positif"        : " | ".join(pos[:4]),
        "faktor_negatif"        : " | ".join(neg[:3]),
    })

df_aj = pd.DataFrame(ajrows)
df_aj.to_csv(f"{OUT}/04_pengajuan_layer1.csv", index=False)
print(f"  ✓ {len(df_aj)} baris")

# Ringkasan
print(f"\n{'='*50}")
print(f"Default rate     : {df_aj['label_default'].mean():.1%}")
print(f"Skor rata-rata   : {df_aj['skor_ai'].mean():.1f}")
print(f"P2P volume       : {len(df_aj[df_aj['platform_tipe']=='P2P Lending']):,}")
print(f"Multifinance vol : {len(df_aj[df_aj['platform_tipe']=='Multifinance']):,}")
print(f"Gig workers      : {df_nsb['is_gig_worker'].sum():,} ({df_nsb['is_gig_worker'].mean():.0%})")
print(f"{'='*50}")
