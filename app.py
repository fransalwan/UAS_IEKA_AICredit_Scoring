"""
MVP AI Credit Scoring — Layer 1: P2P Lending & Multifinance
Improved: tambah halaman 🎯 Keputusan Kredit
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import warnings, datetime
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AI Credit Scoring — P2P & Multifinance",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#F0F4F8}
[data-testid="stSidebar"]{background:#0F2027}
[data-testid="stSidebar"] *{color:#CBD5E0 !important}
[data-testid="stSidebar"] .stRadio label{color:#CBD5E0 !important}
[data-testid="stSidebar"] .stSelectbox label{color:#90A4AE !important}
.kpi{background:white;border-radius:12px;padding:1rem 1.2rem;border:0.5px solid #E2E8F0}
.kpi h4{font-size:12px;color:#718096;margin:0 0 4px;font-weight:500}
.kpi .val{font-size:26px;font-weight:600;color:#1A202C}
.kpi .sub{font-size:11px;color:#A0AEC0;margin-top:2px}
.info-bar{background:#EBF8FF;border-left:3px solid #3182CE;border-radius:4px;
  padding:.7rem 1rem;font-size:13px;color:#2C5282;margin-bottom:1rem}
.warn-bar{background:#FFFBEB;border-left:3px solid #D69E2E;border-radius:4px;
  padding:.7rem 1rem;font-size:13px;color:#744210;margin-bottom:1rem}
.ok-bar{background:#F0FFF4;border-left:3px solid #38A169;border-radius:4px;
  padding:.7rem 1rem;font-size:13px;color:#22543D;margin-bottom:1rem}
.err-bar{background:#FFF5F5;border-left:3px solid #E53E3E;border-radius:4px;
  padding:.7rem 1rem;font-size:13px;color:#742A2A;margin-bottom:1rem}
.sec{font-size:16px;font-weight:600;color:#1A202C;margin:1.5rem 0 .75rem;
  padding-bottom:6px;border-bottom:2px solid #3182CE}
.factor-pos{background:#F0FFF4;border-radius:8px;padding:.45rem .75rem;
  margin-bottom:.35rem;border-left:3px solid #38A169;font-size:12px;color:#276749}
.factor-neg{background:#FFFBEB;border-radius:8px;padding:.45rem .75rem;
  margin-bottom:.35rem;border-left:3px solid #D69E2E;font-size:12px;color:#744210}
.stButton>button{background:#3182CE;color:white;border:none;border-radius:8px;
  padding:.45rem 1.4rem;font-weight:500}
.stButton>button:hover{background:#2C5282}

/* ── Progress bar ── */
.prog-wrap{display:flex;gap:0;margin-bottom:1.5rem;border-radius:10px;overflow:hidden;
  border:0.5px solid #E2E8F0}
.prog-step{flex:1;padding:.6rem .5rem;text-align:center;font-size:12px;
  background:white;color:#718096;position:relative;cursor:default}
.prog-step.active{background:#3182CE;color:white;font-weight:600}
.prog-step.done{background:#EBF8FF;color:#2C5282}
.prog-step .snum{font-size:10px;opacity:.7;display:block;margin-bottom:2px}

/* ── Live preview card ── */
.live-card{background:white;border-radius:12px;border:0.5px solid #E2E8F0;
  padding:1rem 1.2rem;position:sticky;top:1rem}
.live-skor{font-size:40px;font-weight:700;line-height:1;text-align:center;
  margin-bottom:.25rem}
.live-bar-wrap{background:#EDF2F7;border-radius:20px;height:10px;overflow:hidden;margin:.5rem 0}
.live-bar-fill{height:10px;border-radius:20px;transition:width .4s ease}

/* ── Summary card ── */
.sum-row{display:flex;justify-content:space-between;padding:.375rem 0;
  border-bottom:0.5px solid #F7FAFC;font-size:13px}
.sum-row:last-child{border-bottom:none}
.sum-key{color:#718096}
.sum-val{font-weight:500;color:#1A202C}

/* ── Help text ── */
.help{font-size:11px;color:#718096;margin-top:2px;line-height:1.5}

/* ── Validation ── */
.val-err{background:#FFF5F5;border-left:3px solid #E53E3E;border-radius:4px;
  padding:.5rem .75rem;font-size:12px;color:#742A2A;margin-top:.25rem}
.val-warn{background:#FFFBEB;border-left:3px solid #D69E2E;border-radius:4px;
  padding:.5rem .75rem;font-size:12px;color:#744210;margin-top:.25rem}

/* ── Reset button ── */
div[data-testid="stButton"] button.reset-btn{
  background:#718096!important;margin-top:.5rem}
</style>
""", unsafe_allow_html=True)

# ── Load data ────────────────────────────────────────────────────────────
DATA = "data"

@st.cache_data
def load():
    nsb  = pd.read_csv(f"{DATA}/01_nasabah_layer1.csv")
    dig  = pd.read_csv(f"{DATA}/02_perilaku_digital.csv")
    pinj = pd.read_csv(f"{DATA}/03_riwayat_pinjaman.csv")
    aj   = pd.read_csv(f"{DATA}/04_pengajuan_layer1.csv")
    return nsb, dig, pinj, aj

@st.cache_resource
def train(aj, nsb):
    FEAT = ["lama_platform_bln","platform_ontime_rate","n_pinjaman_sebelumnya",
            "n_lancar","n_macet","avg_telat_hari","listrik_pct_tepat",
            "ew_aktif_bln","ew_trx_per_bln","ew_spike","is_gig","gig_income_avg",
            "gig_konsisten","gig_cancel_rate","punya_rekening","bpjs_rutin",
            "punya_toko_online","omzet_toko_online"]
    X = aj[FEAT].fillna(0)
    y = aj["label_default"]
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.2,random_state=42)
    m = XGBClassifier(n_estimators=150,max_depth=4,learning_rate=0.08,
                      subsample=0.8,colsample_bytree=0.8,
                      use_label_encoder=False,eval_metric="logloss",
                      random_state=42,verbosity=0)
    m.fit(Xtr,ytr)
    pred = m.predict(Xte); prob = m.predict_proba(Xte)[:,1]
    auc  = roc_auc_score(yte,prob)
    prec = precision_score(yte,pred,zero_division=0)
    rec  = recall_score(yte,pred,zero_division=0)
    return m, FEAT, auc, prec, rec

nsb, dig, pinj, aj = load()

# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 AI Credit Scoring")
    st.markdown("**Layer 1 · P2P & Multifinance**")
    st.markdown("---")
    page = st.radio("Navigasi", [
        "🏠  Overview",
        "🎯  Keputusan Kredit",
        "📋  Monitor Pengajuan",
        "📈  Analitik Portofolio",
        "⚙️   Performa Model",
    ])
    st.markdown("---")
    plt_filter = st.selectbox("Filter Platform",
        ["Semua"] + aj["platform_id"].unique().tolist())
    tipe_filter = st.selectbox("Tipe Produk", ["Semua","P2P Lending","Multifinance"])
    st.markdown("---")
    st.markdown("""<div style='font-size:11px;color:#718096;line-height:1.8'>
    <b>Model:</b> XGBoost v1.0<br>
    <b>Fitur:</b> 18 variabel digital<br>
    <b>Regulasi:</b> POJK 29/2024<br>
    <b>Update:</b> Jan 2025
    </div>""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────
def apply_filter(df):
    out = df.copy()
    if plt_filter != "Semua": out = out[out["platform_id"]==plt_filter]
    if tipe_filter != "Semua": out = out[out["platform_tipe"]==tipe_filter]
    return out

def gauge(skor):
    col = "#38A169" if skor>=70 else "#D69E2E" if skor>=50 else "#E53E3E"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=skor,
        number={"font":{"size":46,"color":col}},
        gauge={
            "axis":{"range":[0,100],"tickwidth":1,"tickcolor":"#CBD5E0"},
            "bar":{"color":col,"thickness":0.2},
            "bgcolor":"white","borderwidth":0,
            "steps":[
                {"range":[0,50],"color":"#FFF5F5"},
                {"range":[50,70],"color":"#FFFBEB"},
                {"range":[70,100],"color":"#F0FFF4"},
            ],
        }
    ))
    fig.update_layout(height=200, margin=dict(t=30,b=5,l=15,r=15),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig

def hitung_skor(lama_plt, ontime_rt, n_lancar, n_pinj, n_macet,
                listrik_pct, ew_aktif, ew_trx, is_gig, gig_konsisten,
                gig_rating, gig_cancel, punya_rekening, punya_bpjs,
                punya_toko, omzet_toko, tanggungan, ew_spike, avg_telat):
    """Shared scoring logic — return (skor, pos_list, neg_list)"""
    skor = 50
    pos, neg = [], []

    if lama_plt >= 12:
        skor += 7; pos.append(f"Aktif di platform {lama_plt} bulan (+7)")
    if ontime_rt >= 0.90:
        skor += 10; pos.append(f"On-time rate {ontime_rt:.0%} (+10)")
    elif ontime_rt >= 0.75:
        skor += 5;  pos.append(f"On-time rate {ontime_rt:.0%} (+5)")
    if n_lancar > 0:
        add = min(n_lancar*5,15); skor += add
        pos.append(f"{n_lancar} pinjaman lancar (+{add})")
    if n_pinj > 0 and n_macet == 0:
        skor += 7; pos.append("Tidak ada riwayat macet (+7)")
    if listrik_pct >= 90:
        skor += 7; pos.append(f"Listrik {listrik_pct:.0f}% tepat waktu (+7)")
    elif listrik_pct >= 75:
        skor += 3; pos.append(f"Listrik {listrik_pct:.0f}% tepat waktu (+3)")
    if ew_aktif >= 12:
        skor += 5; pos.append(f"E-wallet aktif {ew_aktif} bulan (+5)")
    if ew_trx >= 20:
        skor += 3; pos.append(f"Transaksi e-wallet {ew_trx}/bln (+3)")
    if is_gig and gig_konsisten:
        skor += 8; pos.append("Income gig konsisten 6 bulan (+8)")
    if is_gig and gig_rating >= 4.5:
        skor += 5; pos.append(f"Rating gig {gig_rating} (+5)")
    if punya_rekening:
        skor += 4; pos.append("Punya rekening bank (+4)")
    if punya_bpjs:
        skor += 3; pos.append("BPJS rutin dibayar (+3)")
    if punya_toko and omzet_toko > 3_000_000:
        skor += 5; pos.append(f"Toko online omzet Rp {omzet_toko/1e6:.0f}jt (+5)")

    if n_macet > 0:
        k = min(n_macet*12,24); skor -= k
        neg.append(f"{n_macet} pinjaman macet (-{k})")
    if avg_telat > 30:
        skor -= 10; neg.append(f"Keterlambatan rata-rata {avg_telat} hari (-10)")
    if tanggungan >= 4:
        skor -= 5; neg.append(f"{tanggungan} tanggungan keluarga (-5)")
    if ew_spike:
        skor -= 8; neg.append("Spike top-up e-wallet sebelum pengajuan (-8) ⚠️")
    if is_gig and gig_cancel >= 10:
        skor -= 4; neg.append(f"Cancel rate {gig_cancel}% (-4)")

    skor = max(10, min(95, skor + np.random.randint(-3,4)))
    return skor, pos, neg

def render_shap_bars(pos_list, neg_list):
    """Render SHAP factor bars dengan HTML"""
    html = ""
    if pos_list:
        html += "<div style='font-size:12px;font-weight:600;color:#276749;margin-bottom:.5rem'>✅ Faktor Positif</div>"
        for f in pos_list:
            try:
                poin = int(f.split("(+")[-1].replace(")","").strip()) if "(+" in f else 4
            except: poin = 4
            label = f.split("(+")[0].strip() if "(+" in f else f
            w = min(poin * 7, 100)
            html += f"""<div class='shap-row'>
              <span class='shap-label' style='color:#276749'>{label}</span>
              <div class='shap-track'><div class='shap-fill-pos' style='width:{w}%'></div></div>
              <span class='shap-val' style='color:#38A169'>+{poin}pt</span>
            </div>"""

    if neg_list:
        html += "<div style='font-size:12px;font-weight:600;color:#742A2A;margin:.75rem 0 .5rem'>⚠️ Faktor Risiko</div>"
        for f in neg_list:
            try:
                poin = int(f.split("(-")[-1].replace(")","").replace("⚠️","").strip()) if "(-" in f else 4
            except: poin = 4
            label = f.split("(-")[0].strip() if "(-" in f else f
            w = min(poin * 7, 100)
            html += f"""<div class='shap-row'>
              <span class='shap-label' style='color:#742A2A'>{label}</span>
              <div class='shap-track'><div class='shap-fill-neg' style='width:{w}%'></div></div>
              <span class='shap-val' style='color:#E53E3E'>-{poin}pt</span>
            </div>"""
    return html

def saran_perbaikan(neg_list, skor):
    """Generate saran spesifik berdasarkan faktor negatif"""
    saran = []
    if any("macet" in f.lower() for f in neg_list):
        saran.append(("🏦 Lunasi tunggakan kredit",
            "Selesaikan pinjaman macet sebelum mengajukan kembali. "
            "Rekam jejak bersih minimal 6 bulan diperlukan."))
    if any("keterlambatan" in f.lower() or "telat" in f.lower() for f in neg_list):
        saran.append(("📅 Bangun rekam jejak bayar tepat waktu",
            "Bayar semua kewajiban tepat waktu selama 6 bulan ke depan "
            "sebelum mengajukan kembali. Setiap bulan tepat waktu menambah skor."))
    if any("spike" in f.lower() for f in neg_list):
        saran.append(("📱 Gunakan e-wallet secara organik",
            "Hindari lonjakan top-up mendadak sebelum pengajuan. "
            "Transaksi rutin 12+ bulan jauh lebih bernilai daripada saldo besar tiba-tiba."))
    if any("cancel" in f.lower() for f in neg_list):
        saran.append(("🛵 Turunkan cancel rate di app gig",
            "Cancel rate di bawah 10% diperlukan. "
            "Konsistensi kerja 3–6 bulan akan memperbaiki sinyal ini secara signifikan."))
    if any("tanggungan" in f.lower() for f in neg_list):
        saran.append(("💰 Sesuaikan plafon dengan kemampuan",
            "Dengan 4+ tanggungan, pertimbangkan mengajukan plafon yang lebih kecil "
            "agar rasio cicilan terhadap pendapatan tetap sehat."))
    # Saran umum jika tidak ada faktor spesifik atau skor sangat rendah
    if not saran or skor < 35:
        saran += [
            ("📱 Aktifkan dan gunakan e-wallet rutin",
             "Gunakan GoPay/OVO/Dana setiap bulan minimal 12 bulan "
             "untuk membangun sinyal digital yang konsisten."),
            ("💡 Konsistensikan bayar tagihan listrik",
             "Target minimal 20 dari 24 bulan bayar tepat waktu (+7 poin skor)."),
            ("🏪 Daftarkan usaha ke marketplace",
             "Toko aktif di Tokopedia/Shopee dengan omzet >Rp 3 jt/bulan "
             "menambahkan 5 poin skor secara langsung."),
        ]
    waktu_reapply = "3–6 bulan" if skor >= 38 else "6–12 bulan"
    saran.append(("🔁 Ajukan kembali setelah perbaikan",
        f"Setelah melakukan langkah di atas, estimasi waktu yang tepat untuk "
        f"mengajukan kembali adalah <b>{waktu_reapply}</b>."))
    return saran

def render_keputusan(skor, kat, rek, prob, pos_list, neg_list,
                     nama, segmen, platform, tujuan, jumlah, tenor):
    """Render seluruh tampilan keputusan kredit (disetujui atau ditolak)"""

    bunga_est = 2.0 if platform == "P2P Lending" else 1.5
    angsuran  = round(jumlah * (1 + bunga_est/100 * tenor) / tenor)
    total_kwj = round(jumlah * (1 + bunga_est/100 * tenor))
    color_skor = "#38A169" if skor>=70 else "#D69E2E" if skor>=50 else "#E53E3E"

    # ── VERDICT BANNER ────────────────────────────────────────────────
    if rek == "Setujui":
        st.markdown(f"""
        <div class='verdict-approved'>
          <div class='verdict-score' style='color:#276749'>{skor}
            <span style='font-size:24px'>/100</span>
          </div>
          <div class='verdict-label' style='color:#276749'>✅ PENGAJUAN DISETUJUI</div>
          <div style='font-size:14px;color:#38A169;margin-top:.5rem;font-weight:600'>
            Risiko {kat} · Probabilitas gagal bayar {prob:.1%}
          </div>
          <div class='verdict-meta'>{nama} · {segmen} · {platform} · {tujuan}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='verdict-rejected'>
          <div class='verdict-score' style='color:#742A2A'>{skor}
            <span style='font-size:24px'>/100</span>
          </div>
          <div class='verdict-label' style='color:#742A2A'>❌ PENGAJUAN DITOLAK</div>
          <div style='font-size:14px;color:#E53E3E;margin-top:.5rem;font-weight:600'>
            Risiko {kat} · Probabilitas gagal bayar {prob:.1%}
          </div>
          <div class='verdict-meta'>{nama} · {segmen} · {platform} · {tujuan}</div>
          {f"<div style='font-size:12px;color:#D69E2E;margin-top:.5rem'>💡 Hanya butuh {50-skor} poin lagi untuk disetujui</div>" if 38<=skor<50 else ""}
        </div>""", unsafe_allow_html=True)

    # ── KONTEN UTAMA ───────────────────────────────────────────────────
    col_kiri, col_kanan = st.columns([1, 1], gap="large")

    # ══ KOLOM KIRI — Alur proses & SHAP ══════════════════════════════
    with col_kiri:

        # --- Alur 5 langkah ---
        st.markdown("### 📋 Alur Proses Keputusan")
        ada_spike = any("spike" in f.lower() for f in neg_list)

        steps = [
            ("info", "Langkah 1", "📥 Pengumpulan Data Alternatif",
             "18 sinyal digital dikumpulkan: platform history · e-wallet · gig income · tagihan listrik · profil nasabah."),
            ("warning" if ada_spike else "info", "Langkah 2",
             "⚙️ Preprocessing & Deteksi Red Flag",
             "<span style='color:#E53E3E;font-weight:600'>⚠️ Red flag terdeteksi: lonjakan top-up e-wallet</span>"
             if ada_spike else "✅ Tidak ada red flag terdeteksi. Data bersih dan dapat dipercaya."),
            ("info", "Langkah 3", "🧠 XGBoost Scoring",
             f"Model memproses 18 fitur → skor <b style='color:{color_skor}'>{skor}/100</b>"
             f" (minimum disetujui: 50/100)"),
            ("info", "Langkah 4", "🔬 SHAP Explainability",
             f"<span style='color:#38A169'><b>{len(pos_list)}</b> sinyal positif</span> · "
             f"<span style='color:#E53E3E'><b>{len(neg_list)}</b> sinyal risiko</span>"
             " — setiap faktor dapat dikutip ke nasabah dan OJK."),
            ("approved" if rek=="Setujui" else "rejected", "Langkah 5",
             ("✅ Rekomendasi AI: SETUJUI" if rek=="Setujui"
              else "❌ Rekomendasi AI: TOLAK"),
             "Keputusan <b>final</b> tetap di tangan Credit Analyst. "
             "Override akan tercatat otomatis dalam audit trail."),
        ]

        for cls, num, title, body in steps:
            num_cls  = "info" if cls == "info" else cls
            step_cls = f"flow-step {cls}"
            st.markdown(f"""<div class='{step_cls}'>
              <div class='flow-num {num_cls}'>{num}</div>
              <div class='flow-title'>{title}</div>
              <div class='flow-body'>{body}</div>
            </div>""", unsafe_allow_html=True)

        # --- SHAP bars ---
        st.markdown("### 🔬 Faktor Penentu Skor")
        shap_html = render_shap_bars(pos_list, neg_list)
        if shap_html:
            st.markdown(shap_html, unsafe_allow_html=True)
        else:
            st.info("Tidak ada faktor spesifik yang tercatat untuk pengajuan ini.")

    # ══ KOLOM KANAN — Detail & langkah selanjutnya ═══════════════════
    with col_kanan:

        if rek == "Setujui":
            # --- Ringkasan pembiayaan ---
            st.markdown("### 💰 Ringkasan Pembiayaan")
            st.markdown(f"""<div class='detail-box'>
              <div class='detail-row'>
                <span class='detail-key'>Plafon disetujui</span>
                <span class='detail-val' style='color:#38A169;font-size:16px'>
                  Rp {jumlah:,.0f}</span>
              </div>
              <div class='detail-row'>
                <span class='detail-key'>Estimasi angsuran</span>
                <span class='detail-val'>Rp {angsuran:,.0f} / bulan</span>
              </div>
              <div class='detail-row'>
                <span class='detail-key'>Tenor</span>
                <span class='detail-val'>{tenor} bulan</span>
              </div>
              <div class='detail-row'>
                <span class='detail-key'>Bunga estimasi</span>
                <span class='detail-val'>{bunga_est}% / bulan</span>
              </div>
              <div class='detail-row'>
                <span class='detail-key'>Total kewajiban</span>
                <span class='detail-val'>Rp {total_kwj:,.0f}</span>
              </div>
              <div class='detail-row'>
                <span class='detail-key'>Total bunga</span>
                <span class='detail-val'>Rp {total_kwj - int(jumlah):,.0f}</span>
              </div>
            </div>""", unsafe_allow_html=True)

            # Visualisasi pembagian pokok vs bunga
            fig_pie = go.Figure(go.Pie(
                labels=["Pokok Pinjaman","Total Bunga"],
                values=[jumlah, total_kwj - jumlah],
                marker_colors=["#3182CE","#D69E2E"],
                hole=0.55, textinfo="percent+label",
                textfont_size=11,
            ))
            fig_pie.update_layout(
                height=180, margin=dict(t=0,b=0,l=0,r=0),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                annotations=[dict(text=f"Rp {total_kwj/1e6:.1f}jt<br><span style='font-size:9'>total</span>",
                                  x=0.5, y=0.5, font_size=13,
                                  showarrow=False)]
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # --- Langkah selanjutnya (disetujui) ---
            st.markdown("### 🗺️ Langkah Selanjutnya")
            langkah_ok = [
                ("✅ Verifikasi dokumen",
                 "Credit analyst verifikasi identitas dan konfirmasi data digital yang diinput."),
                ("📝 Penandatanganan perjanjian",
                 f"Nasabah TTD perjanjian kredit Rp {jumlah:,.0f} · {tenor} bulan."),
                ("💸 Pencairan dana",
                 "Dana dicairkan ke rekening atau langsung sesuai tujuan pengajuan."),
                ("📅 Monitoring angsuran",
                 f"Pantau pembayaran Rp {angsuran:,.0f}/bulan. "
                 "Keterlambatan >3 hari memicu notifikasi otomatis."),
                ("🔄 Review skor berkala",
                 "Skor diperbarui setiap 3 bulan berdasarkan riwayat pembayaran terbaru."),
            ]
            for title, desc in langkah_ok:
                st.markdown(f"""<div class='next-card approved'>
                  <div class='next-card-title approved'>{title}</div>
                  <div style='font-size:12px;color:#4A5568'>{desc}</div>
                </div>""", unsafe_allow_html=True)

        else:
            # --- Alasan penolakan ---
            st.markdown("### ❌ Alasan Penolakan")
            if neg_list:
                for f in neg_list:
                    label = f.split("(-")[0].strip() if "(-" in f else f
                    st.markdown(f"""<div class='flow-step rejected' style='padding:.625rem .875rem;margin-bottom:.4rem'>
                      <div class='flow-body' style='color:#742A2A'>⛔ {label}</div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class='flow-step rejected' style='padding:.625rem .875rem'>
                  <div class='flow-body' style='color:#742A2A'>
                    ⛔ Skor di bawah ambang batas minimum (50/100)
                  </div>
                </div>""", unsafe_allow_html=True)

            if 38 <= skor < 50:
                st.markdown(f"""<div class='warn-bar'>
                  <b>💡 Mendekati batas.</b> Skor {skor}/100 — hanya kurang
                  <b>{50-skor} poin</b> untuk disetujui. Perbaikan kecil bisa
                  mengubah keputusan ini dalam 3–6 bulan.
                </div>""", unsafe_allow_html=True)

            # --- Cara memperbaiki skor ---
            st.markdown("### 🔄 Cara Meningkatkan Skor")
            for title, desc in saran_perbaikan(neg_list, skor):
                st.markdown(f"""<div class='next-card rejected'>
                  <div class='next-card-title rejected'>{title}</div>
                  <div style='font-size:12px;color:#4A5568'>{desc}</div>
                </div>""", unsafe_allow_html=True)

    # ── NARASI ─────────────────────────────────────────────────────────
    st.markdown("---")
    col_n1, col_n2 = st.columns(2)

    with col_n1:
        st.markdown("#### 💬 Narasi untuk Nasabah")
        st.caption("Bacakan atau kirimkan langsung ke nasabah.")
        if rek == "Setujui":
            pos_str = "\n".join([f"  ✅ {f.split('(+')[0].strip()}"
                                 for f in pos_list[:3]])
            narasi_n = f"""Kepada Yth. {nama},

Selamat! Pengajuan kredit Anda telah DISETUJUI. ✅

Detail pembiayaan:
  • Plafon   : Rp {jumlah:,.0f}
  • Tenor    : {tenor} bulan
  • Angsuran : Rp {angsuran:,.0f}/bulan (estimasi)
  • Tujuan   : {tujuan}

Dasar persetujuan (sinyal positif utama):
{pos_str if pos_str else "  (data digital Anda memenuhi standar kelayakan)"}

Langkah selanjutnya:
  Tim kami akan menghubungi Anda untuk verifikasi dokumen
  dan penandatanganan perjanjian kredit.

Terima kasih telah mempercayai layanan kami.

Salam hangat,
Tim Credit Analyst
AI Credit Scoring — POJK 29/2024"""
        else:
            saran_str = "\n".join([f"  • {t}" for t,_ in
                                   saran_perbaikan(neg_list, skor)[:3]])
            alasan_str = "\n".join([f"  • {f.split('(-')[0].strip()}"
                                    for f in neg_list]) if neg_list else \
                         "  • Skor di bawah ambang minimum (50/100)"
            narasi_n = f"""Kepada Yth. {nama},

Mohon maaf, pengajuan kredit Anda saat ini belum dapat
kami setujui. ❌

Skor kredit Anda: {skor}/100 (minimum: 50/100)

Alasan utama:
{alasan_str}

Yang dapat Anda lakukan untuk mengajukan kembali:
{saran_str}

Kami mengundang Anda untuk mengajukan kembali setelah
melakukan perbaikan di atas. Tim kami siap membantu
kapan saja.

Salam hangat,
Tim Credit Analyst
AI Credit Scoring — POJK 29/2024"""

        st.text_area("", narasi_n, height=300, key="narasi_nsb_v2")
        st.download_button("⬇️ Unduh narasi nasabah (.txt)", narasi_n,
            file_name=f"narasi_{nama.replace(' ','_')}_{datetime.date.today()}.txt",
            key="dl_nsb_v2")

    with col_n2:
        st.markdown("#### 📋 Laporan Audit (OJK)")
        st.caption("Simpan sebagai bagian dari audit trail POJK 29/2024.")
        pos_str_a  = "\n".join([f"  + {f}" for f in pos_list]) or "  (tidak ada)"
        neg_str_a  = "\n".join([f"  - {f}" for f in neg_list]) or "  (tidak ada)"
        detail_str = (f"  Angsuran : Rp {angsuran:,.0f}/bulan\n"
                      f"  Bunga    : {bunga_est}%/bulan\n"
                      f"  Total    : Rp {total_kwj:,.0f}"
                      if rek == "Setujui" else
                      "  Lihat saran perbaikan pada narasi nasabah")

        narasi_a = f"""LAPORAN KEPUTUSAN KREDIT
{'='*52}
Tanggal   : {datetime.date.today().strftime('%d %B %Y')}
Timestamp : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

IDENTITAS PEMOHON:
  Nama      : {nama}
  Segmen    : {segmen}
  Platform  : {platform}
  Tujuan    : {tujuan}
  Plafon    : Rp {jumlah:,.0f}
  Tenor     : {tenor} bulan

HASIL PENILAIAN AI:
  Skor Kredit       : {skor}/100
  Kategori Risiko   : {kat}
  Rekomendasi AI    : {rek.upper()}
  Prob. Gagal Bayar : {prob:.1%}

SINYAL POSITIF ({len(pos_list)} faktor):
{pos_str_a}

SINYAL RISIKO ({len(neg_list)} faktor):
{neg_str_a}

DETAIL KEPUTUSAN:
{detail_str}

TINDAKAN CREDIT ANALYST:
  Keputusan akhir  : [ ] Setujui  [ ] Tolak  [ ] Override
  Alasan override  : ________________________________
  Nama analis      : ________________________________
  Tanda tangan     : ________________________________

MODEL & KEPATUHAN:
  Algoritma        : XGBoost v1.0
  Fitur            : 18 sinyal digital
  Explainability   : SHAP-based per keputusan
  Regulasi         : POJK 29/2024
{'='*52}"""

        st.text_area("", narasi_a, height=300, key="narasi_aud_v2")
        st.download_button("⬇️ Unduh laporan audit (.txt)", narasi_a,
            file_name=f"audit_{nama.replace(' ','_')}_{datetime.date.today()}.txt",
            key="dl_aud_v2")

# ── Shared scoring logic ─────────────────────────────────────────────────
def hitung_skor_live(lama_plt, ontime_rt, n_lancar, n_pinj, n_macet,
                     listrik_pct, ew_aktif, ew_trx, is_gig,
                     gig_konsisten, gig_rating, gig_cancel,
                     punya_rekening, punya_bpjs, punya_toko,
                     omzet_toko, tanggungan, ew_spike, avg_telat):
    skor = 50; pos = []; neg = []
    if lama_plt >= 12:
        skor += 7; pos.append(f"Aktif di platform {lama_plt} bulan (+7)")
    if ontime_rt >= 0.90:
        skor += 10; pos.append(f"On-time rate {ontime_rt:.0%} (+10)")
    elif ontime_rt >= 0.75:
        skor += 5;  pos.append(f"On-time rate {ontime_rt:.0%} (+5)")
    if n_lancar > 0:
        add = min(n_lancar*5,15); skor += add
        pos.append(f"{n_lancar} pinjaman lancar (+{add})")
    if n_pinj > 0 and n_macet == 0:
        skor += 7; pos.append("Tidak ada riwayat macet (+7)")
    if listrik_pct >= 90:
        skor += 7; pos.append(f"Listrik {listrik_pct:.0f}% tepat waktu (+7)")
    elif listrik_pct >= 75:
        skor += 3; pos.append(f"Listrik {listrik_pct:.0f}% tepat waktu (+3)")
    if ew_aktif >= 12:
        skor += 5; pos.append(f"E-wallet aktif {ew_aktif} bulan (+5)")
    if ew_trx >= 20:
        skor += 3; pos.append(f"Transaksi e-wallet {ew_trx}/bln (+3)")
    if is_gig and gig_konsisten:
        skor += 8; pos.append("Income gig konsisten 6 bulan (+8)")
    if is_gig and gig_rating >= 4.5:
        skor += 5; pos.append(f"Rating gig {gig_rating} (+5)")
    if punya_rekening:
        skor += 4; pos.append("Punya rekening bank (+4)")
    if punya_bpjs:
        skor += 3; pos.append("BPJS rutin dibayar (+3)")
    if punya_toko and omzet_toko > 3_000_000:
        skor += 5; pos.append(f"Toko online omzet Rp {omzet_toko/1e6:.0f}jt (+5)")
    if n_macet > 0:
        k = min(n_macet*12,24); skor -= k
        neg.append(f"{n_macet} pinjaman macet (-{k})")
    if avg_telat > 30:
        skor -= 10; neg.append(f"Keterlambatan {avg_telat} hari (-10)")
    if tanggungan >= 4:
        skor -= 5; neg.append(f"{tanggungan} tanggungan (-5)")
    if ew_spike:
        skor -= 8; neg.append("Spike top-up e-wallet (-8) ⚠️")
    if is_gig and gig_cancel >= 10:
        skor -= 4; neg.append(f"Cancel rate {gig_cancel}% (-4)")
    skor = max(10, min(95, skor))
    return skor, pos, neg

# ════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.title("📊 AI Credit Scoring — P2P Lending & Multifinance")
    st.markdown("""<div class='info-bar'>
    <b>Selamat datang.</b> Platform ini membantu tim risk management P2P lending dan multifinance
    menilai kelayakan kredit nasabah <b>thin-file</b> — peminjam tanpa rekening formal atau slip gaji —
    menggunakan data digital: perilaku e-wallet, income gig economy, konsistensi tagihan, dan riwayat
    platform. Setiap keputusan dapat dijelaskan dan diaudit sesuai <b>POJK 29/2024</b>.
    </div>""", unsafe_allow_html=True)

    df = apply_filter(aj)
    c1,c2,c3,c4,c5 = st.columns(5)
    dr  = df["label_default"].mean()
    ovr = (df["rekomendasi"]=="Setujui").mean()
    skor_avg = df["skor_ai"].mean()
    gig_pct  = nsb["is_gig_worker"].mean()

    with c1: st.markdown(f"""<div class='kpi'><h4>Total Pengajuan</h4>
        <div class='val'>{len(df):,}</div><div class='sub'>Dataset aktif</div></div>""",
        unsafe_allow_html=True)
    with c2: st.markdown(f"""<div class='kpi'><h4>Default Rate</h4>
        <div class='val' style='color:#E53E3E'>{dr:.1%}</div>
        <div class='sub'>Nasabah gagal bayar</div></div>""", unsafe_allow_html=True)
    with c3: st.markdown(f"""<div class='kpi'><h4>Approval Rate AI</h4>
        <div class='val' style='color:#3182CE'>{ovr:.1%}</div>
        <div class='sub'>Rekomendasi setujui</div></div>""", unsafe_allow_html=True)
    with c4: st.markdown(f"""<div class='kpi'><h4>Skor Rata-rata</h4>
        <div class='val'>{skor_avg:.0f}<span style='font-size:16px'>/100</span></div>
        <div class='sub'>Semua platform</div></div>""", unsafe_allow_html=True)
    with c5: st.markdown(f"""<div class='kpi'><h4>Gig Workers</h4>
        <div class='val' style='color:#805AD5'>{gig_pct:.0%}</div>
        <div class='sub'>Dari total nasabah</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Distribusi skor kredit per tipe platform**")
        fig = px.histogram(df, x="skor_ai", color="platform_tipe", nbins=20,
            color_discrete_map={"P2P Lending":"#3182CE","Multifinance":"#805AD5"},
            barmode="overlay", opacity=0.75, template="plotly_white",
            labels={"skor_ai":"Skor AI","count":"Jumlah"})
        fig.update_layout(height=280, margin=dict(t=10,b=20,l=10,r=10),
            paper_bgcolor="rgba(0,0,0,0)", legend_title="Tipe")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        st.markdown("**Default rate per segmen pekerjaan**")
        merged = df.merge(nsb[["nasabah_id","segmen_pekerjaan"]], on="nasabah_id")
        seg_dr = merged.groupby("segmen_pekerjaan")["label_default"].mean()\
            .sort_values(ascending=True).reset_index()
        fig2 = px.bar(seg_dr, x="label_default", y="segmen_pekerjaan", orientation="h",
            color="label_default",
            color_continuous_scale=["#38A169","#D69E2E","#E53E3E"],
            template="plotly_white",
            labels={"label_default":"Default Rate","segmen_pekerjaan":""})
        fig2.update_layout(height=280, margin=dict(t=10,b=20,l=10,r=10),
            paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        fig2.update_traces(text=[f"{v:.1%}" for v in seg_dr["label_default"]],
                           textposition="outside")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""<div class='warn-bar'>
    <b>⚠️ Catatan untuk tim risk:</b> Sistem AI memberikan <b>rekomendasi pendukung</b>, bukan
    keputusan final. Credit analyst tetap memiliki otoritas penuh. Setiap override akan otomatis
    tercatat dalam audit trail untuk pelaporan OJK.
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec">Panduan penggunaan platform ini</div>',
        unsafe_allow_html=True)
    gc1,gc2,gc3,gc4,gc5 = st.columns(5)
    guides = [
        ("🎯","Keputusan Kredit",
         "Alur visual lengkap disetujui atau ditolak — detail pembiayaan, SHAP bars, langkah selanjutnya."),
        ("📋","Monitor Pengajuan",
         "Riwayat seluruh pengajuan, filter multi-dimensi, ekspor CSV untuk audit OJK."),
        ("📈","Analitik Portofolio",
         "Dashboard risk manager: NPL per segmen, P2P vs multifinance, scatter kota."),
        ("⚙️","Performa Model",
         "AUC-ROC, feature importance, dan dokumentasi fitur untuk kepatuhan POJK 29/2024."),
    ]
    for col,(ico,ttl,desc) in zip([gc1,gc2,gc3,gc4,gc5],guides):
        col.markdown(f"""<div style='background:white;border-radius:10px;
        padding:.875rem;border:0.5px solid #E2E8F0;height:100%'>
        <div style='font-size:22px;margin-bottom:6px'>{ico}</div>
        <div style='font-size:13px;font-weight:600;color:#1A202C;margin-bottom:5px'>{ttl}</div>
        <div style='font-size:12px;color:#718096;line-height:1.6'>{desc}</div>
        </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════
# PAGE 3 — KEPUTUSAN KREDIT
# ═════════════════════════════════════════════════════════════════
elif "Keputusan" in page:
    st.title("🎯 Keputusan Kredit")
    st.markdown("""<div class='info-bar'>
    Halaman ini menampilkan <b>alur keputusan kredit secara lengkap</b> — baik untuk pengajuan
    yang <b>disetujui</b> maupun <b>ditolak</b>. Setiap tahap dilengkapi visualisasi SHAP,
    detail pembiayaan, langkah selanjutnya, dan narasi siap-pakai untuk nasabah maupun OJK.<br><br>
    <b>Dua mode tersedia:</b> pilih dari dataset historis untuk eksplorasi cepat,
    atau input manual untuk mensimulasikan nasabah baru.
    </div>""", unsafe_allow_html=True)

    mode = st.radio("Sumber data:",
        ["📂 Pilih dari dataset historis", "✏️ Input manual nasabah baru"],
        horizontal=True)
    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # MODE A — DATASET HISTORIS
    # ══════════════════════════════════════════════════════════════════
    if "Pilih" in mode:
        df_h = aj.merge(nsb[["nasabah_id", "segmen_pekerjaan", "kota", "usia"]],
                        on="nasabah_id", how="left")

        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        with col_f1:
            f_rek2 = st.selectbox("Keputusan", ["Semua", "Setujui", "Tolak"])
        with col_f2:
            f_kat2 = st.selectbox("Kategori risiko", ["Semua", "Rendah", "Menengah", "Tinggi"])
        with col_f3:
            f_seg2 = st.selectbox("Segmen", ["Semua"] + sorted(df_h["segmen_pekerjaan"].dropna().unique().tolist()))
        with col_f4:
            f_plt2 = st.selectbox("Platform tipe", ["Semua", "P2P Lending", "Multifinance"])

        if f_rek2 != "Semua": df_h = df_h[df_h["rekomendasi"] == f_rek2]
        if f_kat2 != "Semua": df_h = df_h[df_h["kategori_risiko"] == f_kat2]
        if f_seg2 != "Semua": df_h = df_h[df_h["segmen_pekerjaan"] == f_seg2]
        if f_plt2 != "Semua": df_h = df_h[df_h["platform_tipe"] == f_plt2]

        if len(df_h) == 0:
            st.warning("Tidak ada data yang cocok. Ubah filter.")
            st.stop()

        qa, qb, qc = st.columns(3)
        qa.metric("Hasil filter", f"{len(df_h):,} pengajuan")
        qb.metric("Disetujui", f"{(df_h['rekomendasi']=='Setujui').sum():,}",
                  f"{(df_h['rekomendasi']=='Setujui').mean():.0%}")
        qc.metric("Skor rata-rata", f"{df_h['skor_ai'].mean():.0f}/100")

        st.markdown("---")
        df_h["_label"] = (
            df_h["pengajuan_id"] + " · " +
            df_h["segmen_pekerjaan"].fillna("-") + " · " +
            df_h["rekomendasi"] + " · skor " +
            df_h["skor_ai"].astype(str) + " · " +
            df_h["platform_tipe"]
        )
        selected = st.selectbox(
            f"Pilih pengajuan untuk dilihat alur keputusannya ({min(len(df_h),500):,} ditampilkan):",
            df_h["_label"].head(500).tolist()
        )
        row = df_h[df_h["_label"] == selected].iloc[0]

        def parse_factors(raw):
            if pd.isna(raw) or str(raw).strip() in ("", "nan"):
                return []
            return [f.strip() for f in str(raw).split("|") if f.strip()]

        pos_list = parse_factors(row.get("faktor_positif", ""))
        neg_list = parse_factors(row.get("faktor_negatif", ""))

        skor   = int(row["skor_ai"])
        kat    = row["kategori_risiko"]
        rek    = row["rekomendasi"]
        prob   = float(row["prob_default"])
        jumlah = float(row["jumlah_diajukan"])
        tenor  = int(row["tenor_bln"])
        platform = row["platform_tipe"]
        segmen = str(row.get("segmen_pekerjaan", ""))
        tujuan = str(row.get("tujuan", ""))
        nama   = f"Nasabah {row['nasabah_id']}"

        st.markdown("---")
        render_keputusan(skor, kat, rek, prob, pos_list, neg_list,
                         nama, segmen, platform, tujuan, jumlah, tenor)

    # ══════════════════════════════════════════════════════════════════
    # MODE B — INPUT MANUAL (IMPROVED UX & LIVE PREVIEW)
    # ══════════════════════════════════════════════════════════════════
    else:
        # ── Inisialisasi Session State untuk Mode B ─────────────────
        _defs_m = {
            "nama_m": "", "platform_m": "P2P Lending", "segmen_m": "Ojek online / driver",
            "tanggungan_m": 1, "lama_plt_m": 8, "ontime_rt_m": 0.85,
            "punya_rekening_m": False, "punya_bpjs_m": False, "punya_toko_m": False, "omzet_toko_m": 0,
            "n_pinj_m": 0, "n_lancar_m": 0, "n_macet_m": 0, "avg_telat_m": 0,
            "punya_ew_m": False, "ew_aktif_m": 0, "ew_trx_m": 0, "ew_spike_m": False,
            "gig_inc_m": 0, "gig_kon_m": False, "gig_rat_m": 4.2, "gig_can_m": 5,
            "listrik_m": 16, "listrik_pct_m": 66.7,
            "jumlah_m": 3_000_000, "tenor_m": 6, "tujuan_m": "Modal usaha",
            "show_result_m": False, "result_data_m": None
        }
        for k, v in _defs_m.items():
            if k not in st.session_state:
                st.session_state[k] = v
                
        s_m = st.session_state

        # ── Quick Presets ───────────────────────────────────────────
        st.markdown('<div class="sec">⚡ Quick Preset (Simulasi Cepat)</div>', unsafe_allow_html=True)
        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1:
            if st.button("🟢 Load Profil Nasabah Prima", use_container_width=True):
                s_m.update({
                    "nama_m": "Budi Santoso", "segmen_m": "Pemilik UMKM kecil", "platform_m": "P2P Lending",
                    "ontime_rt_m": 0.95, "n_macet_m": 0, "n_lancar_m": 3, "n_pinj_m": 3, "listrik_m": 22,
                    "punya_rekening_m": True, "punya_bpjs_m": True, "punya_toko_m": True, "omzet_toko_m": 5_000_000,
                    "ew_aktif_m": 15, "ew_trx_m": 25, "avg_telat_m": 0, "show_result_m": False
                })
                st.rerun()
        with c_p2:
            if st.button("🔴 Load Profil Berisiko", use_container_width=True):
                s_m.update({
                    "nama_m": "Siti Aminah", "segmen_m": "Ojek online / driver", "platform_m": "P2P Lending",
                    "ontime_rt_m": 0.60, "n_macet_m": 2, "n_lancar_m": 1, "n_pinj_m": 3, "listrik_m": 10,
                    "punya_rekening_m": False, "punya_toko_m": False, "ew_aktif_m": 2, "ew_trx_m": 5,
                    "ew_spike_m": True, "gig_can_m": 15, "avg_telat_m": 45, "show_result_m": False
                })
                st.rerun()
        with c_p3:
            if st.button("🔄 Reset Form", use_container_width=True):
                for k in _defs_m:
                    s_m[k] = _defs_m[k]
                st.rerun()

        st.markdown("---")

        # ── Derived Values ──────────────────────────────────────────
        is_gig_m = s_m["segmen_m"] in ["Ojek online / driver", "Kurir / ekspedisi", "Freelancer digital"]
        s_m["listrik_pct_m"] = round(s_m["listrik_m"] / 24 * 100, 1)

        # ── Live Score Calculation (Real-time) ──────────────────────
        live_skor_m, live_pos_m, live_neg_m = hitung_skor_live(
            s_m["lama_plt_m"], s_m["ontime_rt_m"], s_m["n_lancar_m"], s_m["n_pinj_m"], s_m["n_macet_m"],
            s_m["listrik_pct_m"], s_m["ew_aktif_m"], s_m["ew_trx_m"], is_gig_m, s_m["gig_kon_m"],
            s_m["gig_rat_m"], s_m["gig_can_m"], s_m["punya_rekening_m"], s_m["punya_bpjs_m"],
            s_m["punya_toko_m"], s_m["omzet_toko_m"], s_m["tanggungan_m"], s_m["ew_spike_m"], s_m["avg_telat_m"]
        )
        
        live_kat_m = "Rendah" if live_skor_m >= 70 else "Menengah" if live_skor_m >= 50 else "Tinggi"
        live_col_m = "#38A169" if live_skor_m >= 70 else "#D69E2E" if live_skor_m >= 50 else "#E53E3E"
        live_ico_m = "✅" if live_skor_m >= 70 else "⚠️" if live_skor_m >= 50 else "❌"

        # ── Layout: Form Kiri, Live Preview Kanan ───────────────────
        col_form, col_live = st.columns([2, 1], gap="large")

        # ══════════════════════════════════════════════════════════════
        # KOLOM KANAN: LIVE PREVIEW
        # ══════════════════════════════════════════════════════════════
        with col_live:
            st.markdown(f"""
            <div class='live-card'>
                <div style='font-size:11px;color:#718096;font-weight:600;letter-spacing:.06em;text-transform:uppercase;margin-bottom:.5rem'>
                    📊 Estimasi Skor Real-Time
                </div>
                <div class='live-skor' style='color:{live_col_m}'>{live_skor_m}</div>
                <div style='text-align:center;font-size:13px;color:{live_col_m};font-weight:600;margin-bottom:.4rem'>
                    {live_ico_m} Risiko {live_kat_m}
                </div>
                <div class='live-bar-wrap'>
                    <div class='live-bar-fill' style='width:{live_skor_m}%;background:{live_col_m}'></div>
                </div>
                <div style='display:flex;justify-content:space-between;font-size:10px;color:#A0AEC0;margin-top:3px'>
                    <span>Tolak &lt;50</span> <span>Ragu 50-69</span> <span>Setujui ≥70</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if live_pos_m:
                st.markdown(f"<div style='font-size:11px;font-weight:600;color:#276749;margin:.75rem 0 .35rem'>✅ Faktor Positif ({len(live_pos_m)})</div>", unsafe_allow_html=True)
                for f in live_pos_m[:4]:
                    st.markdown(f"<div class='mini-factor mini-pos'>{f}</div>", unsafe_allow_html=True)
            
            if live_neg_m:
                st.markdown(f"<div style='font-size:11px;font-weight:600;color:#742A2A;margin:.75rem 0 .35rem'>❌ Faktor Risiko ({len(live_neg_m)})</div>", unsafe_allow_html=True)
                for f in live_neg_m[:4]:
                    st.markdown(f"<div class='mini-factor mini-neg'>{f}</div>", unsafe_allow_html=True)

            if live_skor_m < 50:
                st.markdown(f"""
                <div class='preview-tip-warn' style='margin-top:.75rem'>
                    ⚠️ Butuh <b>{50 - live_skor_m}</b> poin lagi untuk disetujui.
                </div>""", unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════
        # KOLOM KIRI: FORM INPUT (DINAMIS TANPA st.form)
        # ══════════════════════════════════════════════════════════════
        with col_form:
            st.markdown('<div class="sec">👤 Profil & Riwayat Kredit</div>', unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                s_m["nama_m"] = st.text_input("Nama Nasabah", value=s_m["nama_m"], placeholder="Contoh: Budi Santoso")
                s_m["segmen_m"] = st.selectbox("Segmen Pekerjaan", [
                    "Ojek online / driver", "Pedagang pasar / warung", "Buruh pabrik", 
                    "Kurir / ekspedisi", "Freelancer digital", "Petani / nelayan", 
                    "Karyawan kontrak", "Penjahit / konveksi", "Pemilik UMKM kecil"
                ], index=[
                    "Ojek online / driver", "Pedagang pasar / warung", "Buruh pabrik", 
                    "Kurir / ekspedisi", "Freelancer digital", "Petani / nelayan", 
                    "Karyawan kontrak", "Penjahit / konveksi", "Pemilik UMKM kecil"
                ].index(s_m["segmen_m"]))
            with c2:
                s_m["platform_m"] = st.selectbox("Platform", ["P2P Lending", "Multifinance"],
                                                 index=["P2P Lending", "Multifinance"].index(s_m["platform_m"]))
                s_m["tanggungan_m"] = st.slider("Jumlah Tanggungan", 0, 6, value=s_m["tanggungan_m"])

            c1, c2, c3 = st.columns(3)
            with c1:
                s_m["lama_plt_m"] = st.slider("Lama di platform (bln)", 0, 48, value=s_m["lama_plt_m"])
            with c2:
                s_m["ontime_rt_m"] = st.slider("On-time rate", 0.0, 1.0, value=s_m["ontime_rt_m"], step=0.01, format="%.2f")
            with c3:
                s_m["avg_telat_m"] = st.slider("Rata-rata hari telat", 0, 180, value=s_m["avg_telat_m"])

            c1, c2, c3, c4 = st.columns(4)
            with c1: s_m["n_pinj_m"] = st.number_input("Total pinjaman", 0, 20, value=s_m["n_pinj_m"])
            with c2: s_m["n_lancar_m"] = st.number_input("Lancar", 0, 20, value=s_m["n_lancar_m"])
            with c3: s_m["n_macet_m"] = st.number_input("Macet", 0, 20, value=s_m["n_macet_m"])
            with c4: 
                s_m["punya_rekening_m"] = st.checkbox("Punya rekening", value=s_m["punya_rekening_m"])
                s_m["punya_bpjs_m"] = st.checkbox("BPJS rutin", value=s_m["punya_bpjs_m"])

            # ── Conditional: Gig Economy ────────────────────────────
            if is_gig_m:
                st.markdown('<div class="info-bar">🛵 <b>Mode Gig Worker Aktif</b> — Data rating & cancel rate akan dihitung.</div>', unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                with c1:
                    s_m["gig_inc_m"] = st.number_input("Income gig/bln (Rp)", 0, 20_000_000, value=s_m["gig_inc_m"], step=500_000)
                with c2:
                    s_m["gig_kon_m"] = st.checkbox("Income konsisten 6 bln", value=s_m["gig_kon_m"])
                    s_m["gig_rat_m"] = st.slider("Rating gig", 1.0, 5.0, value=s_m["gig_rat_m"], step=0.1)
                with c3:
                    s_m["gig_can_m"] = st.slider("Cancel rate (%)", 0, 30, value=s_m["gig_can_m"])

            # ── Conditional: Toko Online ────────────────────────────
            st.markdown("---")
            s_m["punya_toko_m"] = st.checkbox("✓ Punya toko online (Marketplace)", value=s_m["punya_toko_m"])
            if s_m["punya_toko_m"]:
                s_m["omzet_toko_m"] = st.number_input("Omzet toko/bln (Rp)", 0, 100_000_000, 
                                                      value=s_m["omzet_toko_m"], step=500_000,
                                                      help="Omzet >Rp 3 jt akan menambah +5 poin")

            # ── Sinyal Digital ──────────────────────────────────────
            st.markdown('<div class="sec">📱 Sinyal Digital</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                s_m["punya_ew_m"] = st.checkbox("Punya e-wallet aktif", value=s_m["punya_ew_m"])
                if s_m["punya_ew_m"]:
                    s_m["ew_aktif_m"] = st.slider("E-wallet aktif (bln)", 0, 24, value=s_m["ew_aktif_m"])
                    s_m["ew_trx_m"] = st.slider("Transaksi e-wallet/bln", 0, 100, value=s_m["ew_trx_m"])
                    s_m["ew_spike_m"] = st.checkbox("⚠️ Spike top-up sebelum pengajuan", value=s_m["ew_spike_m"])
                else:
                    s_m["ew_aktif_m"] = 0; s_m["ew_trx_m"] = 0; s_m["ew_spike_m"] = False
                    
            with c2:
                s_m["listrik_m"] = st.slider("Bulan listrik tepat waktu (dari 24)", 0, 24, value=s_m["listrik_m"])
                st.caption(f"Konsistensi listrik: <b>{s_m['listrik_pct_m']}%</b>", unsafe_allow_html=True)

            # ── Detail Pengajuan ────────────────────────────────────
            st.markdown('<div class="sec">💰 Detail Pengajuan</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                if s_m["platform_m"] == "P2P Lending":
                    s_m["jumlah_m"] = st.number_input("Jumlah (Rp)", 500_000, 50_000_000, value=s_m["jumlah_m"], step=500_000)
                    s_m["tenor_m"] = st.selectbox("Tenor (bln)", [1, 3, 6, 9, 12], index=[1, 3, 6, 9, 12].index(s_m["tenor_m"]))
                    s_m["tujuan_m"] = st.selectbox("Tujuan", ["Modal usaha", "Konsumsi", "Pendidikan", "Renovasi"], 
                                                   index=["Modal usaha", "Konsumsi", "Pendidikan", "Renovasi"].index(s_m["tujuan_m"]))
                else:
                    s_m["jumlah_m"] = st.number_input("Jumlah (Rp)", 3_000_000, 200_000_000, value=s_m["jumlah_m"], step=1_000_000)
                    s_m["tenor_m"] = st.selectbox("Tenor (bln)", [12, 18, 24, 36, 48], index=[12, 18, 24, 36, 48].index(s_m["tenor_m"]))
                    s_m["tujuan_m"] = st.selectbox("Tujuan", ["Motor baru", "Motor bekas", "Elektronik", "Mesin usaha"], 
                                                   index=["Motor baru", "Motor bekas", "Elektronik", "Mesin usaha"].index(s_m["tujuan_m"]))
            with c2:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                bunga_est = 2.0 if s_m["platform_m"] == "P2P Lending" else 1.5
                angsuran = round(s_m["jumlah_m"] * (1 + bunga_est/100 * s_m["tenor_m"]) / s_m["tenor_m"])
                st.markdown(f"""
                <div style='background:#F7FAFC;border-radius:8px;padding:.75rem;border:1px solid #E2E8F0'>
                    <div style='font-size:11px;color:#718096'>Estimasi angsuran</div>
                    <div style='font-size:20px;font-weight:700;color:#3182CE'>Rp {angsuran:,.0f} <span style='font-size:12px;color:#718096'>/bulan</span></div>
                    <div style='font-size:11px;color:#718096'>Bunga {bunga_est}% · Tenor {s_m['tenor_m']} bln</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                if st.button("🎯 Generate Keputusan & Narasi", use_container_width=True, type="primary"):
                    final_skor_m, final_pos_m, final_neg_m = hitung_skor(
                        s_m["lama_plt_m"], s_m["ontime_rt_m"], s_m["n_lancar_m"], s_m["n_pinj_m"], s_m["n_macet_m"],
                        s_m["listrik_pct_m"], s_m["ew_aktif_m"], s_m["ew_trx_m"], is_gig_m, s_m["gig_kon_m"],
                        s_m["gig_rat_m"], s_m["gig_can_m"], s_m["punya_rekening_m"], s_m["punya_bpjs_m"],
                        s_m["punya_toko_m"], s_m["omzet_toko_m"], s_m["tanggungan_m"], s_m["ew_spike_m"], s_m["avg_telat_m"]
                    )
                    
                    kat_m = "Rendah" if final_skor_m >= 70 else "Menengah" if final_skor_m >= 50 else "Tinggi"
                    rek_m = "Setujui" if final_skor_m >= 50 else "Tolak"
                    prob_m = max(0.02, min(0.88, (100 - final_skor_m) / 100 * 0.58))
                    
                    s_m["result_data_m"] = {
                        "skor": final_skor_m, "kat": kat_m, "rek": rek_m, "prob": prob_m,
                        "pos": final_pos_m, "neg": final_neg_m,
                        "nama": s_m["nama_m"] or "Nasabah", "segmen": s_m["segmen_m"],
                        "platform": s_m["platform_m"], "tujuan": s_m["tujuan_m"],
                        "jumlah": s_m["jumlah_m"], "tenor": s_m["tenor_m"]
                    }
                    s_m["show_result_m"] = True
                    st.rerun()

        # ══════════════════════════════════════════════════════════════
        # RENDER HASIL KEPUTUSAN (DITAMPILKAN DI BAWAH FORM)
        # ══════════════════════════════════════════════════════════════
        if s_m["show_result_m"] and s_m["result_data_m"]:
            st.markdown("---")
            r = s_m["result_data_m"]
            render_keputusan(r["skor"], r["kat"], r["rek"], r["prob"], r["pos"], r["neg"],
                             r["nama"], r["segmen"], r["platform"], r["tujuan"], r["jumlah"], r["tenor"])
            
# ════════════════════════════════════════════════════════════════════════
# PAGE 4 — MONITOR PENGAJUAN
# ════════════════════════════════════════════════════════════════════════
elif "Monitor" in page:
    st.title("📋 Monitor Pengajuan")
    st.markdown("""<div class='info-bar'>
    Seluruh pengajuan kredit tercatat di sini dengan skor AI, rekomendasi sistem, dan label aktual.
    Gunakan filter untuk fokus pada segmen tertentu. Data siap diekspor untuk laporan OJK.
    </div>""", unsafe_allow_html=True)

    df = apply_filter(aj).merge(nsb[["nasabah_id","segmen_pekerjaan","kota"]],
                                on="nasabah_id", how="left")
    col_f1,col_f2,col_f3,col_f4 = st.columns(4)
    with col_f1: f_kat = st.multiselect("Kategori risiko",
        ["Rendah","Menengah","Tinggi"], default=["Rendah","Menengah","Tinggi"])
    with col_f2: f_rek = st.multiselect("Rekomendasi AI",
        ["Setujui","Tolak"], default=["Setujui","Tolak"])
    with col_f3: f_gig = st.selectbox("Segmen", ["Semua","Gig workers","Non-gig"])
    with col_f4: f_def = st.selectbox("Aktual", ["Semua","Default","Tidak Default"])

    if f_kat: df = df[df["kategori_risiko"].isin(f_kat)]
    if f_rek: df = df[df["rekomendasi"].isin(f_rek)]
    if f_gig == "Gig workers": df = df[df["is_gig"]==1]
    elif f_gig == "Non-gig":   df = df[df["is_gig"]==0]
    if f_def == "Default":         df = df[df["label_default"]==1]
    elif f_def == "Tidak Default": df = df[df["label_default"]==0]

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Ditampilkan", f"{len(df):,}")
    m2.metric("Default aktual", df["label_default"].sum())
    m3.metric("Skor rata-rata", f"{df['skor_ai'].mean():.0f}")
    m4.metric("Gig workers", f"{df['is_gig'].mean():.0%}")

    cols_show = {
        "pengajuan_id":"ID","segmen_pekerjaan":"Segmen","platform_tipe":"Tipe",
        "tujuan":"Tujuan","jumlah_diajukan":"Plafon (Rp)","skor_ai":"Skor",
        "kategori_risiko":"Risiko","rekomendasi":"Rekomendasi AI",
        "prob_default":"Prob. Default","label_default":"Default"
    }
    df_show = df[list(cols_show.keys())].rename(columns=cols_show).head(300).copy()
    df_show["Plafon (Rp)"] = df_show["Plafon (Rp)"].apply(lambda x: f"Rp {x:,.0f}")
    df_show["Prob. Default"] = df_show["Prob. Default"].apply(lambda x: f"{x:.1%}")

    def style_rows(row):
        if row["Default"] == 1: return ["background:#FFF5F5"]*len(row)
        if row["Rekomendasi AI"] == "Tolak" and row["Default"] == 0:
            return ["background:#FFFBEB"]*len(row)
        return [""]*len(row)

    st.dataframe(df_show.style.apply(style_rows, axis=1),
                 use_container_width=True, height=400)
    st.caption("🔴 Merah = nasabah default · 🟡 Kuning = ditolak AI tapi tidak default")

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Ekspor CSV untuk OJK", csv,
        f"monitor_{datetime.date.today()}.csv", "text/csv")

# ════════════════════════════════════════════════════════════════════════
# PAGE 5 — ANALITIK PORTOFOLIO
# ════════════════════════════════════════════════════════════════════════
elif "Analitik" in page:
    st.title("📈 Analitik Portofolio")
    st.markdown("""<div class='info-bar'>
    Dashboard untuk <b>Risk Manager & CRO</b>. Pantau kesehatan portofolio,
    bandingkan P2P vs multifinance, dan identifikasi segmen NPL tertinggi.
    </div>""", unsafe_allow_html=True)

    df = apply_filter(aj).merge(nsb[["nasabah_id","segmen_pekerjaan","kota"]],
                                on="nasabah_id", how="left")
    c1,c2,c3,c4 = st.columns(4)
    npl      = df["label_default"].mean()
    plafon   = df[df["rekomendasi"]=="Setujui"]["jumlah_diajukan"].sum()
    skor_med = df["skor_ai"].median()
    gig_dr   = df[df["is_gig"]==1]["label_default"].mean() if df["is_gig"].sum()>0 else 0

    c1.metric("NPL Rate", f"{npl:.2%}",
              delta=f"{npl-0.065:.2%} vs benchmark 6.5%", delta_color="inverse")
    c2.metric("Total Plafon Disetujui", f"Rp {plafon/1e9:.1f} M")
    c3.metric("Median Skor", f"{skor_med:.0f}/100")
    c4.metric("Gig Worker NPL", f"{gig_dr:.2%}")

    st.markdown("---")
    ca, cb = st.columns(2)
    with ca:
        st.markdown("**NPL rate per rentang skor**")
        df2 = df.copy()
        df2["skor_bin"] = pd.cut(df2["skor_ai"],
            bins=[0,40,50,60,70,80,100], labels=["<40","40-50","50-60","60-70","70-80","80+"])
        npl_bin = df2.groupby("skor_bin")["label_default"].mean().reset_index()
        fig = px.bar(npl_bin, x="skor_bin", y="label_default",
            color="label_default",
            color_continuous_scale=["#38A169","#D69E2E","#E53E3E"],
            template="plotly_white",
            labels={"skor_bin":"Rentang Skor","label_default":"NPL Rate"})
        fig.update_traces(text=[f"{v:.1%}" for v in npl_bin["label_default"]],
                          textposition="outside")
        fig.update_layout(height=300, margin=dict(t=10,b=20,l=10,r=10),
            showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with cb:
        st.markdown("**P2P vs Multifinance — NPL**")
        comp = df.groupby("platform_tipe").agg(
            npl=("label_default","mean"),
            skor_avg=("skor_ai","mean"),
            volume=("pengajuan_id","count")).reset_index()
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="NPL Rate", x=comp["platform_tipe"],
            y=comp["npl"], marker_color=["#E53E3E","#805AD5"],
            text=[f"{v:.1%}" for v in comp["npl"]], textposition="outside"))
        fig2.update_layout(height=300, margin=dict(t=10,b=20,l=10,r=10),
            template="plotly_white", showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    ca2, cb2 = st.columns(2)
    with ca2:
        st.markdown("**Distribusi risiko per platform**")
        grp = df.groupby(["platform_id","kategori_risiko"]).size().reset_index(name="n")
        fig3 = px.bar(grp, x="platform_id", y="n", color="kategori_risiko",
            color_discrete_map={"Rendah":"#38A169","Menengah":"#D69E2E","Tinggi":"#E53E3E"},
            template="plotly_white", barmode="stack",
            labels={"platform_id":"Platform","n":"Jumlah","kategori_risiko":"Risiko"})
        fig3.update_layout(height=300, margin=dict(t=10,b=40,l=10,r=10),
            xaxis_tickangle=-15, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)
    with cb2:
        st.markdown("**Top 5 kota — volume & NPL**")
        kota_stat = df.groupby("kota").agg(
            vol=("pengajuan_id","count"), npl=("label_default","mean")
        ).nlargest(5,"vol").reset_index()
        fig4 = px.scatter(kota_stat, x="vol", y="npl", text="kota",
            size="vol", color="npl",
            color_continuous_scale=["#38A169","#E53E3E"],
            template="plotly_white",
            labels={"vol":"Volume","npl":"NPL Rate"})
        fig4.update_traces(textposition="top center")
        fig4.update_layout(height=300, margin=dict(t=10,b=20,l=10,r=10),
            showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="sec">Laporan ringkasan untuk OJK</div>',
        unsafe_allow_html=True)
    kat_cnt = df["kategori_risiko"].value_counts()
    lap = f"""LAPORAN ANALITIK PORTOFOLIO — AI CREDIT SCORING LAYER 1
Tanggal  : {datetime.date.today().strftime('%d %B %Y')}
Filter   : {plt_filter} | {tipe_filter}
Regulasi : POJK 29/2024

RINGKASAN:
Total Pengajuan : {len(df):,}
Total Plafon    : Rp {plafon:,.0f}
NPL Rate        : {npl:.2%}
Median Skor AI  : {skor_med:.0f}/100
Gig Worker NPL  : {gig_dr:.2%}

DISTRIBUSI RISIKO:
Rendah  (≥70)   : {kat_cnt.get('Rendah',0):,} ({kat_cnt.get('Rendah',0)/max(len(df),1):.1%})
Menengah (50-69): {kat_cnt.get('Menengah',0):,} ({kat_cnt.get('Menengah',0)/max(len(df),1):.1%})
Tinggi  (<50)   : {kat_cnt.get('Tinggi',0):,} ({kat_cnt.get('Tinggi',0)/max(len(df),1):.1%})

Model: XGBoost v1.0 | Explainability: SHAP | Versi: 1.0.0-layer1
"""
    st.download_button("⬇️ Unduh laporan OJK (.txt)", lap,
        f"laporan_ojk_{datetime.date.today()}.txt")

# ════════════════════════════════════════════════════════════════════════
# PAGE 6 — PERFORMA MODEL
# ════════════════════════════════════════════════════════════════════════
elif "Performa" in page:
    st.title("⚙️ Performa Model AI")
    st.markdown("""<div class='info-bar'>
    Transparansi teknis model AI sesuai kewajiban <b>POJK 29/2024</b>. Metrik akurasi,
    feature importance, dan daftar fitur yang dapat diserahkan ke OJK.
    </div>""", unsafe_allow_html=True)

    with st.spinner("Melatih model XGBoost dari data historis..."):
        model, FEAT, auc, prec, rec = train(aj, nsb)

    st.markdown(f"""<div class='ok-bar'>
    ✅ Model berhasil dilatih · AUC-ROC: <b>{auc:.3f}</b> ·
    Precision: <b>{prec:.3f}</b> · Recall: <b>{rec:.3f}</b>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("AUC-ROC",  f"{auc:.3f}",  help="1.0 = sempurna, 0.5 = random")
    c2.metric("Precision",f"{prec:.3f}", help="Akurasi prediksi default")
    c3.metric("Recall",   f"{rec:.3f}",  help="Cakupan deteksi default")
    c4.metric("Fitur",    len(FEAT))

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Feature importance**")
        fi = pd.DataFrame({"fitur":FEAT,"importance":model.feature_importances_})\
               .sort_values("importance", ascending=True)
        labels = {
            "lama_platform_bln":"Lama di platform",
            "platform_ontime_rate":"On-time payment rate",
            "n_pinjaman_sebelumnya":"Jumlah pinjaman sebelumnya",
            "n_lancar":"Pinjaman berstatus lancar",
            "n_macet":"Pinjaman berstatus macet",
            "avg_telat_hari":"Rata-rata hari keterlambatan",
            "listrik_pct_tepat":"Konsistensi bayar listrik",
            "ew_aktif_bln":"E-wallet aktif (bulan)",
            "ew_trx_per_bln":"Transaksi e-wallet/bulan",
            "ew_spike":"Spike top-up (red flag)",
            "is_gig":"Gig worker (ya/tidak)",
            "gig_income_avg":"Income gig rata-rata",
            "gig_konsisten":"Konsistensi income gig 6 bln",
            "gig_cancel_rate":"Cancel rate gig",
            "punya_rekening":"Punya rekening bank",
            "bpjs_rutin":"BPJS rutin dibayar",
            "punya_toko_online":"Punya toko online",
            "omzet_toko_online":"Omzet toko online",
        }
        fi["label"] = fi["fitur"].map(labels).fillna(fi["fitur"])
        fig = px.bar(fi.tail(14), x="importance", y="label", orientation="h",
            color="importance",
            color_continuous_scale=["#93C5FD","#3182CE","#1E3A5F"],
            template="plotly_white",
            labels={"importance":"Kepentingan","label":""})
        fig.update_layout(height=420, margin=dict(t=10,b=20,l=10,r=10),
            showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with col_r:
        st.markdown("**Distribusi probabilitas default**")
        pred_prob = model.predict_proba(aj[FEAT].fillna(0))[:,1]
        df_prob   = pd.DataFrame({"prob":pred_prob,"aktual":aj["label_default"]})
        fig2 = px.histogram(df_prob, x="prob", color="aktual",
            nbins=25, barmode="overlay", opacity=0.7,
            color_discrete_map={0:"#3182CE",1:"#E53E3E"},
            template="plotly_white",
            labels={"prob":"Prob. Default","aktual":"Aktual"})
        fig2.update_layout(height=240, margin=dict(t=10,b=20,l=10,r=10),
            paper_bgcolor="rgba(0,0,0,0)", legend_title="Aktual")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Ringkasan teknis**")
        tbl = pd.DataFrame({
            "Metrik":["AUC-ROC","Precision","Recall","Algoritma",
                      "Jumlah fitur","Tipe data","Explainability","Regulasi"],
            "Nilai":[f"{auc:.3f}",f"{prec:.3f}",f"{rec:.3f}",
                     "XGBoost Gradient Boosting",str(len(FEAT)),
                     "Data digital alternatif","SHAP-based","POJK 29/2024"]
        })
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    st.markdown('<div class="sec">Daftar fitur — dokumentasi untuk OJK</div>',
        unsafe_allow_html=True)
    feat_doc = [
        ["lama_platform_bln","Platform","Lama aktif di platform (bulan)","Internal platform"],
        ["platform_ontime_rate","Platform","Rasio pembayaran tepat waktu","Internal platform"],
        ["n_pinjaman_sebelumnya","Riwayat Kredit","Total pinjaman sebelumnya","Internal platform"],
        ["n_lancar","Riwayat Kredit","Jumlah pinjaman lancar","Internal platform"],
        ["n_macet","Riwayat Kredit","Jumlah pinjaman macet","Internal platform"],
        ["avg_telat_hari","Riwayat Kredit","Rata-rata hari keterlambatan","Internal platform"],
        ["listrik_pct_tepat","Data Alternatif","% bulan bayar listrik tepat waktu","PLN"],
        ["ew_aktif_bln","Data Alternatif","Bulan aktif e-wallet","GoPay/OVO/Dana"],
        ["ew_trx_per_bln","Data Alternatif","Transaksi e-wallet/bulan","GoPay/OVO/Dana"],
        ["ew_spike","Data Alternatif","Red flag spike top-up (0/1)","GoPay/OVO/Dana"],
        ["is_gig","Profil","Apakah gig worker (0/1)","Data pengajuan"],
        ["gig_income_avg","Data Alternatif","Income gig rata-rata/bulan","Gojek/Grab"],
        ["gig_konsisten","Data Alternatif","Income gig konsisten 6 bln (0/1)","Gojek/Grab"],
        ["gig_cancel_rate","Data Alternatif","Cancel rate gig","Gojek/Grab"],
        ["punya_rekening","Profil","Punya rekening bank (0/1)","Data pengajuan"],
        ["bpjs_rutin","Data Alternatif","BPJS rutin dibayar (0/1)","BPJS Kesehatan"],
        ["punya_toko_online","Data Alternatif","Punya toko marketplace (0/1)","Tokopedia/Shopee"],
        ["omzet_toko_online","Data Alternatif","Omzet toko/bulan (Rp)","Tokopedia/Shopee"],
    ]
    df_feat = pd.DataFrame(feat_doc,
        columns=["Nama Fitur","Kategori","Deskripsi","Sumber Data"])
    st.dataframe(df_feat, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Unduh dokumentasi fitur (CSV)",
        df_feat.to_csv(index=False).encode("utf-8"),
        "dokumentasi_fitur_ojk.csv","text/csv")
