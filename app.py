import streamlit as st
import os
import json
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ══════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════
st.set_page_config(
    page_title="मजदूर अधिकार सहायक",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════
# CUSTOM CSS — Full Visual Design
# ══════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@300;400;500;600;700;800;900&display=swap');

/* ── Root Variables ── */
:root {
    --clr-primary: #1e2952;
    --clr-accent: #c2410c;
    --clr-accent-light: #ea580c;
    --clr-saffron: #FF9933;
    --clr-green: #138808;
    --clr-bg: #faf6f0;
    --clr-surface: rgba(255,255,255,0.65);
    --clr-text: #1a1512;
    --clr-muted: #8c7b6b;
    --clr-border: rgba(194,149,106,0.12);
    --radius: 16px;
    --glass: blur(12px);
}

/* ── Global Overrides ── */
.stApp {
    background: linear-gradient(168deg, #faf6f0 0%, #f5ede0 30%, #f0e8d8 60%, #f5ede0 100%) !important;
    font-family: 'Noto Sans Devanagari', sans-serif !important;
}

/* Background texture */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background-image:
        radial-gradient(ellipse at 15% 20%, rgba(255,153,51,0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 85% 70%, rgba(19,136,8,0.05) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(30,41,82,0.03) 0%, transparent 60%);
}

/* Hide streamlit defaults */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 880px !important; }

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #1e2952 0%, #0f1a3a 40%, #0a1128 100%);
    margin: -1rem -4rem 0 -4rem;
    padding: 0;
    position: relative;
    overflow: hidden;
}
.tricolor-bar {
    display: flex; height: 4px;
}
.tricolor-bar div:nth-child(1) { flex:1; background:#FF9933; }
.tricolor-bar div:nth-child(2) { flex:1; background:#ffffff; }
.tricolor-bar div:nth-child(3) { flex:1; background:#138808; }

.header-content {
    display: flex; align-items: center; gap: 18px;
    max-width: 880px; margin: 0 auto; padding: 22px 24px;
    position: relative; z-index: 1;
}
.header-logo {
    width: 56px; height: 56px; border-radius: 14px;
    background: linear-gradient(135deg, #FF9933, #e6821a);
    display: flex; align-items: center; justify-content: center;
    font-size: 28px;
    box-shadow: 0 4px 20px rgba(255,153,51,0.3);
    animation: glow 3s ease-in-out infinite;
}
.header-title { margin:0; font-size:24px; color:#fff; font-weight:900; letter-spacing:-0.5px; }
.header-sub { margin:3px 0 0; font-size:13px; color:rgba(255,255,255,0.5); letter-spacing:1px; }

@keyframes glow {
    0%,100% { box-shadow: 0 0 20px rgba(255,153,51,0.2); }
    50% { box-shadow: 0 0 40px rgba(255,153,51,0.4); }
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(20px); }
    to { opacity:1; transform:translateY(0); }
}
@keyframes float {
    0%,100% { transform:translateY(0); }
    50% { transform:translateY(-8px); }
}
@keyframes pulse {
    0%,100% { transform:scale(1); }
    50% { transform:scale(1.03); }
}

/* ── Glass Card ── */
.glass-card {
    background: var(--clr-surface);
    backdrop-filter: var(--glass);
    -webkit-backdrop-filter: var(--glass);
    border-radius: var(--radius);
    padding: 26px 22px;
    border: 1px solid var(--clr-border);
    box-shadow: 0 8px 32px rgba(0,0,0,0.04);
    margin-bottom: 16px;
    animation: fadeUp 0.4s ease;
}

/* ── Hero Card ── */
.hero-card {
    position: relative; border-radius: 20px; overflow: hidden;
    background: linear-gradient(135deg, #1e2952 0%, #0f1a3a 50%, #1a1040 100%);
    box-shadow: 0 20px 60px rgba(15,26,58,0.25);
    padding: 36px 28px; margin-bottom: 28px;
    animation: fadeUp 0.5s ease;
}
.hero-card::before {
    content:''; position:absolute; right:-30px; top:-30px;
    width:180px; height:180px; border-radius:50%;
    background: radial-gradient(circle, rgba(255,153,51,0.15) 0%, transparent 70%);
}
.hero-card::after {
    content:''; position:absolute; left:-20px; bottom:-40px;
    width:140px; height:140px; border-radius:50%;
    background: radial-gradient(circle, rgba(19,136,8,0.1) 0%, transparent 70%);
}
.hero-title {
    font-size:28px; font-weight:900; line-height:1.3;
    background: linear-gradient(90deg, #fff, #FFD699);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 12px; position:relative; z-index:1;
}
.hero-text {
    font-size:15px; color:rgba(255,255,255,0.6); line-height:1.9;
    max-width:480px; margin:0; position:relative; z-index:1;
}
.hero-emojis {
    display:flex; gap:8px; margin-bottom:16px; position:relative; z-index:1;
}
.hero-emojis span { font-size:28px; animation: float 3s ease-in-out infinite; }
.hero-emojis span:nth-child(2) { animation-delay: 0.4s; }
.hero-emojis span:nth-child(3) { animation-delay: 0.8s; }

/* ── Feature Grid ── */
.feature-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(155px, 1fr));
    gap: 14px; margin-top: 18px;
}
.feature-card {
    background: rgba(255,255,255,0.7); backdrop-filter: blur(8px);
    border-radius: 16px; padding: 22px 16px;
    border: 1px solid var(--clr-border);
    text-align: center; cursor: pointer;
    transition: all 0.35s cubic-bezier(0.4,0,0.2,1);
    animation: fadeUp 0.4s ease both;
    text-decoration: none; color: var(--clr-text);
}
.feature-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 20px 40px rgba(15,23,42,0.12);
}
.feature-icon {
    font-size: 30px; width: 52px; height: 52px; border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 8px;
}
.feature-title { font-size:14px; font-weight:800; color:var(--clr-primary); margin-bottom:4px; }
.feature-desc { font-size:12px; color:var(--clr-muted); line-height:1.5; }

/* ── SOS Banner ── */
.sos-banner {
    background: linear-gradient(135deg, #dc2626, #991b1b);
    border-radius: 16px; padding: 18px 20px;
    display: flex; align-items: center; justify-content: center; gap: 12px;
    margin-top: 24px; animation: pulse 2s ease-in-out infinite;
    box-shadow: 0 8px 30px rgba(220,38,38,0.2);
    cursor: pointer; border: none; width: 100%;
    color: #fff; font-size: 16px; font-weight: 800;
    text-decoration: none;
}

/* ── Section Head ── */
.section-head {
    display: flex; align-items: flex-start; gap: 14px; margin-bottom: 22px;
}
.section-icon {
    font-size: 28px; width: 52px; height: 52px; border-radius: 16px; flex-shrink: 0;
    background: linear-gradient(135deg, rgba(194,65,12,0.1), rgba(255,153,51,0.1));
    display: flex; align-items: center; justify-content: center;
}
.section-title { margin:0; font-size:20px; font-weight:900; color:var(--clr-primary); }
.section-desc { margin:4px 0 0; font-size:13px; color:var(--clr-muted); }

/* ── Chat Bubbles ── */
.chat-container {
    background: linear-gradient(180deg, rgba(245,237,224,0.5), rgba(240,232,216,0.3));
    border-radius: 20px; padding: 18px; min-height: 320px;
    border: 1px solid var(--clr-border);
    box-shadow: inset 0 2px 12px rgba(0,0,0,0.03);
}
.chat-user {
    display:flex; justify-content:flex-end; margin-bottom:12px;
}
.chat-user .bubble {
    background: linear-gradient(135deg,#1e2952,#0f1a3a); color:#fff;
    padding:12px 18px; border-radius:18px 18px 4px 18px;
    max-width:80%; font-size:14px; line-height:1.8;
    box-shadow: 0 4px 15px rgba(30,41,82,0.15);
}
.chat-bot {
    display:flex; justify-content:flex-start; margin-bottom:12px;
}
.chat-bot .bubble {
    background: rgba(255,255,255,0.85); color:var(--clr-text);
    padding:12px 18px; border-radius:18px 18px 18px 4px;
    max-width:80%; font-size:14px; line-height:1.8;
    border: 1px solid var(--clr-border);
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}
.chat-empty {
    display:flex; flex-direction:column; align-items:center;
    justify-content:center; height:280px; color:#b8a894; gap:8px;
}
.chat-empty .icon { font-size:52px; animation: float 3s ease-in-out infinite; }

/* ── Quick Tags ── */
.quick-tag {
    display:inline-block; padding:8px 16px; border-radius:24px;
    border:1px solid rgba(194,149,106,0.2);
    background:rgba(255,255,255,0.6); backdrop-filter:blur(8px);
    font-size:13px; color:var(--clr-primary); font-weight:500;
    cursor:pointer; margin:0 6px 8px 0; transition:all 0.3s;
    text-decoration:none;
}
.quick-tag:hover {
    transform:translateY(-2px); box-shadow:0 8px 20px rgba(0,0,0,0.08);
    background:rgba(255,255,255,0.9);
}

/* ── Issue Grid ── */
.issue-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(95px, 1fr));
    gap: 10px; margin-top: 12px;
}
.issue-btn {
    display:flex; flex-direction:column; align-items:center; gap:6px;
    padding:14px 8px; border-radius:14px; cursor:pointer; text-align:center;
    border:1.5px solid var(--clr-border); background:rgba(255,255,255,0.5);
    transition:all 0.3s; font-family:inherit;
}
.issue-btn:hover { transform:translateY(-2px); box-shadow:0 8px 20px rgba(0,0,0,0.06); }
.issue-btn.active {
    border:2px solid var(--clr-accent);
    background:linear-gradient(135deg,#fff7ed,#fef3c7);
    box-shadow:0 4px 20px rgba(194,65,12,0.12);
}

/* ── Info Banners ── */
.law-banner {
    padding:14px 18px; border-radius:14px;
    background:linear-gradient(135deg,#eff6ff,#dbeafe);
    border:1px solid #93c5fd; font-size:13px; color:#1e40af;
    margin-top:16px; animation:fadeUp 0.3s ease;
}
.office-banner {
    padding:16px 18px; border-radius:14px;
    background:linear-gradient(135deg,#fffbeb,#fef3c7);
    border:1px solid #fde68a; margin-top:16px; animation:fadeUp 0.3s ease;
}
.office-banner h4 { margin:0 0 6px; color:#92400e; font-size:14px; }
.office-banner p { margin:0; font-size:13px; color:#78350f; line-height:1.7; }

/* ── Document Preview ── */
.doc-preview {
    background: rgba(250,246,240,0.8); border:1px solid var(--clr-border);
    border-radius:14px; padding:22px; font-size:13px; line-height:2.1;
    white-space:pre-wrap; max-height:400px; overflow-y:auto;
}

/* ── SOS Section ── */
.sos-header {
    border-radius:20px; overflow:hidden;
    background:linear-gradient(135deg,#dc2626,#991b1b,#7f1d1d);
    padding:28px 24px; text-align:center;
    box-shadow:0 12px 40px rgba(220,38,38,0.2);
    animation: glow 2s ease-in-out infinite;
    margin-bottom:24px; position:relative;
}
.sos-header::before {
    content:''; position:absolute; inset:0;
    background:radial-gradient(circle at 50% 0%, rgba(255,255,255,0.08), transparent 60%);
}
.sos-header h2 { margin:0; font-size:22px; color:#fff; font-weight:900; position:relative; }
.sos-header p { margin:6px 0 0; color:#fecaca; font-size:13px; position:relative; }

.helpline-card {
    display:flex; align-items:center; gap:14px; padding:16px 18px;
    background:rgba(255,255,255,0.7); backdrop-filter:blur(8px);
    border-radius:16px; border:1px solid var(--clr-border);
    margin-bottom:12px; transition:all 0.35s;
}
.helpline-card:hover { transform:translateY(-2px); box-shadow:0 12px 30px rgba(0,0,0,0.08); }
.helpline-icon {
    font-size:24px; width:48px; height:48px; border-radius:14px;
    background:linear-gradient(135deg,#dcfce7,#d1fae5);
    display:flex; align-items:center; justify-content:center;
}
.call-btn {
    padding:10px 18px; border-radius:12px; font-weight:800; font-size:14px;
    background:linear-gradient(135deg,#dcfce7,#bbf7d0); color:#15803d;
    text-decoration:none; white-space:nowrap; border:1px solid #86efac;
    box-shadow:0 2px 10px rgba(21,128,61,0.1);
}

/* ── Quiz ── */
.quiz-option {
    padding:15px 20px; border-radius:14px; font-size:15px; font-weight:600;
    border:1.5px solid var(--clr-border); background:rgba(255,255,255,0.5);
    cursor:pointer; text-align:left; transition:all 0.3s; margin-bottom:8px;
    width:100%;
}
.quiz-option:hover { background:rgba(255,255,255,0.8); }
.quiz-option.correct { border:2px solid #22c55e; background:linear-gradient(135deg,#dcfce7,#d1fae5); }
.quiz-option.wrong { border:2px solid #ef4444; background:linear-gradient(135deg,#fee2e2,#fecaca); }

/* ── Progress Bar ── */
.progress-track {
    height:8px; border-radius:8px; background:rgba(194,149,106,0.1);
    overflow:hidden; margin-top:14px;
}
.progress-fill {
    height:100%; border-radius:8px;
    background:linear-gradient(90deg,#FF9933,#138808);
    transition:width 0.6s;
}

/* ── Status Badge ── */
.status-badge {
    display:inline-block; padding:5px 14px; border-radius:24px;
    font-size:12px; font-weight:700;
    background:linear-gradient(135deg,#dbeafe,#eff6ff);
    color:#1e40af; border:1px solid #93c5fd;
}

/* ── Evidence ── */
.evidence-item {
    display:flex; gap:14px; align-items:center; padding:14px;
    border-radius:14px; background:rgba(255,255,255,0.5);
    border:1px solid var(--clr-border); margin-bottom:8px;
}

/* ── Primary Button Override ── */
.stButton > button {
    background: linear-gradient(135deg, #c2410c, #ea580c) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 12px 28px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    font-family: 'Noto Sans Devanagari', sans-serif !important;
    box-shadow: 0 6px 20px rgba(194,65,12,0.2) !important;
    transition: all 0.3s !important;
    width: 100%;
}
.stButton > button:hover {
    box-shadow: 0 8px 30px rgba(194,65,12,0.35) !important;
    transform: translateY(-2px);
}

/* ── Input Overrides ── */
.stTextInput input, .stTextArea textarea, .stSelectbox select, .stDateInput input {
    border-radius: 12px !important;
    border: 1.5px solid rgba(194,149,106,0.18) !important;
    background: rgba(255,255,255,0.6) !important;
    font-family: 'Noto Sans Devanagari', sans-serif !important;
    padding: 12px 16px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #c2410c !important;
    box-shadow: 0 0 0 4px rgba(194,65,12,0.08) !important;
}

/* ── Tab Overrides ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(250,246,240,0.85);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid rgba(194,149,106,0.15);
    gap: 0; border-radius: 0;
    padding: 0 8px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Noto Sans Devanagari', sans-serif !important;
    font-weight: 600 !important;
    color: #8c7b6b !important;
    padding: 12px 16px !important;
    font-size: 14px !important;
}
.stTabs [aria-selected="true"] {
    color: #c2410c !important;
    font-weight: 800 !important;
    border-bottom-color: #c2410c !important;
}

/* ── Footer ── */
.app-footer {
    text-align:center; padding:24px 16px 32px;
    border-top:1px solid rgba(194,149,106,0.1); margin-top:40px;
}
.footer-tricolor {
    display:flex; justify-content:center; gap:6px; margin-bottom:10px;
}
.footer-tricolor div { width:20px; height:3px; border-radius:2px; }

/* ── Misc ── */
.stExpander { border: 1px solid var(--clr-border) !important; border-radius: var(--radius) !important; }
.stDownloadButton > button { background: rgba(255,255,255,0.6) !important; color: #c2410c !important; border: 1.5px solid rgba(194,65,12,0.2) !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════
ISSUE_TYPES = {
    "💰 वेतन नहीं मिला": "Payment of Wages Act, 1936 / Minimum Wages Act, 1948",
    "⏰ ओवरटाइम नहीं दिया": "Factories Act, 1948 - Section 59",
    "🚪 बिना नोटिस निकाला": "Industrial Disputes Act, 1947 - Section 25F",
    "⚠️ असुरक्षित कार्यस्थल": "Factories Act, 1948 / Building Workers Act, 1996",
    "🛡️ उत्पीड़न / शोषण": "POSH Act, 2013 / IPC Section 354",
    "👶 बाल श्रम": "Child Labour (Prohibition) Act, 1986",
    "⛓️ बंधुआ मजदूरी": "Bonded Labour System (Abolition) Act, 1976",
    "🏦 PF / ESI नहीं काटा": "EPF Act, 1952 / ESI Act, 1948",
    "📋 अन्य समस्या": "",
}

STATES_DATA = {
    "दिल्ली": {"office": "श्रम आयुक्त कार्यालय, 5 शाम नाथ मार्ग, दिल्ली", "phone": "011-23962406"},
    "उत्तर प्रदेश": {"office": "श्रम आयुक्त कार्यालय, लखनऊ", "phone": "0522-2238218"},
    "मध्य प्रदेश": {"office": "श्रम आयुक्त कार्यालय, भोपाल", "phone": "0755-2551399"},
    "महाराष्ट्र": {"office": "श्रम आयुक्त कार्यालय, मुंबई", "phone": "022-24934091"},
    "राजस्थान": {"office": "श्रम आयुक्त कार्यालय, जयपुर", "phone": "0141-2227810"},
    "बिहार": {"office": "श्रम आयुक्त कार्यालय, पटना", "phone": "0612-2504810"},
    "हरियाणा": {"office": "श्रम आयुक्त कार्यालय, चंडीगढ़", "phone": "0172-2701373"},
    "गुजरात": {"office": "श्रम आयुक्त कार्यालय, अहमदाबाद", "phone": "079-25506789"},
    "तमिलनाडु": {"office": "श्रम आयुक्त कार्यालय, चेन्नई", "phone": "044-25671067"},
    "पंजाब": {"office": "श्रम आयुक्त कार्यालय, चंडीगढ़", "phone": "0172-2744011"},
}

HELPLINES = [
    {"name": "श्रम हेल्पलाइन", "number": "14434", "desc": "केंद्र सरकार श्रम शिकायत", "icon": "⚖️"},
    {"name": "महिला हेल्पलाइन", "number": "181", "desc": "महिला श्रमिकों के लिए", "icon": "👩"},
    {"name": "चाइल्ड लेबर", "number": "1098", "desc": "बाल श्रम शिकायत", "icon": "👶"},
    {"name": "पुलिस", "number": "100", "desc": "आपातकालीन स्थिति", "icon": "🚔"},
    {"name": "EPFO हेल्पलाइन", "number": "1800-118-005", "desc": "PF शिकायत", "icon": "🏦"},
    {"name": "ESIC हेल्पलाइन", "number": "1800-11-2526", "desc": "बीमा शिकायत", "icon": "🏥"},
]

QUIZ_QUESTIONS = [
    {"q": "न्यूनतम वेतन कौन तय करता है?", "options": ["मालिक", "सरकार", "कोर्ट", "पुलिस"], "correct": 1},
    {"q": "एक दिन में अधिकतम कितने घंटे काम?", "options": ["8 घंटे", "10 घंटे", "12 घंटे", "कोई सीमा नहीं"], "correct": 2},
    {"q": "ओवरटाइम का भुगतान कितना होना चाहिए?", "options": ["सामान्य", "1.5 गुना", "दोगुना", "तिगुना"], "correct": 2},
    {"q": "PF में कर्मचारी का योगदान?", "options": ["8%", "10%", "12%", "15%"], "correct": 2},
    {"q": "मजदूरी न मिलने पर शिकायत कहाँ?", "options": ["पुलिस", "श्रम आयुक्त", "कोर्ट", "ये सभी"], "correct": 3},
    {"q": "गर्भवती महिला को कितने दिन छुट्टी?", "options": ["30", "60", "182", "365"], "correct": 2},
]


# ══════════════════════════════════════════════════
# API KEY
# ══════════════════════════════════════════════════
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
elif os.environ.get("OPENAI_API_KEY"):
    pass
else:
    st.error("🔑 API Key नहीं मिली! कृपया Streamlit Secrets में 'OPENAI_API_KEY' जोड़ें।")
    st.stop()


# ══════════════════════════════════════════════════
# RAG SYSTEM INIT
# ══════════════════════════════════════════════════
@st.cache_resource
def initialize_rag():
    if not os.path.exists("workers_rights.txt"):
        st.error("workers_rights.txt फाइल नहीं मिली।")
        return None

    with open("workers_rights.txt", "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text])
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    chat_prompt = PromptTemplate(
        template="""आप एक अनुभवी कानूनी सहायक हैं। मजदूरों की समस्या का समाधान हिंदी में दें।

        नियम:
        1. संबंधित कानून (Act) का नाम जरूर लिखें।
        2. शिकायत करने का तरीका और हेल्पलाइन नंबर बताएं।
        3. भाषा बहुत सरल और मददगार होनी चाहिए।

        Context: {context}
        प्रश्न: {question}
        """,
        input_variables=['context', 'question']
    )

    complaint_prompt = PromptTemplate(
        template="""आप एक कानूनी दस्तावेज़ लेखक हैं। नीचे दी गई जानकारी के आधार पर
        श्रम आयुक्त कार्यालय को एक औपचारिक शिकायत पत्र हिंदी में लिखें।

        पत्र में शामिल करें: विषय, सम्बोधन, पूरा विवरण, प्रार्थना, और हस्ताक्षर।

        Context: {context}
        शिकायत विवरण: {question}
        """,
        input_variables=['context', 'question']
    )

    rti_prompt = PromptTemplate(
        template="""RTI Act 2005 के तहत एक औपचारिक RTI आवेदन पत्र हिंदी में लिखें।

        Context: {context}
        आवेदन विवरण: {question}
        """,
        input_variables=['context', 'question']
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def make_chain(prompt):
        return (
            RunnableParallel({'context': retriever | RunnableLambda(format_docs), 'question': RunnablePassthrough()})
            | prompt | llm | StrOutputParser()
        )

    return {
        "chat": make_chain(chat_prompt),
        "complaint": make_chain(complaint_prompt),
        "rti": make_chain(rti_prompt),
    }


# ══════════════════════════════════════════════════
# INIT SESSION STATE
# ══════════════════════════════════════════════════
defaults = {
    "messages": [],
    "cases": [],
    "evidence_files": [],
    "quiz_index": 0,
    "quiz_score": 0,
    "quiz_answered": None,
    "quiz_done": False,
    "complaint_submitted": False,
    "generated_doc": None,
    "rti_doc": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <div class="tricolor-bar"><div></div><div></div><div></div></div>
    <div class="header-content">
        <div class="header-logo">⚖️</div>
        <div>
            <h1 class="header-title">मजदूर अधिकार सहायक</h1>
            <p class="header-sub">WORKERS' RIGHTS PROTECTION PLATFORM</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════
chains = initialize_rag()

tab_home, tab_chat, tab_complaint, tab_docs, tab_evidence, tab_tracker, tab_quiz, tab_sos = st.tabs([
    "🏠 होम", "💬 AI सहायक", "📝 शिकायत दर्ज", "📄 दस्तावेज़",
    "📸 सबूत", "📊 ट्रैकर", "🧠 क्विज़", "🆘 SOS"
])


# ════════════════════════════════════════
# TAB: HOME
# ════════════════════════════════════════
with tab_home:
    st.markdown("""
    <div class="hero-card">
        <div class="hero-emojis"><span>🇮🇳</span><span>⚖️</span><span>✊</span></div>
        <h2 class="hero-title">आपके अधिकार,<br/>आपकी ताकत</h2>
        <p class="hero-text">भारत के हर मजदूर के लिए — अपने अधिकार जानें, शिकायत दर्ज करें, कानूनी दस्तावेज़ बनाएं, सबूत जमा करें।</p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(3)
    features = [
        ("📝", "ऑटो शिकायत", "AI से शिकायत पत्र बनाएं"),
        ("💬", "AI सहायक", "कानूनी सवाल पूछें"),
        ("📄", "RTI पत्र", "RTI आवेदन तैयार करें"),
        ("📸", "सबूत जमा", "फ़ोटो और दस्तावेज़ सेव"),
        ("📊", "केस ट्रैकर", "शिकायत की स्थिति देखें"),
        ("🧠", "अधिकार क्विज़", "खेल-खेल में सीखें"),
    ]
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon" style="background:rgba(194,65,12,0.08);">{icon}</div>
                <div class="feature-title">{title}</div>
                <div class="feature-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sos-banner">
        <span style="font-size:28px;">🆘</span>
        <span>आपातकालीन मदद — हेल्पलाइन नंबर → SOS टैब देखें</span>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════
# TAB: AI CHAT
# ════════════════════════════════════════
with tab_chat:
    st.markdown("""
    <div class="section-head">
        <div class="section-icon">💬</div>
        <div><h2 class="section-title">AI कानूनी सहायक</h2>
        <p class="section-desc">अपनी समस्या बताएं — AI कानूनी सलाह देगा</p></div>
    </div>
    """, unsafe_allow_html=True)

    # Quick suggestions
    quick_options = ["न्यूनतम वेतन की जानकारी", "मालिक पैसे नहीं दे रहा", "ओवरटाइम के नियम", "बिना नोटिस निकाल दिया"]
    selected = st.pills("सुझाव:", quick_options, selection_mode="single", label_visibility="collapsed")

    # Chat history display
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input handling
    user_query = None
    if selected:
        user_query = selected
    if prompt_input := st.chat_input("अपनी समस्या यहाँ विस्तार से लिखें..."):
        user_query = prompt_input

    if user_query and chains:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        with st.chat_message("assistant"):
            with st.spinner("🔍 समाधान खोजा जा रहा है..."):
                response = chains["chat"].invoke(user_query)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})


# ════════════════════════════════════════
# TAB: COMPLAINT
# ════════════════════════════════════════
with tab_complaint:
    st.markdown("""
    <div class="section-head">
        <div class="section-icon">📝</div>
        <div><h2 class="section-title">शिकायत दर्ज करें</h2>
        <p class="section-desc">जानकारी भरें — शिकायत पत्र और नजदीकी कार्यालय स्वचालित मिलेगा</p></div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.complaint_submitted:
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                c_name = st.text_input("आपका नाम *", placeholder="पूरा नाम लिखें")
                c_state = st.selectbox("राज्य *", ["-- राज्य चुनें --"] + list(STATES_DATA.keys()))
                c_employer = st.text_input("मालिक / कंपनी का नाम", placeholder="कंपनी / ठेकेदार")
            with col2:
                c_phone = st.text_input("फ़ोन नंबर", placeholder="मोबाइल नंबर")
                c_district = st.text_input("जिला", placeholder="जिले का नाम")
                c_date = st.date_input("घटना की तारीख", value=None)

            c_issue = st.selectbox("समस्या का प्रकार *", ["-- चुनें --"] + list(ISSUE_TYPES.keys()))

            if c_issue and c_issue != "-- चुनें --" and ISSUE_TYPES[c_issue]:
                st.markdown(f"""
                <div class="law-banner">📜 लागू कानून: <strong>{ISSUE_TYPES[c_issue]}</strong></div>
                """, unsafe_allow_html=True)

            c_desc = st.text_area("समस्या का विवरण *", placeholder="क्या हुआ, कब हुआ, कैसे हुआ — विस्तार से बताएं...", height=150)

            if c_state and c_state != "-- राज्य चुनें --" and c_state in STATES_DATA:
                info = STATES_DATA[c_state]
                st.markdown(f"""
                <div class="office-banner">
                    <h4>📍 नजदीकी श्रम कार्यालय</h4>
                    <p>{info['office']}<br/>📞 {info['phone']}</p>
                </div>
                """, unsafe_allow_html=True)

            if st.button("📄 शिकायत पत्र बनाएं और दर्ज करें", use_container_width=True):
                if c_name and c_issue != "-- चुनें --" and c_desc and c_state != "-- राज्य चुनें --" and chains:
                    with st.spinner("⏳ शिकायत पत्र बनाया जा रहा है..."):
                        complaint_text = f"""
                        शिकायतकर्ता: {c_name}, फ़ोन: {c_phone}
                        राज्य: {c_state}, जिला: {c_district}
                        नियोक्ता: {c_employer}
                        समस्या: {c_issue} ({ISSUE_TYPES.get(c_issue, '')})
                        विवरण: {c_desc}
                        तारीख: {c_date or 'आज'}
                        """
                        doc = chains["complaint"].invoke(complaint_text)
                        st.session_state.generated_doc = doc
                        st.session_state.complaint_submitted = True
                        st.session_state.cases.append({
                            "name": c_name, "issue": c_issue,
                            "state": c_state,
                            "date": datetime.now().strftime("%d/%m/%Y"),
                            "status": "दर्ज"
                        })
                        st.rerun()
                else:
                    st.warning("⚠️ कृपया नाम, राज्य, समस्या, और विवरण भरें।")
    else:
        st.success("✅ शिकायत सफलतापूर्वक दर्ज!")
        st.caption("पत्र डाउनलोड या कॉपी करें और श्रम कार्यालय में जमा करें।")
        st.markdown(f'<div class="doc-preview">{st.session_state.generated_doc}</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button("⬇️ डाउनलोड", st.session_state.generated_doc, file_name="shikayat_patra.txt", mime="text/plain")
        with col2:
            if st.button("📋 कॉपी"):
                st.code(st.session_state.generated_doc, language=None)
        with col3:
            if st.button("➕ नई शिकायत"):
                st.session_state.complaint_submitted = False
                st.session_state.generated_doc = None
                st.rerun()


# ════════════════════════════════════════
# TAB: DOCUMENTS (RTI)
# ════════════════════════════════════════
with tab_docs:
    st.markdown("""
    <div class="section-head">
        <div class="section-icon">📄</div>
        <div><h2 class="section-title">कानूनी दस्तावेज़ बनाएं</h2>
        <p class="section-desc">RTI आवेदन AI की मदद से बनाएं</p></div>
    </div>
    """, unsafe_allow_html=True)

    rti_name = st.text_input("आपका नाम", placeholder="पूरा नाम", key="rti_name")
    rti_dept = st.text_input("विभाग / कार्यालय *", placeholder="जैसे: श्रम विभाग, नगर निगम", key="rti_dept")
    rti_question = st.text_area("क्या जानकारी चाहिए? *", placeholder="विस्तार से लिखें...", height=120, key="rti_q")

    if st.button("📄 RTI पत्र बनाएं", key="rti_btn", use_container_width=True):
        if rti_dept and rti_question and chains:
            with st.spinner("⏳ RTI पत्र बनाया जा रहा है..."):
                rti_text = f"आवेदक: {rti_name or '___'}\nविभाग: {rti_dept}\nजानकारी: {rti_question}"
                doc = chains["rti"].invoke(rti_text)
                st.session_state.rti_doc = doc
        else:
            st.warning("⚠️ कृपया विभाग और प्रश्न भरें।")

    if st.session_state.rti_doc:
        st.markdown(f'<div class="doc-preview">{st.session_state.rti_doc}</div>', unsafe_allow_html=True)
        st.download_button("⬇️ RTI डाउनलोड", st.session_state.rti_doc, file_name="rti_application.txt", mime="text/plain")


# ════════════════════════════════════════
# TAB: EVIDENCE
# ════════════════════════════════════════
with tab_evidence:
    st.markdown("""
    <div class="section-head">
        <div class="section-icon">📸</div>
        <div><h2 class="section-title">सबूत जमा करें</h2>
        <p class="section-desc">फ़ोटो और दस्तावेज़ अपलोड करें — तारीख स्वचालित दर्ज होगी</p></div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("📁 फ़ोटो या PDF अपलोड करें", accept_multiple_files=True, type=["png", "jpg", "jpeg", "pdf"])
    if uploaded:
        for f in uploaded:
            exists = any(e["name"] == f.name for e in st.session_state.evidence_files)
            if not exists:
                st.session_state.evidence_files.append({
                    "name": f.name,
                    "size": f.size,
                    "type": f.type,
                    "time": datetime.now().strftime("%d/%m/%Y %H:%M"),
                })

    if st.session_state.evidence_files:
        st.markdown(f"**जमा सबूत ({len(st.session_state.evidence_files)})**")
        for i, ev in enumerate(st.session_state.evidence_files):
            col1, col2, col3 = st.columns([1, 4, 1])
            with col1:
                st.markdown("📄" if "pdf" in ev["type"] else "🖼️")
            with col2:
                st.markdown(f"**{ev['name']}**")
                st.caption(f"🕐 {ev['time']} • {ev['size']/1024:.1f} KB")
            with col3:
                if st.button("✕", key=f"del_{i}"):
                    st.session_state.evidence_files.pop(i)
                    st.rerun()


# ════════════════════════════════════════
# TAB: TRACKER
# ════════════════════════════════════════
with tab_tracker:
    st.markdown("""
    <div class="section-head">
        <div class="section-icon">📊</div>
        <div><h2 class="section-title">केस ट्रैकर</h2>
        <p class="section-desc">दर्ज शिकायतों की स्थिति</p></div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.cases:
        st.info("📋 अभी कोई शिकायत दर्ज नहीं है। 'शिकायत दर्ज' टैब से शिकायत करें।")
    else:
        for case in st.session_state.cases:
            with st.container():
                st.markdown(f"""
                <div class="glass-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                        <strong style="font-size:16px; color:#1e2952;">{case['name']}</strong>
                        <span class="status-badge">{case['status']}</span>
                    </div>
                    <p style="margin:10px 0 0; font-size:13px; color:#8c7b6b;">
                        समस्या: {case['issue']} | राज्य: {case['state']} | तारीख: {case['date']}
                    </p>
                    <div class="progress-track"><div class="progress-fill" style="width:33%;"></div></div>
                    <div style="display:flex; justify-content:space-between; margin-top:6px; font-size:11px; color:#b8a894;">
                        <span>● दर्ज</span><span>○ समीक्षा</span><span>○ समाधान</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ════════════════════════════════════════
# TAB: QUIZ
# ════════════════════════════════════════
with tab_quiz:
    st.markdown("""
    <div class="section-head">
        <div class="section-icon">🧠</div>
        <div><h2 class="section-title">अपने अधिकार जानें</h2>
        <p class="section-desc">क्विज़ खेलें और कानूनी अधिकार सीखें</p></div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.quiz_done:
        qi = st.session_state.quiz_index
        q = QUIZ_QUESTIONS[qi]

        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"सवाल {qi + 1} / {len(QUIZ_QUESTIONS)}")
        with col2:
            st.markdown(f"**⭐ स्कोर: {st.session_state.quiz_score}**")

        st.progress((qi + 1) / len(QUIZ_QUESTIONS))
        st.markdown(f"### {q['q']}")

        for j, opt in enumerate(q["options"]):
            label = opt
            if st.session_state.quiz_answered is not None:
                if j == q["correct"]:
                    label = f"✅ {opt}"
                elif j == st.session_state.quiz_answered:
                    label = f"❌ {opt}"

            if st.button(label, key=f"quiz_{qi}_{j}", use_container_width=True,
                         disabled=st.session_state.quiz_answered is not None):
                st.session_state.quiz_answered = j
                if j == q["correct"]:
                    st.session_state.quiz_score += 1
                st.rerun()

        if st.session_state.quiz_answered is not None:
            if qi + 1 >= len(QUIZ_QUESTIONS):
                if st.button("🏆 नतीजा देखें", use_container_width=True):
                    st.session_state.quiz_done = True
                    st.rerun()
            else:
                if st.button("अगला सवाल ➤", use_container_width=True):
                    st.session_state.quiz_index += 1
                    st.session_state.quiz_answered = None
                    st.rerun()
    else:
        score = st.session_state.quiz_score
        total = len(QUIZ_QUESTIONS)
        emoji = "🏆" if score >= 4 else "👍" if score >= 2 else "📚"
        st.markdown(f"""
        <div style="text-align:center; padding:30px 0;">
            <div style="font-size:64px;">{emoji}</div>
            <h2 style="background:linear-gradient(90deg,#1e2952,#c2410c);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                font-size:32px; margin:16px 0 6px;">{score} / {total}</h2>
            <p style="color:#8c7b6b; font-size:15px;">
                {"बहुत बढ़िया! आप अपने अधिकार जानते हैं!" if score >= 4 else "और सीखें! अधिकार जानना ज़रूरी है।"}</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔄 फिर से खेलें", use_container_width=True):
            st.session_state.quiz_index = 0
            st.session_state.quiz_score = 0
            st.session_state.quiz_answered = None
            st.session_state.quiz_done = False
            st.rerun()


# ════════════════════════════════════════
# TAB: SOS
# ════════════════════════════════════════
with tab_sos:
    st.markdown("""
    <div class="sos-header">
        <span style="font-size:44px; display:block; margin-bottom:8px; position:relative;">🆘</span>
        <h2>आपातकालीन सहायता</h2>
        <p>तुरंत मदद के लिए नीचे कॉल करें</p>
    </div>
    """, unsafe_allow_html=True)

    for h in HELPLINES:
        st.markdown(f"""
        <div class="helpline-card">
            <div class="helpline-icon">{h['icon']}</div>
            <div style="flex:1;">
                <div style="font-weight:700; font-size:15px; color:#1e2952;">{h['name']}</div>
                <div style="font-size:12px; color:#8c7b6b;">{h['desc']}</div>
            </div>
            <a href="tel:{h['number']}" class="call-btn">📞 {h['number']}</a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="office-banner" style="margin-top:24px;">
        <h4>💡 याद रखें</h4>
        <p>बंधुआ मजदूरी, बाल श्रम, या शारीरिक शोषण में तुरंत पुलिस (100) या
        चाइल्ड हेल्पलाइन (1098) कॉल करें। पहचान गोपनीय रखी जाएगी।</p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════
st.markdown("""
<div class="app-footer">
    <div class="footer-tricolor">
        <div style="background:#FF9933;"></div>
        <div style="background:#fff; border:1px solid #ddd;"></div>
        <div style="background:#138808;"></div>
    </div>
    <p style="margin:0; font-size:13px; color:#8c7b6b; font-weight:500;">⚖️ मजदूर अधिकार सहायक — हर मजदूर का साथी</p>
    <p style="margin:4px 0 0; font-size:11px; color:#b8a894;">यह ऐप कानूनी सलाह का विकल्प नहीं है। गंभीर मामलों में वकील से संपर्क करें।</p>
</div>
""", unsafe_allow_html=True)
