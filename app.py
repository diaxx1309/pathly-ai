import streamlit as st
from model import train_model

st.set_page_config(
    page_title="Pathly AI — Know Your Path Before It's Too Late",
    page_icon="🎓",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
* { font-family: 'Inter', sans-serif; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
section[data-testid="stMain"] {
    background: #080808 !important;
    color: #e8e8e8;
}
[data-testid="stHeader"] { background: #080808 !important; }

/* ── Hero ── */
.hero {
    padding: 4rem 0 2rem;
    text-align: center;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.3em;
    color: #ff1a5e;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-size: clamp(2.8rem, 6vw, 4.2rem);
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1;
    color: #ffffff;
    margin-bottom: 0.5rem;
}
.hero-title span {
    background: linear-gradient(90deg, #ff1a5e, #ff6b35);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-tagline {
    font-size: 1rem;
    color: #4a4a4a;
    font-weight: 400;
    margin-bottom: 0.6rem;
    letter-spacing: 0.01em;
}
.hero-story {
    font-size: 0.82rem;
    color: #333;
    max-width: 480px;
    margin: 0 auto 2.5rem;
    line-height: 1.7;
    font-style: italic;
}

/* ── Stats bar ── */
.stats {
    display: flex;
    justify-content: center;
    gap: 0;
    border: 1px solid #141414;
    border-radius: 8px;
    overflow: hidden;
    max-width: 420px;
    margin: 0 auto 3rem;
}
.stat-item {
    flex: 1;
    padding: 1rem 0;
    text-align: center;
    border-right: 1px solid #141414;
}
.stat-item:last-child { border-right: none; }
.stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 500;
    color: #ff1a5e;
    line-height: 1;
}
.stat-key {
    font-size: 0.6rem;
    color: #2e2e2e;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 0.35rem;
}

/* ── Form section ── */
.form-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    color: #2a2a2a;
    text-transform: uppercase;
    margin: 2.5rem 0 1rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #111;
}

/* Streamlit widget overrides */
div[data-testid="stSlider"] label,
div[data-testid="stSelectbox"] label {
    color: #3a3a3a !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    font-family: 'JetBrains Mono', monospace !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: #0e0e0e !important;
    border: 1px solid #1c1c1c !important;
    color: #e8e8e8 !important;
    border-radius: 6px !important;
    font-size: 0.9rem !important;
}
div[data-testid="stSlider"] [data-testid="stTickBar"] { display: none; }

.stButton > button {
    width: 100% !important;
    background: #ff1a5e !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.85rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    margin-top: 1.4rem !important;
    transition: background 0.2s !important;
    cursor: pointer !important;
}
.stButton > button:hover { background: #d4154f !important; }

/* ── Results ── */
.results-section { margin-top: 1rem; }
.result-divider {
    height: 1px;
    background: #111;
    margin: 2rem 0 1.5rem;
}
.result-label-row {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    color: #2a2a2a;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.card {
    background: #0d0d0d;
    border: 1px solid #161616;
    border-radius: 8px;
    padding: 1.4rem 1.5rem;
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 2px; height: 100%;
    background: linear-gradient(180deg, #ff1a5e, #ff6b35);
}
.card-eye {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    color: #2e2e2e;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.card-val {
    font-size: 1.55rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem;
}
.card-conf {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #ff1a5e;
}

/* ── Learning path ── */
.path-box {
    background: #0a0a0a;
    border: 1px solid #131313;
    border-radius: 8px;
    padding: 1.5rem;
    margin-top: 1rem;
}
.path-head {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: #2a2a2a;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}
.path-row {
    display: flex;
    gap: 1rem;
    align-items: flex-start;
    padding: 0.65rem 0;
    border-bottom: 1px solid #0f0f0f;
}
.path-row:last-child { border-bottom: none; }
.path-idx {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: #ff1a5e;
    opacity: 0.5;
    min-width: 1.4rem;
    padding-top: 0.15rem;
}
.path-text {
    font-size: 0.86rem;
    color: #888;
    line-height: 1.6;
}

/* ── Footer ── */
.foot {
    text-align: center;
    padding: 3rem 0 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    color: #1e1e1e;
    letter-spacing: 0.12em;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load model ──
exam_model, college_model, le_stream, le_interest, le_exam, le_college, exam_acc, college_acc = train_model()

# ── Hero ──
st.markdown(f"""
<div class="hero">
    <div class="hero-eyebrow">AI · Career Guidance · India</div>
    <div class="hero-title">PATHLY <span>AI</span></div>
    <p class="hero-tagline">Know your path before it's too late.</p>
    <p class="hero-story">
        After Class 12, most students don't know what exam to prepare for
        or which college to target. I built Pathly AI to solve exactly that —
        so no student has to figure it out alone.
    </p>
    <div class="stats">
        <div class="stat-item">
            <div class="stat-val">{exam_acc}%</div>
            <div class="stat-key">Exam Acc.</div>
        </div>
        <div class="stat-item">
            <div class="stat-val">{college_acc}%</div>
            <div class="stat-key">College Acc.</div>
        </div>
        <div class="stat-item">
            <div class="stat-val">8+</div>
            <div class="stat-key">Career Paths</div>
        </div>
        <div class="stat-item">
            <div class="stat-val">500+</div>
            <div class="stat-key">Students</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Inputs ──
st.markdown('<div class="form-label">Tell us about yourself</div>', unsafe_allow_html=True)

marks = st.slider("Your Class 12 Marks (%)", 40, 100, 75)
stream = st.selectbox("Your Stream", ["Science", "Commerce", "Arts"])
interest = st.selectbox("What excites you most?", [
    "Engineering", "Medical", "Computer Science",
    "Business", "Finance", "Management", "Law", "Design"
])

go = st.button("Find My Path →")

if go:
    try:
        s_enc = le_stream.transform([stream])[0]
        i_enc = le_interest.transform([interest])[0]
        inp = [[marks, s_enc, i_enc]]

        exam_pred = exam_model.predict(inp)
        college_pred = college_model.predict(inp)
        exam_result = le_exam.inverse_transform(exam_pred)[0]
        college_result = le_college.inverse_transform(college_pred)[0]
        exam_conf = round(max(exam_model.predict_proba(inp)[0]) * 100, 1)
        college_conf = round(max(college_model.predict_proba(inp)[0]) * 100, 1)

        st.markdown('<div class="result-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="result-label-row">Your Personalised Recommendation</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="card">
                <div class="card-eye">Target Exam</div>
                <div class="card-val">{exam_result}</div>
                <div class="card-conf">{exam_conf}% match</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="card">
                <div class="card-eye">College Type</div>
                <div class="card-val">{college_result}</div>
                <div class="card-conf">{college_conf}% match</div>
            </div>""", unsafe_allow_html=True)

        paths = {
            "JEE": [
                "Physics, Chemistry, Maths — master fundamentals before shortcuts",
                "Solve previous year JEE papers daily from 11th grade itself",
                "Physics Wallah or Unacademy work well for free/low-cost prep",
                "Target 85%+ in school exams alongside JEE prep"
            ],
            "NEET": [
                "NCERT is your actual syllabus — read every line, every diagram",
                "Practice 50 MCQs daily minimum — speed matters in NEET",
                "Biology alone is 360 marks — treat it as your scoring subject",
                "Start mock tests 6 months before the exam"
            ],
            "CAT": [
                "Quant, VARC, DILR — all three sections need equal attention",
                "Take one full mock CAT every week without skipping",
                "Reading editorials daily builds VARC speed over time",
                "Strong 12th Commerce base helps — don't ignore school"
            ],
            "CLAT": [
                "Legal Reasoning and GK are your highest scoring sections",
                "Read one quality newspaper every single morning",
                "Practice previous CLAT papers for pattern familiarity",
                "English comprehension matters more than most students think"
            ],
            "NIFT": [
                "Your portfolio will matter more than your marks here",
                "Sketch daily — even 20 minutes builds visual discipline",
                "Study fashion history and current global design trends",
                "Entrance has a situation test — practice creative thinking"
            ],
            "CA": [
                "Accountancy fundamentals from Class 11-12 are your foundation",
                "CA Foundation is cleared with consistent daily practice",
                "Join a study group early — CA is a long journey",
                "Don't skip mock tests — time management decides results"
            ],
        }

        tips = paths.get(exam_result, [
            "Build strong subject fundamentals — no shortcuts here",
            "Practice past papers to understand what's actually tested",
            "Find 2-3 good online resources and stick to them",
            "Consistency over 12 months beats cramming every time"
        ])

        rows = "".join([
            f'<div class="path-row"><div class="path-idx">0{i+1}</div><div class="path-text">{t}</div></div>'
            for i, t in enumerate(tips)
        ])

        st.markdown(f"""
        <div class="path-box">
            <div class="path-head">How to prepare for {exam_result}</div>
            {rows}
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"This stream + interest combination isn't in our training data yet. Try a different one. ({e})")

st.markdown('<div class="foot">PATHLY AI · Built for every Class 12 student who didn\'t know where to start</div>', unsafe_allow_html=True)