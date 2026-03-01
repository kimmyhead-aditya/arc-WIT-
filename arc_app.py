"""
ARC Speech Intelligibility Test
Rewritten for clinical use — bug fixes, mobile/iPad compatibility,
robust error handling, and professional UI.
"""
import sys
import streamlit as st

st.set_page_config(
    page_title="ARC Speech Intelligibility Test",
    layout="centered",
    initial_sidebar_state="expanded",
)

# =======================
# IMPORTS
# =======================
import csv
import os
import subprocess
import numpy as np
import sounddevice as sd
import soundfile as sf


import sqlite3
from datetime import datetime


def init_db():
    conn = sqlite3.connect("arc.db")
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT UNIQUE,
            created_at TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            clinician TEXT,
            date TEXT,
            z_score REAL,
            y_score REAL,
            arc_score REAL,
            FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
        )
    """)

    conn.commit()
    conn.close()

init_db()    

# =======================
# NAVIGATION
# =======================

page = st.sidebar.radio(
    "Navigation",
    ["New Assessment", "Patient History"]
)

# =======================
# CONFIG
# =======================
WORD_FILE       = "words_test.txt"
SENTENCE_FILE   = "sentences.csv"
WORD_AUDIO_DIR  = "audio_16k"
SENT_AUDIO_DIR  = "audio_sent"
SAMPLE_RATE     = 16000
MAX_DURATION_S  = 30          # hard cap per utterance (seconds)

os.makedirs(WORD_AUDIO_DIR, exist_ok=True)
os.makedirs(SENT_AUDIO_DIR, exist_ok=True)

# =======================
# SESSION STATE — initialise once, cleanly
# =======================
DEFAULTS = {
    "phase":        "patient_info",   # patient_info → word → sentence → result
    "index":        0,
    "recording":    False,
    "audio_buffer": [],
    "stream":       None,
    "patient_id":   "",
    "clinician":    "",
    "arc_score":    None,
    "z_score":      None,
    "y_score":      None,
    "record_error": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =======================
# LOAD DATA — with graceful errors
# =======================
@st.cache_data
def load_words():
    if not os.path.exists(WORD_FILE):
        return None, f"Word file not found: {WORD_FILE}"
    try:
        with open(WORD_FILE, encoding="utf-8") as f:
            words = [w.strip() for w in f if w.strip()]
        if not words:
            return None, f"{WORD_FILE} is empty."
        return words, None
    except Exception as e:
        return None, str(e)

@st.cache_data
def load_sentences():
    if not os.path.exists(SENTENCE_FILE):
        return None, f"Sentence file not found: {SENTENCE_FILE}"
    try:
        with open(SENTENCE_FILE, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None, f"{SENTENCE_FILE} is empty."
        if "utt_id" not in rows[0] or "reference" not in rows[0]:
            return None, "sentences.csv must have columns: utt_id, reference"
        return rows, None
    except Exception as e:
        return None, str(e)

words, words_err = load_words()
sentences, sent_err = load_sentences()

# =======================
# STYLE — clean clinical aesthetic
# =======================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    background-color: #F7F8FA;
    color: #1A1D23;
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #F7F8FA;
}


/* Header bar */
.arc-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 0 12px 0;
    border-bottom: 1.5px solid #E2E5EC;
    margin-bottom: 36px;
}
.arc-logo {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    color: #1A1D23;
    letter-spacing: -0.5px;
}
.arc-logo span { color: #2563EB; }
.arc-badge {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #6B7280;
}

/* Progress bar */
.arc-progress-wrap {
    background: #E2E5EC;
    border-radius: 999px;
    height: 5px;
    margin-bottom: 10px;
    overflow: hidden;
}
.arc-progress-fill {
    height: 100%;
    background: #2563EB;
    border-radius: 999px;
    transition: width 0.4s ease;
}
.arc-progress-label {
    font-size: 12px;
    color: #6B7280;
    margin-bottom: 32px;
    font-weight: 500;
}

/* Prompt display */
.arc-prompt {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(52px, 10vw, 100px);
    text-align: center;
    line-height: 1.1;
    color: #1A1D23;
    margin: 40px 0 48px 0;
    letter-spacing: -1px;
}
.arc-sentence-prompt {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(28px, 5vw, 52px);
    text-align: center;
    line-height: 1.3;
    color: #1A1D23;
    margin: 40px 0 48px 0;
}

/* Phase label */
.arc-phase-label {
    text-align: center;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #2563EB;
    margin-bottom: 16px;
}

/* Recording indicator */
.arc-recording {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    color: #DC2626;
    font-weight: 600;
    font-size: 14px;
    margin-bottom: 8px;
}
.arc-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #DC2626;
    animation: blink 1s infinite;
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.2; }
}

/* Buttons */
div.stButton > button {
    font-family: 'DM Sans', sans-serif;
    font-size: 16px;
    font-weight: 600;
    padding: 14px 40px;
    border-radius: 10px;
    border: none;
    cursor: pointer;
    transition: all 0.15s;
    width: 100%;
}
div.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

/* Score card */
.arc-score-card {
    background: white;
    border: 1.5px solid #E2E5EC;
    border-radius: 16px;
    padding: 36px;
    text-align: center;
    margin: 24px 0;
    box-shadow: 0 2px 16px rgba(0,0,0,0.05);
}
.arc-score-num {
    font-family: 'DM Serif Display', serif;
    font-size: 96px;
    color: #1A1D23;
    line-height: 1;
    margin: 8px 0;
}
.arc-score-label {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #6B7280;
}
.arc-severity {
    display: inline-block;
    font-size: 14px;
    font-weight: 600;
    padding: 6px 18px;
    border-radius: 999px;
    margin-top: 16px;
}
.arc-sub-scores {
    display: flex;
    justify-content: center;
    gap: 32px;
    margin-top: 24px;
    padding-top: 24px;
    border-top: 1px solid #E2E5EC;
}
.arc-sub {
    text-align: center;
}
.arc-sub-num {
    font-family: 'DM Serif Display', serif;
    font-size: 36px;
    color: #1A1D23;
}
.arc-sub-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #9CA3AF;
    margin-top: 4px;
}

/* Info card */
.arc-info-card {
    background: white;
    border: 1.5px solid #E2E5EC;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 24px;
    font-size: 13px;
    color: #6B7280;
}
.arc-info-card strong { color: #1A1D23; }

/* Input label fix */
label { font-weight: 500 !important; font-size: 14px !important; }
            
/* Force headings to use dark text */
h1, h2, h3, h4, h5, h6 {
    color: #1A1D23 !important;
}

/* Fix form labels */
label, .stTextInput label {
    color: #1A1D23 !important;
}            

/* Hide streamlit chrome */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# =======================
# HEADER
# =======================
st.markdown("""
<div class="arc-header">
    <div class="arc-logo">ARC <span>Speech</span></div>
    <div class="arc-badge">Intelligibility Test</div>
</div>
""", unsafe_allow_html=True)


# =======================
# FATAL ERROR GUARD
# =======================
if words_err:
    st.error(f"⚠️ Cannot load words: {words_err}")
    st.stop()
if sent_err:
    st.error(f"⚠️ Cannot load sentences: {sent_err}")
    st.stop()


# =======================
# RECORDING FUNCTIONS
# =======================
def start_recording():
    st.session_state.record_error = None

    local_buffer = []

    def callback(indata, frames, time_info, status):
        if status:
            return
        local_buffer.append(indata.copy())

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=callback,
            device=None,
        )
        stream.start()

        st.session_state.stream = stream
        st.session_state._local_buffer = local_buffer
        st.session_state.recording = True  # move here

    except Exception as e:
        st.session_state.record_error = f"Could not open microphone: {e}"
        st.session_state.recording = False


def stop_recording_and_save(filename):
    """Stop recording, validate buffer, and write WAV. Returns True on success."""
    try:
        if st.session_state.stream:
            st.session_state.stream.stop()
            st.session_state.stream.close()
            st.session_state.stream = None
    except Exception as e:
        st.session_state.record_error = f"Error closing stream: {e}"

    st.session_state.recording = False

    try:
        local_buffer = st.session_state.get("_local_buffer", [])

        if not local_buffer:
            st.session_state.record_error = "No audio captured."
            return False

        audio = np.concatenate(local_buffer, axis=0)

        # Enforce max duration
        max_samples = SAMPLE_RATE * MAX_DURATION_S
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # Check for silence
        rms = np.sqrt(np.mean(audio**2))
        if rms < 0.001:
            st.session_state.record_error = "Recording appears silent. Please check microphone and try again."
            return False

        sf.write(filename, audio, SAMPLE_RATE, subtype="PCM_16")

        # Clear buffer safely
        st.session_state._local_buffer = []

        return True

    except Exception as e:
        st.session_state.record_error = f"Failed to save audio: {e}"
        return False


# =======================
# HELPERS
# =======================
def progress_bar(current, total, label=""):
    pct = int((current / total) * 100) if total else 0
    st.markdown(f"""
    <div class="arc-progress-wrap">
        <div class="arc-progress-fill" style="width:{pct}%"></div>
    </div>
    <div class="arc-progress-label">{label}</div>
    """, unsafe_allow_html=True)


def severity_label(score):
    """Return severity category and colour based on ARC score."""
    if score >= 90:
        return "Normal / Unimpaired", "#D1FAE5", "#065F46"
    elif score >= 75:
        return "Mild Impairment", "#FEF3C7", "#92400E"
    elif score >= 50:
        return "Moderate Impairment", "#FED7AA", "#9A3412"
    elif score >= 25:
        return "Severe Impairment", "#FEE2E2", "#991B1B"
    else:
        return "Profound Impairment", "#FEE2E2", "#7F1D1D"


def record_error_display():
    if st.session_state.record_error:
        st.error(st.session_state.record_error)
        st.session_state.record_error = None

# =======================
# DATABASE FUNCTIONS
# =======================

def save_assessment():
    conn = sqlite3.connect("arc.db")
    c = conn.cursor()

    c.execute("""
        INSERT OR IGNORE INTO patients (patient_id, created_at)
        VALUES (?, ?)
    """, (st.session_state.patient_id, datetime.now().isoformat()))

    c.execute("""
        INSERT INTO assessments
        (patient_id, clinician, date, z_score, y_score, arc_score)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        st.session_state.patient_id,
        st.session_state.clinician,
        datetime.now().isoformat(),
        st.session_state.z_score,
        st.session_state.y_score,
        st.session_state.arc_score
    ))

    conn.commit()
    conn.close()

# =======================
# PHASE: PATIENT INFO
# =======================
if page == "New Assessment":
    st.markdown("### Start New Assessment")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    with st.form("patient_form"):
        patient_id  = st.text_input("Patient ID / Name", placeholder="e.g. PT-2025-001")
        clinician   = st.text_input("Clinician Name", placeholder="e.g. Dr. Sharma")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        submitted   = st.form_submit_button("Begin Test →")

    if submitted:
        if not patient_id.strip():
            st.error("Please enter a Patient ID before starting.")
        else:
            st.session_state.patient_id = patient_id.strip()
            st.session_state.clinician  = clinician.strip()
            st.session_state.phase      = "word"
            st.session_state.index      = 0
            st.rerun()

    st.markdown("""
    <div class="arc-info-card" style="margin-top:24px">
        <strong>Test Overview</strong><br>
        The patient will read <strong>""" + str(len(words)) + """ words</strong> followed by
        <strong>""" + str(len(sentences)) + """ sentences</strong> aloud.
        The clinician records each utterance. An ARC intelligibility score is computed at the end.
    </div>
    """, unsafe_allow_html=True)

# =======================
# PATIENT HISTORY PAGE
# =======================
elif page == "Patient History":

    st.markdown("## Patient History")

    import sqlite3
    import pandas as pd

    conn = sqlite3.connect("arc.db")
    c = conn.cursor()

    patient_list = c.execute(
        "SELECT DISTINCT patient_id FROM assessments"
    ).fetchall()

    conn.close()

    if not patient_list:
        st.info("No patient data available yet.")
    else:
        patient_ids = [p[0] for p in patient_list]
        selected_patient = st.selectbox("Select Patient", patient_ids)

        conn = sqlite3.connect("arc.db")

        query = """
            SELECT date, z_score, y_score, arc_score
            FROM assessments
            WHERE patient_id = ?
            ORDER BY date ASC
        """

        data = conn.execute(query, (selected_patient,)).fetchall()
        conn.close()

        if data:
            df = pd.DataFrame(
                data,
                columns=["Date", "Z Score", "Y Score", "ARC Score"]
            )

            df["Date"] = pd.to_datetime(df["Date"])

            st.dataframe(df)

            st.markdown("### ARC Trend Over Time")
            st.line_chart(df.set_index("Date")[["ARC Score"]])    


# =======================
# PHASE: WORD
# =======================
elif st.session_state.phase == "word":

    total = len(words)
    idx   = st.session_state.index

    if idx < total:
        progress_bar(idx, total,
                     f"Word {idx + 1} of {total}  ·  Patient: {st.session_state.patient_id}")

        st.markdown('<div class="arc-phase-label">Word Reading</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="arc-prompt">{words[idx]}</div>', unsafe_allow_html=True)

        record_error_display()

        if st.session_state.recording:
            st.markdown("""
            <div class="arc-recording">
                <div class="arc-dot"></div> Recording…
            </div>
            """, unsafe_allow_html=True)
            if st.button("⏹  Stop Recording", key="stop_word"):
                filename = os.path.join(WORD_AUDIO_DIR, f"utt{idx+1:02d}.wav")
                success  = stop_recording_and_save(filename)
                if success:
                    st.session_state.index += 1  # only advance on success
                st.rerun()
        else:
            if st.button("🎤  Record", key="rec_word"):
                start_recording()
                st.rerun()
    else:
        # All words done — transition
        st.session_state.phase = "sentence"
        st.session_state.index = 0
        st.rerun()


# =======================
# PHASE: SENTENCE
# =======================
elif st.session_state.phase == "sentence":

    total = len(sentences)
    idx   = st.session_state.index

    if idx < total:
        row      = sentences[idx]
        utt_id   = row["utt_id"]
        sentence = row["reference"]

        progress_bar(idx, total,
                     f"Sentence {idx + 1} of {total}  ·  Patient: {st.session_state.patient_id}")

        st.markdown('<div class="arc-phase-label">Sentence Reading</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="arc-sentence-prompt">{sentence}</div>', unsafe_allow_html=True)

        record_error_display()

        if st.session_state.recording:
            st.markdown("""
            <div class="arc-recording">
                <div class="arc-dot"></div> Recording…
            </div>
            """, unsafe_allow_html=True)
            if st.button("⏹  Stop Recording", key="stop_sent"):
                filename = os.path.join(SENT_AUDIO_DIR, f"{utt_id}.wav")
                success  = stop_recording_and_save(filename)
                if success:
                    st.session_state.index += 1  # only advance on success
                st.rerun()
        else:
            if st.button("🎤  Record", key="rec_sent"):
                start_recording()
                st.rerun()
    else:
        st.session_state.phase = "result"
        st.rerun()


# =======================
# PHASE: RESULT
# =======================
elif st.session_state.phase == "result":

    st.markdown("### Assessment Complete")
    st.markdown(f"""
    <div class="arc-info-card">
        <strong>Patient:</strong> {st.session_state.patient_id} &nbsp;·&nbsp;
        <strong>Clinician:</strong> {st.session_state.clinician or '—'}
    </div>
    """, unsafe_allow_html=True)

    record_error_display()

    # Show cached score if already computed
    if st.session_state.arc_score is not None:
        score = st.session_state.arc_score
        sev_label, sev_bg, sev_fg = severity_label(score)
        st.markdown(f"""
        <div class="arc-score-card">
            <div class="arc-score-label">ARC Score</div>
            <div class="arc-score-num">{score:.1f}</div>
            <div class="arc-severity" style="background:{sev_bg};color:{sev_fg}">
                {sev_label}
            </div>
            <div class="arc-sub-scores">
                <div class="arc-sub">
                    <div class="arc-sub-num">{st.session_state.z_score:.1f}</div>
                    <div class="arc-sub-label">Word Score (Z)</div>
                </div>
                <div class="arc-sub">
                    <div class="arc-sub-num">{st.session_state.y_score:.1f}</div>
                    <div class="arc-sub-label">Sentence Score (Y)</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔄  Start New Assessment"):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()

    else:
        if st.button("📊  Compute ARC Score"):
            with st.spinner("Scoring recordings…"):
                try:
                    # Clean up old results
                    for f in ("z_results.csv", "y_results.csv"):
                        if os.path.exists(f):
                            os.remove(f)

                    # Run scoring scripts
                    z_result = subprocess.run(
                        [sys.executable, "score_z.py"],
                        capture_output=True, text=True, timeout=120
                    )
                    y_result = subprocess.run(
                        [sys.executable, "score_y.py"],
                        capture_output=True, text=True, timeout=120
                    )

                    # Check for script errors
                    if z_result.returncode != 0:
                        st.error(f"Word scoring failed:\n{z_result.stderr}")
                        st.stop()
                    if y_result.returncode != 0:
                        st.error(f"Sentence scoring failed:\n{y_result.stderr}")
                        st.stop()

                    # Validate result files exist
                    if not os.path.exists("z_results.csv"):
                        st.error("score_z.py ran but did not produce z_results.csv")
                        st.stop()
                    if not os.path.exists("y_results.csv"):
                        st.error("score_y.py ran but did not produce y_results.csv")
                        st.stop()

                    import pandas as pd
                    z_df = pd.read_csv("z_results.csv")
                    y_df = pd.read_csv("y_results.csv")

                    if "z" not in z_df.columns:
                        st.error("z_results.csv must contain a 'z' column.")
                        st.stop()
                    if "y" not in y_df.columns:
                        st.error("y_results.csv must contain a 'y' column.")
                        st.stop()

                    z_score   = float(z_df["z"].mean())
                    y_score   = float(y_df["y"].mean())
                    arc_score = (z_score + y_score) / 2

                    # Range validation
                    for name, val in [("Z", z_score), ("Y", y_score), ("ARC", arc_score)]:
                        if not (0 <= val <= 100):
                            st.warning(
                                f"{name} score ({val:.2f}) is outside the expected 0–100 range. "
                                "Check scoring scripts."
                            )

                    st.session_state.z_score   = z_score
                    st.session_state.y_score   = y_score
                    st.session_state.arc_score = arc_score

                    save_assessment()
                    st.rerun()

                except subprocess.TimeoutExpired:
                    st.error("Scoring timed out. The audio files may be too large or the scoring scripts are hanging.")
                except FileNotFoundError as e:
                    st.error(f"Scoring script not found: {e}")
                except Exception as e:
                    st.error(f"Unexpected error during scoring: {e}")
