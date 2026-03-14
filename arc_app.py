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
from score_z import decode_word
from score_y import decode_sentence
import streamlit.components.v1 as components



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
    "per_score": None,
    "dtw_score": None,
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

html, body {
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
    font-size: clamp(40px, 7vw, 72px);
    text-align: center;
    line-height: 1.1;
    color: #1A1D23;
    margin: 40px 0 48px 0;
    letter-spacing: -1px;
}
.arc-sentence-prompt {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(24px, 4vw, 44px);
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
    padding: 16px 28px;
    border-radius: 14px;
    border: none;
    cursor: pointer;
    transition: all 0.15s;

    width: 100%;
    min-width: 140px;

    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;

    white-space: nowrap;
}

/* Record */
button:has(span:contains("🎤")) {
    background-color: #10B981;
    color: white;
}

/* Stop */
button:has(span:contains("⏹")) {
    background-color: #EF4444;
    color: white;
}

/* Play */
button:has(span:contains("🔊")) {
    background-color: #2563EB;
    color: white;
}

div.stButton > button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}

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
/* Fix markdown text visibility */
.stMarkdown p {
    color: #1A1D23 !important;
}

.stMarkdown li {
    color: #1A1D23 !important;
}

.stMarkdown span {
    color: inherit !important;
}
/* Fix form labels */
label, .stTextInput label {
    color: #1A1D23 !important;
}            

/* Hide streamlit chrome */
#MainMenu, footer { visibility: hidden; }

/* Instruction caption styling */
div[data-testid="stCaptionContainer"] {
    text-align: center;
    font-size: 13px;
    color: #6B7280;
    margin-top: -10px;
    margin-bottom: 28px;
}

                                    
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

components.html(
"""
<script>

document.addEventListener('keydown', function(e) {

if (e.target.tagName === "INPUT") return;

    if (e.key === 'r' || e.key === 'R') {
        window.parent.document.querySelectorAll('button')
        .forEach(btn => { if(btn.innerText.includes("Record")) btn.click() })
    }

    if (e.key === 's' || e.key === 'S') {
        window.parent.document.querySelectorAll('button')
        .forEach(btn => { if(btn.innerText.includes("Stop")) btn.click() })
    }

    if (e.key === 'p' || e.key === 'P') {
        window.parent.document.querySelectorAll('button')
        .forEach(btn => { if(btn.innerText.includes("Play")) btn.click() })
    }

});

</script>
""",
height=0
)


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

    # close any previous stream safely
    old_stream = st.session_state.get("stream", None)

    if old_stream is not None:
        try:
            old_stream.stop()
            old_stream.close()
        except:
            pass

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
        )

        stream.start()

        st.session_state.stream = stream
        st.session_state._local_buffer = local_buffer
        st.session_state.recording = True

    except Exception as e:
        st.session_state.record_error = f"Could not open microphone: {e}"
        st.session_state.recording = False


def stop_recording_and_save(filename):

    stream = st.session_state.get("stream", None)

    try:
        if stream is not None:
            stream.stop()
            stream.close()
            st.session_state.stream = None
    except Exception as e:
        st.session_state.record_error = f"Error stopping stream: {e}"
        return False

    st.session_state.recording = False

    try:
        # safely read buffer
        local_buffer = st.session_state.get("_local_buffer", [])

        if not local_buffer:
            st.session_state.record_error = "No audio captured."
            return False

        audio = np.concatenate(local_buffer, axis=0)

        # enforce max duration
        max_samples = SAMPLE_RATE * MAX_DURATION_S
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # compute RMS
        rms = np.sqrt(np.mean(audio**2))

        if rms < 0.001:
            st.session_state.record_error = "Recording appears silent."
            return False

        # save wav
        sf.write(filename, audio, SAMPLE_RATE, subtype="PCM_16")

        # clear buffer after save
        st.session_state._local_buffer = []

        return rms

    except Exception as e:
        st.session_state.record_error = f"Failed to save audio: {e}"
        return False

def play_prompt(filepath):

    try:
        audio, sr = sf.read(filepath)
        sd.play(audio, sr)
        sd.wait()

    except Exception as e:
        st.warning(f"Playback failed: {e}")    

from vosk import Model, KaldiRecognizer
import json

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model")
@st.cache_resource
def load_model():
    return Model(MODEL_PATH)

vosk_model = load_model()

def transcribe_wav(filepath):

    with sf.SoundFile(filepath) as f:
        grammar = json.dumps(words + ["[unk]"])
        rec = KaldiRecognizer(vosk_model, f.samplerate, grammar)
        rec.SetWords(True)

        transcript = ""

        while True:
            data = f.read(8000, dtype="int16")
            
            if len(data) == 0:
                break
            
            data = data.tobytes()
            
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                transcript += " " + res.get("text", "")

        final = json.loads(rec.FinalResult())
        transcript += " " + final.get("text", "")

    return normalize_text(transcript.strip())

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

def normalize_text(text):
    return text.strip().replace("।", "").replace(".", "")

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
# PAGE: NEW ASSESSMENT
# =======================
if page == "New Assessment":

    # =======================
    # PHASE: PATIENT INFO
    # =======================
    if st.session_state.phase == "patient_info":

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

                import shutil

                # Clean old recordings before new test
                if os.path.exists(WORD_AUDIO_DIR):
                    shutil.rmtree(WORD_AUDIO_DIR)

                if os.path.exists(SENT_AUDIO_DIR):
                    shutil.rmtree(SENT_AUDIO_DIR)

                os.makedirs(WORD_AUDIO_DIR, exist_ok=True)
                os.makedirs(SENT_AUDIO_DIR, exist_ok=True)

                st.session_state.patient_id = patient_id.strip()
                st.session_state.clinician  = clinician.strip()
                st.session_state.phase      = "warmup"
                st.session_state.index      = 0

                st.rerun()

        st.markdown(f"""
        <div class="arc-info-card" style="margin-top:24px">
            <strong>Test Overview</strong><br>
            The patient will read <strong>{len(words)} words</strong> followed by
            <strong>{len(sentences)} sentences</strong> aloud.
            The clinician records each utterance. An ARC intelligibility score is computed at the end.
        </div>
        """, unsafe_allow_html=True)


    # =======================
    # PHASE: WARMUP
    # =======================
    elif st.session_state.phase == "warmup":

        warmup_words = ["एक", "दो"]
        total = len(warmup_words)
        idx   = st.session_state.index

        if idx < total:

            progress_bar(idx, total,
                        f"Warm-up {idx + 1} of {total}  ·  Patient: {st.session_state.patient_id}")

            st.markdown('<div class="arc-phase-label">Warm-up Calibration</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="arc-prompt">{warmup_words[idx]}</div>', unsafe_allow_html=True)

            record_error_display()

            if st.session_state.recording:
                st.markdown("""
                <div class="arc-recording">
                    <div class="arc-dot"></div> Recording…
                </div>
                """, unsafe_allow_html=True)

            spacer, col1, col2, spacer2 = st.columns([2,1,1,2])

            with col1:
                if st.button("🎤 Record", disabled=st.session_state.recording):
                    start_recording()
                    st.rerun()

            with col2:
                if st.button("⏹ Stop", disabled=not st.session_state.recording):

                    filename = "warmup_audio.wav"
                    rms = stop_recording_and_save(filename)

                    if rms:

                        MIN_RMS = 0.003
                        MAX_RMS = 0.5

                        if rms < MIN_RMS:
                            st.error("Speech too soft. Please speak louder and retry.")

                        elif rms > MAX_RMS:
                            st.error("Audio too loud or distorted. Reduce microphone gain.")

                        else:
                            st.success("Microphone calibration successful.")
                            st.session_state.index += 1
                            st.rerun()

        else:
            st.session_state.phase = "word"
            st.session_state.index = 0
            st.rerun()


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
            st.markdown(f'<div class="arc-sentence-prompt">{words[idx]}</div>', unsafe_allow_html=True)
            
            st.caption(
            "For standardised testing please avoid playing the prompt more than twice unless the patient did not hear it clearly."
            )

            record_error_display()

            if st.session_state.recording:
                st.markdown("""
                <div class="arc-recording">
                    <div class="arc-dot"></div> Recording…
                </div>
                """, unsafe_allow_html=True)

            spacer, col1, col2, col3, spacer2 = st.columns([1.2,1.2,1.2,1.2,1.2])

            with col1:
                if st.button("🎤 Record", disabled=st.session_state.recording):
                    start_recording()
                    st.rerun()

            with col2:
                if st.button("⏹ Stop", disabled=not st.session_state.recording):

                    filename = os.path.join(WORD_AUDIO_DIR, f"utt{idx+1:02d}.wav")
                    success = stop_recording_and_save(filename)

                    if success:
                        st.session_state.index += 1
                        st.rerun()

            with col3:

                prompt_file = f"audio_prompts_wav/utt{idx+1:02d}.wav"

                if st.button("🔊 Play", disabled=st.session_state.recording):

                    if os.path.exists(prompt_file):
                        play_prompt(prompt_file)
                    else:
                        st.warning("Prompt audio missing.")

        else:
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

            spacer, col1, col2, col3, spacer2 = st.columns([1.2,1.2,1.2,1.2,1.2])

            with col1:
                if st.button("🎤 Record", disabled=st.session_state.recording):
                    start_recording()
                    st.rerun()

            with col2:
                if st.button("⏹ Stop", disabled=not st.session_state.recording):

                    filename = os.path.join(SENT_AUDIO_DIR, f"{utt_id}.wav")
                    success = stop_recording_and_save(filename)

                    if success:
                        st.session_state.index += 1
                        st.rerun()

            with col3:

                prompt_file = f"audio_prompts_sent/{utt_id}.wav"

                if st.button("🔊 Play", disabled=st.session_state.recording):

                    if os.path.exists(prompt_file):
                        play_prompt(prompt_file)
                    else:
                        st.warning("Sentence prompt audio missing.")            

        else:
            st.session_state.phase = "result"
            st.session_state.index = 0
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

        # =======================
        # SHOW RESULT IF ALREADY COMPUTED
        # =======================
        if st.session_state.arc_score is not None:

            score = st.session_state.arc_score
            sev_label, sev_bg, sev_fg = severity_label(score)

            st.markdown(
            f"""
            <div class="arc-score-card">

            <div class="arc-score-label">ARC Score</div>
            <div class="arc-score-num">{score:.1f}</div>

            <div class="arc-severity" style="background:{sev_bg};color:{sev_fg}">
            {sev_label}
            </div>

            <div class="arc-sub-scores">

            <div class="arc-sub">
            <div class="arc-sub-num">{st.session_state.z_score:.1f}</div>
            <div class="arc-sub-label">Word Intelligibility (Z)</div>
            </div>

            <div class="arc-sub">
            <div class="arc-sub-num">{st.session_state.per_score:.1f}</div>
            <div class="arc-sub-label">Phonetic Accuracy (PER)</div>
            </div>

            <div class="arc-sub">
            <div class="arc-sub-num">{st.session_state.dtw_score:.1f}</div>
            <div class="arc-sub-label">Acoustic Similarity (DTW)</div>
            </div>

            <div class="arc-sub">
            <div class="arc-sub-num">{st.session_state.y_score:.1f}</div>
            <div class="arc-sub-label">Sentence Score (Y)</div>
            </div>

            </div>
            """,
            unsafe_allow_html=True
            )

            # ==========================
            # WITHIN SESSION CONSISTENCY
            # ==========================

            import pandas as pd

            try:
                z_df = pd.read_csv("z_results.csv")

                st.markdown("### Within-Session Consistency Check")

                repeat_words = (
                    z_df.groupby("reference")
                    .filter(lambda x: len(x) > 1)
                )

                if not repeat_words.empty:

                    grouped = repeat_words.groupby("reference")

                    for word, group in grouped:

                        scores = group["z"].tolist()
                        mean_score = round(sum(scores) / len(scores), 1)

                        st.markdown(f"**{word} (Mean: {mean_score}%)**")

                        for i, score in enumerate(scores, start=1):

                            if score >= 80:
                                color = "green"
                            elif score >= 50:
                                color = "orange"
                            else:
                                color = "red"

                            st.markdown(
                                f"- Attempt {i} — <span style='color:{color}'>{score:.0f}%</span>",
                                unsafe_allow_html=True
                            )

                        st.markdown("")

                else:
                    st.info("No repeated words found in this test.")

            except Exception as e:
                st.warning("Consistency analysis unavailable.")

            # ==========================

            if st.button("🔄  Start New Assessment"):

                for k, v in DEFAULTS.items():
                    st.session_state[k] = v

                st.rerun()


        
        # =======================
        # COMPUTE ARC SCORE
        # =======================
        else:

            if st.button("📊  Compute ARC Score"):

                progress = st.progress(0)
                status = st.empty()

                try:

                    # -----------------------
                    # STEP 1: cleanup
                    # -----------------------
                    status.text("Preparing scoring environment...")
                    progress.progress(10)

                    for f in ("z_results.csv", "y_results.csv"):
                        if os.path.exists(f):
                            os.remove(f)

                    # -----------------------
                    # STEP 2: word scoring
                    # -----------------------
                    status.text("Scoring word recordings (Z + PER + DTW)...")
                    progress.progress(30)

                    # run external scoring pipeline
                    result = subprocess.run(
                        [sys.executable, "score_z.py"],
                        capture_output=True,
                        text=True
                    )

                    if result.returncode != 0:
                        st.error("Word scoring failed")
                        st.error(result.stderr)
                        st.stop()

                    # load results produced by score_z.py
                    import pandas as pd

                    if not os.path.exists("z_results.csv"):
                        st.error("z_results.csv was not created by score_z.py")
                        st.stop()

                    z_df = pd.read_csv("z_results.csv")

                    # validate expected columns
                    required_cols = ["z", "per", "dtw"]
                    for col in required_cols:
                        if col not in z_df.columns:
                            st.error(f"Missing column '{col}' in z_results.csv")
                            st.stop()

                    # compute averages
                    z_score = float(z_df["z"].mean())
                    per_score = float(z_df["per"].mean())
                    dtw_score = float(z_df["dtw"].mean())

                    # store in session state
                    st.session_state.z_score = z_score
                    st.session_state.per_score = per_score
                    st.session_state.dtw_score = dtw_score

                    

                    # -----------------------
                    # STEP 3: sentence scoring
                    # -----------------------
                    status.text("Scoring sentence recordings (Y score)...")
                    progress.progress(55)

                    rows = []

                    for row in sentences:

                        utt_id = row["utt_id"]
                        reference = row["reference"]

                        wav_path = os.path.join(SENT_AUDIO_DIR, f"{utt_id}.wav")

                        hypothesis = decode_sentence(wav_path, vosk_model)

                        from difflib import SequenceMatcher
                        score = SequenceMatcher(None, reference, hypothesis).ratio() * 100

                        rows.append({
                            "utt_id": utt_id,
                            "reference": reference,
                            "hypothesis": hypothesis,
                            "y": round(score, 2)
                        })

                    y_df = pd.DataFrame(rows)
                    y_df.to_csv("y_results.csv", index=False)

                    

                    # -----------------------
                    # STEP 4: load results
                    # -----------------------
                    status.text("Loading scoring results...")
                    progress.progress(75)

                    import pandas as pd

                    if not os.path.exists("z_results.csv"):
                        st.error("score_z.py did not produce z_results.csv")
                        st.stop()

                    if not os.path.exists("y_results.csv"):
                        st.error("score_y.py did not produce y_results.csv")
                        st.stop()

                    z_df = pd.read_csv("z_results.csv")
                    y_df = pd.read_csv("y_results.csv")

                    # detect repeated words
                    repeat_words = (
                        z_df.groupby("reference")
                        .filter(lambda x: len(x) > 1)
                    )

                    if "z" not in z_df.columns:
                        st.error("z_results.csv must contain a 'z' column.")
                        st.stop()

                    if "y" not in y_df.columns:
                        st.error("y_results.csv must contain a 'y' column.")
                        st.stop()

                    # -----------------------
                    # STEP 5: compute ARC
                    # -----------------------
                    status.text("Computing ARC score...")
                    progress.progress(90)

                    z_score = float(z_df["z"].mean())
                    y_score = float(y_df["y"].mean())
                    arc_score = (z_score + y_score) / 2

                    for name, val in [("Z", z_score), ("Y", y_score), ("ARC", arc_score)]:
                        if not (0 <= val <= 100):
                            st.warning(
                                f"{name} score ({val:.2f}) is outside the expected 0–100 range."
                            )

                    st.session_state.z_score = z_score
                    st.session_state.y_score = y_score
                    st.session_state.arc_score = arc_score

                    # -----------------------
                    # STEP 6: save result
                    # -----------------------
                    status.text("Saving assessment...")
                    progress.progress(100)

                    save_assessment()

                    status.text("Scoring complete ✓")

                    st.rerun()

                except subprocess.TimeoutExpired:
                    st.error("Scoring timed out.")

                except FileNotFoundError as e:
                    st.error(f"Scoring script not found: {e}")

                except Exception as e:
                    st.error(f"Unexpected error during scoring: {e}")
# =======================
# PAGE: PATIENT HISTORY
# =======================
elif page == "Patient History":

    st.markdown("### Patient History")

    conn = sqlite3.connect("arc.db")
    c = conn.cursor()

    try:

        query = """
        SELECT
            a.patient_id,
            a.clinician,
            a.date,
            a.z_score,
            a.y_score,
            a.arc_score
        FROM assessments a
        ORDER BY a.date DESC
        """

        import pandas as pd
        df = pd.read_sql_query(query, conn)

        if df.empty:
            st.info("No assessments recorded yet.")
            st.stop()

        # Format date
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d %H:%M")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:
        st.error(f"Failed to load patient history: {e}")

    finally:
        conn.close()    