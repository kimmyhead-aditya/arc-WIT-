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
import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf
from score_z import decode_word
from score_y import decode_sentence
from scoring import score_words_inline, compute_wer
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
            per_score REAL,
            dtw_score REAL,
            severity TEXT,
            clinician_notes TEXT,  
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
    "clinician_notes":"",
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
    import librosa
    import numpy as np
    # warm up librosa so first DTW call has no cold-start penalty
    _dummy = np.zeros(16000, dtype=np.float32)
    librosa.feature.mfcc(y=_dummy, sr=16000, n_mfcc=13)
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
    return None, None, None


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

    label, _, _ = severity_label(st.session_state.arc_score)
    c.execute("""
        INSERT INTO assessments
        (patient_id, clinician, date, z_score, y_score, arc_score, per_score, dtw_score, severity, clinician_notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        st.session_state.patient_id,
        st.session_state.clinician,
        datetime.now().isoformat(),
        st.session_state.z_score,
        st.session_state.y_score,
        st.session_state.arc_score,
        st.session_state.per_score,
        st.session_state.dtw_score,
        label,
        st.session_state.clinician_notes
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

        # phonetically rich warmup words — covers stops, aspirates, nasals
        # chosen to exercise the full articulatory range the test depends on
        warmup_words = ["पानी", "हाथ"]
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

                    MIN_RMS = 0.003
                    MAX_RMS = 0.5

                    # explicit False check — rms=0.0 is a valid failure, not a pass
                    if rms is False:
                        st.error("Recording failed. Check microphone connection and retry.")

                    elif rms < MIN_RMS:
                        st.error(
                            f"Signal too weak (RMS: {rms:.4f}). "
                            "Please speak at normal conversational volume "
                            "and ensure microphone is within 7–10 cm."
                        )

                    elif rms > MAX_RMS:
                        st.error(
                            f"Signal clipping (RMS: {rms:.4f}). "
                            "Microphone gain is too high — reduce input volume "
                            "in system sound settings and retry."
                        )

                    else:
                        st.success(
                            f"Microphone calibrated ✓  "
                            f"(RMS: {rms:.4f} — within acceptable range)"
                        )
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
            st.caption(
                "For standardised testing please avoid playing the prompt more than once unless the patient did not hear it clearly."
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

        st.session_state.clinician_notes = st.text_area(
            "Clinical Observations",
            value=st.session_state.clinician_notes,
            placeholder="e.g. Patient fatigued today. Medication changed last week. Recorded at 7cm with headset.",
            height=100,
            help="These notes are saved with the assessment and visible in Patient History."
        )

        record_error_display()

        # =======================
        # SHOW RESULT IF ALREADY COMPUTED
        # =======================
        if st.session_state.arc_score is not None:

            score = st.session_state.arc_score
            

            st.markdown(
            f"""
            <div class="arc-score-card">

            <div class="arc-score-label">ARC Score</div>
            <div class="arc-score-num">{score:.1f}</div>

            

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

            
               

            st.markdown("### Within-Session Consistency Check")

            z_df = st.session_state.get("z_df", None)

            if z_df is None or z_df.empty:
                st.info("No data available for consistency analysis.")
            else:

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

            # ==========================

            if st.button("🔄  Start New Assessment"):

                for k, v in DEFAULTS.items():
                    st.session_state[k] = v

                st.rerun()

            # ==========================
            # PHONETIC RADAR CHART
            # ==========================

            st.markdown("### Phonetic Category Profile")
            st.caption("Accuracy by phonetic category based on words attempted in this session.")

            import plotly.graph_objects as go

            CATEGORY_MAP = {
                "Stops":        ["क", "ग", "ट", "ड", "त", "द", "प", "ब"],
                "Aspirated":    ["ख", "घ", "छ", "झ", "ठ", "ढ", "थ", "ध", "फ", "भ"],
                "Nasals":       ["न", "म"],
                "Fricatives":   ["स", "ह"],
                "Liquids and Glides": ["य", "र", "ल", "व"],
            }

            category_scores = {}

            for category, chars in CATEGORY_MAP.items():
                matching = z_df[
                    z_df["reference"].apply(
                        lambda w: any(c in w for c in chars)
                    )
                ]
                if not matching.empty:
                    category_scores[category] = round(matching["per"].mean(), 1)
                else:
                    category_scores[category] = None

            labels  = list(category_scores.keys())
            values  = [category_scores[k] if category_scores[k] is not None else 0 for k in labels]
            has_data = [category_scores[k] is not None for k in labels]

            # close the polygon
            labels_closed  = labels + [labels[0]]
            values_closed  = values + [values[0]]

            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=labels_closed,
                fill="toself",
                fillcolor="rgba(37, 99, 235, 0.12)",
                line=dict(color="#2563EB", width=2.5),
                marker=dict(size=7, color="#2563EB"),
                name="Phonetic Accuracy",
                hovertemplate="%{theta}: %{r:.1f}%<extra></extra>",
            ))

            # grey dot for missing categories
            missing_labels = [labels[i] for i in range(len(labels)) if not has_data[i]]
            if missing_labels:
                fig.add_trace(go.Scatterpolar(
                    r=[0] * len(missing_labels),
                    theta=missing_labels,
                    mode="markers",
                    marker=dict(size=7, color="#D1D5DB"),
                    name="No data",
                    hovertemplate="%{theta}: No words tested<extra></extra>",
                ))

            fig.update_layout(
                polar=dict(
                    bgcolor="#F7F8FA",
                    angularaxis=dict(
                        tickfont=dict(family="DM Sans", size=13, color="#1A1D23"),
                        linecolor="#E2E5EC",
                        rotation=90,
                        direction="clockwise",
                    ),
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        tickvals=[0, 25, 50, 75, 100],
                        tickfont=dict(family="DM Sans", size=10, color="#9CA3AF"),
                        gridcolor="#E2E5EC",
                        linecolor="#E2E5EC",
                    ),
                ),
                showlegend=False,
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                margin=dict(t=20, b=20, l=40, r=40),
                height=380,
                font=dict(family="DM Sans"),
                dragmode=False,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "staticPlot": True,        # disables ALL interaction — no drag, no zoom, no rotate
                    "displayModeBar": False,   # hides the plotly toolbar
                }
            )

            if st.button("↺  Reset Chart View"):
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
                    
                    # -----------------------
                    # STEP 2: word scoring (INLINE)
                    # -----------------------
                    status.text("Scoring word recordings (Z + PER + DTW)...")
                    progress.progress(30)

                    z_df = score_words_inline(WORD_AUDIO_DIR, words, vosk_model)
                    

                    # compute averages
                    z_score = float(z_df["z"].mean())
                    per_score = float(z_df["per"].mean())
                    dtw_score = float(z_df["dtw"].mean())

                    # store in session state
                    st.session_state.z_score = z_score
                    st.session_state.per_score = per_score
                    st.session_state.dtw_score = dtw_score
                    st.session_state.z_df = z_df

                    

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

                        if not os.path.exists(wav_path):
                            hypothesis = ""
                            st.warning(f"Missing audio for {utt_id}")
                        else:
                            hypothesis = decode_sentence(wav_path, vosk_model)

                        reference = normalize_text(reference)
                        hypothesis = normalize_text(hypothesis)
                        if not hypothesis.strip():
                            score = 0.0
                        else:
                            
                            wer = compute_wer(reference, hypothesis)

                            # 🔥 amplified Y
                            score = (1 - wer) ** 1.5 * 100

                            # clamp
                            score = max(0, min(100, score))

                        rows.append({
                            "utt_id": utt_id,
                            "reference": reference,
                            "hypothesis": hypothesis,
                            "y": round(score, 2)
                        })

                    y_df = pd.DataFrame(rows)
                    

                    

                    # -----------------------
                    # STEP 4: load results
                    # -----------------------
                    status.text("Loading scoring results...")
                    progress.progress(75)

                    import pandas as pd

                    

                    

                    
                    

                    # detect repeated words
                    repeat_words = (
                        z_df.groupby("reference")
                        .filter(lambda x: len(x) > 1)
                    )

                    

                    

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

    try:
        import pandas as pd
        import plotly.graph_objects as go

        # load all assessments
        query = """
        SELECT
            a.patient_id,
            a.clinician,
            a.date,
            a.z_score,
            a.y_score,
            a.arc_score,
            a.per_score,
            a.dtw_score,
            a.clinician_notes
        FROM assessments a
        ORDER BY a.date ASC
        """

        df = pd.read_sql_query(query, conn)

        if df.empty:
            st.info("No assessments recorded yet.")
            st.stop()

        df["date"] = pd.to_datetime(df["date"])

        # -------------------------
        # PATIENT SELECTOR
        # -------------------------
        patients = sorted(df["patient_id"].unique().tolist())

        selected = st.selectbox(
            "Select Patient",
            patients,
            index=0
        )

        patient_df = df[df["patient_id"] == selected].copy()
        patient_df = patient_df.sort_values("date").reset_index(drop=True)
        patient_df["session"] = [f"S{i+1}" for i in range(len(patient_df))]

        st.markdown(f"""
        <div class="arc-info-card">
            <strong>Patient:</strong> {selected} &nbsp;·&nbsp;
            <strong>Total Sessions:</strong> {len(patient_df)} &nbsp;·&nbsp;
            <strong>First Assessment:</strong> {patient_df['date'].iloc[0].strftime('%d %b %Y')} &nbsp;·&nbsp;
            <strong>Last Assessment:</strong> {patient_df['date'].iloc[-1].strftime('%d %b %Y')}
        </div>
        """, unsafe_allow_html=True)

        # -------------------------
        # TREND CHART
        # -------------------------
        st.markdown("### Score Trajectory")
        st.caption("Each point is one full ARC assessment session. Track how speech metrics change over time.")

        fig = go.Figure()

        # ARC
        fig.add_trace(go.Scatter(
            x=patient_df["session"],
            y=patient_df["arc_score"],
            mode="lines+markers",
            name="ARC Score",
            line=dict(color="#2563EB", width=3),
            marker=dict(size=9, color="#2563EB"),
            hovertemplate="<b>%{x}</b><br>ARC: %{y:.1f}<extra></extra>",
        ))

        # Z
        fig.add_trace(go.Scatter(
            x=patient_df["session"],
            y=patient_df["z_score"],
            mode="lines+markers",
            name="Word Score (Z)",
            line=dict(color="#10B981", width=2, dash="dot"),
            marker=dict(size=7, color="#10B981"),
            hovertemplate="<b>%{x}</b><br>Z: %{y:.1f}<extra></extra>",
        ))

        # Y
        fig.add_trace(go.Scatter(
            x=patient_df["session"],
            y=patient_df["y_score"],
            mode="lines+markers",
            name="Sentence Score (Y)",
            line=dict(color="#F59E0B", width=2, dash="dot"),
            marker=dict(size=7, color="#F59E0B"),
            hovertemplate="<b>%{x}</b><br>Y: %{y:.1f}<extra></extra>",
        ))

        fig.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F7F8FA",
            font=dict(family="DM Sans", size=13, color="#1A1D23"),
            yaxis=dict(
                range=[0, 100],
                gridcolor="#E2E5EC",
                title="Score",
                tickfont=dict(size=11, color="#6B7280"),
            ),
            xaxis=dict(
                gridcolor="#E2E5EC",
                title="Session",
                tickfont=dict(size=11, color="#6B7280"),
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                font=dict(size=12),
            ),
            margin=dict(t=40, b=40, l=40, r=20),
            height=380,
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True, config={
            "displayModeBar": False,
        })

        # -------------------------
        # SESSION DETAIL TABLE
        # -------------------------
        st.markdown("### Session Log")

        display_df = patient_df[[
            "session", "date", "clinician",
            "arc_score", "z_score", "y_score",
            "per_score", "dtw_score", "clinician_notes"
        ]].copy()

        display_df["date"] = display_df["date"].dt.strftime("%d %b %Y  %H:%M")
        display_df.columns = [
            "Session", "Date", "Clinician",
            "ARC", "Z (Word)", "Y (Sentence)",
            "PER", "DTW", "Clinical Notes"
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # -------------------------
        # CHANGE SUMMARY
        # -------------------------
        if len(patient_df) >= 2:
            st.markdown("### Change from First to Last Session")

            first = patient_df.iloc[0]
            last  = patient_df.iloc[-1]

            def delta_card(label, first_val, last_val):
                delta = last_val - first_val
                color = "#10B981" if delta >= 0 else "#EF4444"
                arrow = "▲" if delta >= 0 else "▼"
                st.markdown(f"""
                <div class="arc-info-card" style="text-align:center">
                    <div style="font-size:11px;font-weight:600;letter-spacing:1px;
                                text-transform:uppercase;color:#6B7280">{label}</div>
                    <div style="font-size:28px;font-family:'DM Serif Display',serif;
                                color:#1A1D23;margin:4px 0">{last_val:.1f}</div>
                    <div style="font-size:14px;font-weight:600;color:{color}">
                        {arrow} {abs(delta):.1f} pts from Session 1
                    </div>
                </div>
                """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                delta_card("ARC Score", first["arc_score"], last["arc_score"])
            with col2:
                delta_card("Word Score (Z)", first["z_score"], last["z_score"])
            with col3:
                delta_card("Sentence Score (Y)", first["y_score"], last["y_score"])

    except Exception as e:
        st.error(f"Failed to load patient history: {e}")

    finally:
        conn.close() 