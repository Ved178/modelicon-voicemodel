import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
import time
import queue
import sounddevice as sd   # <-- new dependency for real-time mic

# -------------------------------------------------
# LOAD TFLITE MODEL
# -------------------------------------------------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="models/mantra_cnn.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

SR = 16000
WINDOW_SEC = 1.0
STEP_SEC = 0.2
MIN_GAP = 0.9
TARGET_W = 38
LOUD_ENOUGH = 0.02
CONF_THRESH = 0.80

# -------------------------------------------------
# AUDIO → MEL
# -------------------------------------------------
def audio_to_mel(audio):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_mels=64
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # ---- FIX: make width always = 38 ----
    h, w = mel_db.shape

    if w > TARGET_W:
        mel_db = mel_db[:, :TARGET_W]        # crop
    elif w < TARGET_W:
        pad = TARGET_W - w
        mel_db = np.pad(mel_db, ((0,0),(0,pad)), mode="constant")

    return mel_db.astype(np.float32)

def energy(chunk):
    return np.sqrt(np.mean(chunk**2))

def run_inference(mel):
    
    mel = mel[..., np.newaxis]      # (64, 38, 1)
    mel = np.expand_dims(mel, 0)    # (1, 64, 38, 1)

    interpreter.set_tensor(input_details[0]['index'], mel)
    interpreter.invoke()

    out = interpreter.get_tensor(output_details[0]['index'])
    return out[0]   # [p_non_mantra, p_mantra]

# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------
st.title("AI-Based Mantra Counter")

st.markdown("""
### Chant counter  
""")

start = st.button("Start Listening")
stop = st.button("Stop")

# Live display placeholders
count_box = st.empty()
status_box = st.empty()

# Shared state
if "running" not in st.session_state:
    st.session_state.running = False

if "count" not in st.session_state:
    st.session_state.count = 0

if "last_count_time" not in st.session_state:
    st.session_state.last_count_time = 0

# -------------------------------------------------
# REAL-TIME AUDIO STREAM
# -------------------------------------------------
audio_q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_q.put(indata.copy())

def process_stream():
    buffer = np.zeros(int(WINDOW_SEC * SR))
    step = int(STEP_SEC * SR)

    status_box.info("Listening... chant now!")

    with sd.InputStream(
        samplerate=SR,
        channels=1,
        callback=audio_callback,
        blocksize=int(STEP_SEC * SR),
    ):
        while st.session_state.running:
            try:
                chunk = audio_q.get(timeout=0.1).flatten()

                # update rolling buffer
                buffer = np.concatenate([buffer[len(chunk):], chunk])

                # compute features
                mel = audio_to_mel(buffer)
                probs = run_inference(mel)
                p_mantra = float(probs[1])
                e = energy(buffer)

                t = time.time()

                # HYBRID COUNTING RULE
                if (p_mantra > CONF_THRESH) and (e > LOUD_ENOUGH) and (t - st.session_state.last_count_time > MIN_GAP):
                    st.session_state.count += 1
                    st.session_state.last_count_time = t

                count_box.success(f"Chant count: **{st.session_state.count}**")

            except queue.Empty:
                pass

# -------------------------------------------------
# BUTTON LOGIC
# -------------------------------------------------
if start:
    st.session_state.running = True
    st.session_state.count = 0
    st.session_state.last_count_time = 0
    count_box.success(f"Chant count: **0**")
    process_stream()

if stop:
    st.session_state.running = False
    status_box.warning("Stopped listening.")
