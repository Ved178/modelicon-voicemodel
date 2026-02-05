import os
import numpy as np
import soundfile as sf

RAW_DIR = "data/raw_16k"
OUT_DIR = "data/processed/windows"

os.makedirs(OUT_DIR + "/mantra", exist_ok=True)
os.makedirs(OUT_DIR + "/non_mantra", exist_ok=True)

SR = 16000
WINDOW_SEC = 1.2      # length of each sample
STEP_SEC = 0.3        # overlap

win = int(WINDOW_SEC * SR)
step = int(STEP_SEC * SR)

def energy(chunk):
    return np.sqrt(np.mean(chunk**2))

for fname in os.listdir(RAW_DIR):
    if not fname.endswith(".wav"):
        continue

    audio, _ = sf.read(os.path.join(RAW_DIR, fname))

    print(f"Processing {fname}, length: {len(audio)/SR:.1f} sec")

    for i in range(0, len(audio) - win, step):
        chunk = audio[i:i+win]

        # auto-label using energy
        if energy(chunk) > 0.015:   # you can tune this
            label = "mantra"
        else:
            label = "non_mantra"

        out_name = f"{fname.replace('.wav','')}_{i}.npy"
        np.save(f"{OUT_DIR}/{label}/{out_name}", chunk)

print("Window splitting complete.")
