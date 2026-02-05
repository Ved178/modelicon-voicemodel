import os
import numpy as np
import librosa

WIN_DIR = "data/processed/windows"
FEAT_DIR = "data/processed/features"

os.makedirs(FEAT_DIR, exist_ok=True)

X = []
y = []

def to_mel(chunk):
    mel = librosa.feature.melspectrogram(
        y=chunk,
        sr=16000,
        n_mels=64
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

for label, lab_id in [("mantra",1), ("non_mantra",0)]:
    path = os.path.join(WIN_DIR, label)

    for f in os.listdir(path):
        chunk = np.load(os.path.join(path, f))
        feat = to_mel(chunk)

        X.append(feat)
        y.append(lab_id)

X = np.array(X)
y = np.array(y)

np.save(FEAT_DIR + "/X.npy", X)
np.save(FEAT_DIR + "/y.npy", y)

print("Feature dataset created:", X.shape)
