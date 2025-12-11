# config.py
import torch

TRAIN_CSV = "dataset/competition_train.csv"
VAL_CSV = "dataset/competition_val.csv"
TEST_CSV = "dataset/competition_test.csv"

# We save SONAR embeddings here
TRAIN_EMBEDS = "train_sonar.pt"
VAL_EMBEDS = "val_sonar.pt"
TEST_EMBEDS = "test_sonar.pt"
SUBMISSION_FILE = "submission.csv"

# SONAR outputs 1024 dim vectors
INPUT_DIM = 1024 
NUM_LABELS = 6

EMOTION_MAP = {'anger': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5}
ID_TO_EMOTION = {v: k for k, v in EMOTION_MAP.items()}

# SONAR Language Codes
LANG_MAP = {
    'Kashmiri': 'kas_Arab', 
    'Santali': 'sat_Beng', 
    'Manipuri': 'mni_Beng'
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 128
EPOCH = 10