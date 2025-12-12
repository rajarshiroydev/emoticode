# config.py
import torch

TRAIN_CSV = "dataset/competition_train.csv"
VAL_CSV = "dataset/competition_val.csv"
TEST_CSV = "dataset/competition_test.csv"

# Paths for Fused Embeddings
TRAIN_EMBEDS = "train_fused.pt"
VAL_EMBEDS = "val_fused.pt"
TEST_EMBEDS = "test_fused.pt"
SUBMISSION_FILE = "submission.csv"

# Dims: NLLB(1024) + LaBSE(768) = 1792
INPUT_DIM = 1792 
NUM_LABELS = 6

EMOTION_MAP = {'anger': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5}
ID_TO_EMOTION = {v: k for k, v in EMOTION_MAP.items()}

# Language Mapping (NLLB style)
LANG_MAP = {
    'Kashmiri': 'kas_Arab', 
    'Santali': 'sat_Beng', 
    'Manipuri': 'mni_Beng'
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 32
EPOCH = 30