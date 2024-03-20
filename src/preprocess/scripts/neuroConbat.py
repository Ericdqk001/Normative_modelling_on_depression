import pickle
from pathlib import Path

with open(Path("ABCD_mVAE_LizaEric/data/train_val_subs.pkl"), "rb") as f:
    TRAIN_VAL_SUBS = pickle.load(f)

with open(Path("ABCD_mVAE_LizaEric/data/subs_test.pkl"), "rb") as f:
    TEST_SUBS = pickle.load(f)
