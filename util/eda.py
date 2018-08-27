import json
from multiprocessing.pool import Pool
from os import listdir
import pandas as pd
import numpy as np
import librosa
from os.path import join


def get_duration(file):
    duration = librosa.get_duration(filename=file)
    print(f"{file}: {duration}")
    return {"file": file, "duration": duration}


def analyze(files, info_tmp_file, reload=True):
    if reload:
        p = Pool(20)
        data = p.map(get_duration, files)
        json.dump(data, open(info_tmp_file, "w"))
    else:
        data = json.load(open(info_tmp_file))
    df = pd.DataFrame(data)
    print("TEST DATA EDA")
    print("\nDuration Analysis")
    print(df["duration"].describe(percentiles=np.linspace(0, 1, 41, endpoint=True)))
    print("\nTop 20 shortest file")
    print(df.sort_values(["duration"]).head(20).to_string(index=False, columns=["file", "duration"]))
    print("\nTop 20 longest file")
    print(df.sort_values(["duration"]).tail(20).to_string(index=False, columns=["file", "duration"]))


DATA_FOLDER = "../data"
TEST_FOLDER = join(DATA_FOLDER, "public_test")
TEST_INFO_FILE = "../tmp/test_info.json"
files = listdir(TEST_FOLDER)
files = [join(TEST_FOLDER, file) for file in files]
analyze(files, info_tmp_file=TEST_INFO_FILE, reload=True)

TRAIN_FOLDER = join(DATA_FOLDER, "train")
TRAIN_INFO_FILE = "../tmp/train_info.json"
subfolders = listdir(TRAIN_FOLDER)
files = []
for subfolder in subfolders:
    files_ = listdir(join(TRAIN_FOLDER, subfolder))
    files_ = [join(TRAIN_FOLDER, subfolder, file) for file in files_]
    n = len(files)
    print(f"{subfolder} -> {n}")
    files.extend(files_)
analyze(files, info_tmp_file=TRAIN_INFO_FILE, reload=True)
