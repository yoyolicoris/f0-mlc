"""Convert MIR-1K .pv pitch labels (MIDI) to .csv (time, Hz)."""

import os
from glob import glob

import numpy as np
import rootutils
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", dotenv=True)

path_data = os.path.join(os.environ["DATA_DIR"], "MIR-1K")
dir_annot = "PitchLabel"
dir_annot_new = "PitchLabel_csv"
frame_period = 0.02  # seconds between consecutive frames

os.makedirs(os.path.join(path_data, dir_annot_new), exist_ok=True)

for file_path in tqdm(glob(os.path.join(path_data, dir_annot, "*.pv"))):
    data = np.loadtxt(file_path)

    # MIDI -> Hz, preserving unvoiced (0) frames
    data_hz = 440 * 2 ** ((data - 69) / 12)
    data_hz[data == 0] = 0

    t = (np.arange(len(data)) + 1) * frame_period
    data_new = np.stack((t, data_hz), axis=1)

    file_path_new = os.path.join(
        path_data, dir_annot_new, f"{os.path.basename(file_path)[:-3]}.csv"
    )
    np.savetxt(file_path_new, data_new, fmt="%.6f", delimiter=",")
