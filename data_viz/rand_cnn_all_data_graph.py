import csv
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# data config
FILE_PREFIX = "pong-stdim-10k"
PLOT_TITLE = f"F1 Scores for Pong using ST-DIM ({FILE_PREFIX})"
INDEX_OF_KEYWORD = 2
DATA_FOLDER = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "..", "runs")
CSV_FILES = glob.glob(os.path.join(DATA_FOLDER, f"{FILE_PREFIX}*.csv"))
GAME = 'Pong-v0'

# set game label information
game_mappings = {'Pong-v0': {'player_y': 0, 'enemy_y': 1,
                             'ball_x': 2, 'ball_y': 3, 'enemy_score': 4, 'player_score': 5}}
headers = [key for key in game_mappings[GAME].keys()]
shortcut_to_y_label = {
    "acc": "Accuracy",
    "f1": "F1 Score",
    "loss": "Loss"
}

# set viz options
style_str = 'seaborn-whitegrid'
# style_str = 'dark_background'
# style_str = 'fivethirtyeight'

plt.style.use(style_str)
fig, ax = plt.subplots(len(CSV_FILES))

for (file_index, file_path) in enumerate(CSV_FILES):
    file_name = os.path.basename(file_path)
    y_label = shortcut_to_y_label[file_name.split("_")[INDEX_OF_KEYWORD]]

    print(f"Processing \"{file_name}\"")
    pd_frame = pd.read_csv(file_path, header=None, names=headers)
    for column in pd_frame.columns:
        ax[file_index].plot(
            pd_frame.index, pd_frame[column], label=f"{column}")
        ax[file_index].set_title(f"{file_name}", fontweight="bold")
        ax[file_index].set_xlabel("Epoch")
        ax[file_index].set_ylabel(f"{y_label}")
plt.legend(bbox_to_anchor=(-0.05, 5), frameon=True)
plt.subplots_adjust(top=.98, bottom=0.05, hspace=0.5)
# full screen
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.show()
