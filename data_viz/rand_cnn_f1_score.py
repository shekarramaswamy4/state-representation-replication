import csv
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# data config

FILE_PREFIX = "pong-stdim-50k-samples-43-epochs"
PLOT_TITLE = f"F1 Scores for Pong ({FILE_PREFIX})"
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

# Filter for only F1 Scores
files_to_plot = []
for file_path in CSV_FILES:
    file_name = os.path.basename(file_path)
    y_label = shortcut_to_y_label[file_name.split("_")[INDEX_OF_KEYWORD]]

    if y_label != "F1 Score":
        continue
    else:
        files_to_plot.append(file_path)

plt.style.use(style_str)
num_subplots = 3
vars_per_subplot = 2
fig, ax = plt.subplots(num_subplots)
plt.suptitle(f"{PLOT_TITLE}",
             fontweight="bold", va="center", fontsize="xx-large")

color_cycle = ['red', 'green', 'orange', 'blue', 'violet', 'black']

for (file_index, file_path) in enumerate(files_to_plot):
    file_name = os.path.basename(file_path)
    y_label = shortcut_to_y_label[file_name.split("_")[INDEX_OF_KEYWORD]]

    print(f"Processing \"{file_name}\"")
    pd_frame = pd.read_csv(file_path, header=None, names=headers)
    cur_title = ""
    for (col_ind, column) in enumerate(pd_frame.columns):
        subplot_index = col_ind // vars_per_subplot

        ax[subplot_index].plot(
            pd_frame.index, pd_frame[column], label=f"{column}", color=color_cycle[col_ind % len(color_cycle)])
        ax[subplot_index].set_xlabel("Epoch")
        ax[subplot_index].set_ylabel(f"{y_label}")
        ax[subplot_index].legend(
            frameon=True, loc="lower right", fontsize="x-large")
        # ax[subplot_index].legend(bbox_to_anchor=(-0.05, 0.7), frameon=True)

        # finalize subplot
        cur_title += f"{column} & "
        if (col_ind + 1) % vars_per_subplot == 0:
            ax[subplot_index].set_title(
                f"{cur_title[:-3]}", fontweight="bold", fontsize="x-large")
            cur_title = ""
            ax[subplot_index].set_ylim([0.0, 1.01])

plt.subplots_adjust(bottom=0.05, hspace=0.5)
# full screen
mng = plt.get_current_fig_manager()
mng.full_screen_toggle()
plt.show()
