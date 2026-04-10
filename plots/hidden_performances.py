import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from matplotlib.gridspec import GridSpec


sns.set_style(
    "whitegrid",
    rc={"axes.edgecolor": ".15", "xtick.bottom": True, "ytick.left": True},
)

base_path = "/home/diego/Documents/area_science/ricerca/open/hidden_pool_embeddings/results/performace_evals"
path = f"{base_path}/qwen3_base_0.6b_mteb_eng_v2_hidden_states_2gpu_200k_max_queries_results.json"
with open(f"{path}", "r") as f:
    data = json.load(f)


fig = plt.figure(figsize=(15, 6))
gs = GridSpec(2, 4)
for i, (task, dataset_list) in enumerate(data.items()):
    color = f"C{i}"
    ax = fig.add_subplot(gs[i])
    for j, item in enumerate(dataset_list):
        dataset_name = next(iter(item))
        layer_dict = next(iter(item.values()))[0]

        values = list(layer_dict.values())

        layer_index = [key.split(".")[-1] for key in layer_dict.keys()]
        x_ticks = list(range(len(values)))
        if j == 0:
            sns.lineplot(ax=ax, x=x_ticks, y=values, color=color, label=task, alpha=0.2)
        else:
            sns.lineplot(ax=ax, x=x_ticks, y=values, color=color, alpha=0.2)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(list(range(0, 2 * len(x_ticks), 2)))
    ax.legend()
gs.tight_layout(fig)


model_types = ["base", "embedding"]
model_types = ["encoder", "decoder"]

fig = plt.figure(figsize=(10, 4.5))
gs = GridSpec(1, 2)

for i, type in enumerate(model_types):
    ax = fig.add_subplot(gs[i])
    # path = f"qwen3_{type}_0.6b_mteb_eng_v2_hidden_states_2gpu_200k_max_queries_summary.json"

    path = f"t5gemma2_{type}_base_270m_mteb_eng_v2_hidden_states_2gpu_200k_max_queries_summary.json"
    with open(f"{base_path}/{path}", "r") as f:
        data = json.load(f)

    for i, (task_name, values) in enumerate(data.items()):
        color = f"C{i}"
        x_ticks = list(range(len(values)))
        sns.lineplot(
            ax=ax, x=x_ticks, y=values, color=color, label=task_name, marker="o"
        )
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(list(range(0, 2 * len(x_ticks), 2)))

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),  # move below the axes
        ncol=3,  # optional: spread items horizontally
        frameon=False,  # remove the box
        fontsize=8,
    )
gs.tight_layout(fig)


dayase
