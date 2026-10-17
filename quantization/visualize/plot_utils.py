import os

import matplotlib
matplotlib.use('agg')
import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt
import numpy as np


def plot_activations(acts, num_tokens, name, output_path):
    # set up the figure and axes
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')

    # plot activation
    acts = acts[:num_tokens, :]
    n_tokens, hidden_dim = acts.shape
    x = np.arange(1, hidden_dim + 1)
    y = np.arange(1, n_tokens + 1)
    X, Y = np.meshgrid(x, y)
    surf = ax1.plot_surface(X, Y, acts, cmap='coolwarm',
                          rstride=1, cstride=1, linewidth=0.5, 
                          antialiased=True, zorder=1)
    # fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)

    # ax1.set_title(f'Layer {name.split(".")[-1]}')
    ax1.set_xlabel('Channel')
    ax1.set_ylabel('Token')
    # ax1.set_zlabel('Absolute Value')

    os.makedirs(output_path, exist_ok=True)
    jpg_path = os.path.join(output_path, f'{name}.jpg')
    plt.savefig(jpg_path, bbox_inches='tight', pad_inches=0, dpi=800)
    plt.close()

    # convert jpg to pdf
    import img2pdf
    pdf_path = os.path.join(output_path, f'{name}.pdf')
    with open(pdf_path, "wb") as f:
        f.write(img2pdf.convert(jpg_path))
    os.remove(jpg_path)


def plot_attn_map(attn, name, output_path):
    os.makedirs(output_path, exist_ok=True)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_facecolor("grey")

    attn_mask = np.triu(np.ones_like(attn), k=1)
    attn[attn_mask == 1] = np.nan
    im = ax1.imshow(attn, cmap='coolwarm')
    plt.colorbar(im)

    plt.savefig(os.path.join(output_path, f'{name}.pdf'), bbox_inches='tight', pad_inches=0.0)
    plt.close()


def plot_massive_activations_stats(massive_act_stats_list, output_path):
    os.makedirs(output_path, exist_ok=True)

    colors = ["cornflowerblue", "mediumseagreen", "C4", "teal",  "dimgrey"]
    markers = ["o", "*", "^", "x", "d"]

    num_layers = len(massive_act_stats_list)
    stat_names = massive_act_stats_list[0].keys()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.grid(color='0.75')
    for i, stat_name in enumerate(stat_names):
        y = [massive_act_stats_list[layer][stat_name] for layer in range(num_layers)]
        ax1.plot(np.arange(num_layers), y, marker=markers[i], markerfacecolor='none', markersize=5,
                linewidth=2, color=colors[i], linestyle="-", label=stat_name)
    xtick_label = [1, num_layers//4, num_layers//2, num_layers*3//4, num_layers]
    ax1.set_xticks(xtick_label, xtick_label, fontsize=16)
    ax1.set_xlabel('Blocks', fontsize=18, labelpad=0.8)
    ax1.set_ylabel('Values', fontsize=18)
    ax1.tick_params(axis='x', which='major', pad=1.0)
    ax1.tick_params(axis='y', which='major', pad=0.4)
    ax1.legend(loc="upper right")

    plt.savefig(os.path.join(output_path, f'stats.pdf'), bbox_inches='tight', pad_inches=0, dpi=800)
    plt.close()


def plot_flatness(output_dir, name, vectors, vector_names):
    colors = ['#FF7F7F', '#ADD8E6', '#FFD580', '#90EE90']  # Hex codes for light red, light blue, light orange, light green
    fontsize = 20
    label_fontsize = 25
    linewidth = 4
    step_cnt = 100
    fig, ax1 = plt.subplots(1, 1)

    # plot distribution envelope
    for i in range(len(vectors)):
        x = np.linspace(0, len(vectors[i]) - 1, step_cnt)
        y = np.interp(x, range(len(vectors[i])), vectors[i])
        ax1.plot(x, y, color=colors[i], linewidth=linewidth, zorder=1000*(len(vectors) - i), label=vector_names[i], alpha=0.5)
    
    ax1.set_ylabel("Magnitude", fontsize=label_fontsize, fontweight='bold')
    ax1.set_xlabel("Channels", fontsize=label_fontsize, fontweight='bold')
    ax1.grid(axis='x', linestyle='--', alpha=0.6)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    ax1.tick_params(axis="x", labelsize=fontsize-2)
    ax1.tick_params(axis="y", labelsize=fontsize-2)
    # ax1.legend(loc="upper right", fontsize=fontsize-2)
    ax1.set_ylim(bottom=0)

    handles, labels = plt.gca().get_legend_handles_labels()

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, f"{name}.pdf"), 
        format='pdf', 
        dpi=300
    )
    plt.close()

    return handles, labels


def save_legend_as_pdf(handles, labels, output_dir, filename="legend.pdf"):
    """
    Save the legend as a separate PDF file.
    """
    # Create a new figure for the legend
    fig_legend = plt.figure(figsize=(8, 0.5))  # Adjust size as needed
    ax_legend = fig_legend.add_subplot(111)

    # Add the legend to the new figure
    ax_legend.legend(handles, labels, loc='center', ncol=len(labels), fontsize=18)

    # Hide axes
    ax_legend.axis('off')

    # Save the legend as a PDF
    fig_legend.savefig(
        os.path.join(output_dir, filename), 
        format='pdf', 
        bbox_inches='tight', 
        dpi=300
    )
    plt.close(fig_legend)
