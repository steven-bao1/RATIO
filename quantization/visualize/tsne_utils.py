import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def tsne_visualization(batch_list, output_dir, name, label_names, max_samples_per_batch=None):
    """
    Visualizes high-dimensional vectors using t-SNE.

    Parameters:
    - batch_list (list of numpy arrays): A list where each element is a numpy array representing 
      a batch of high-dimensional vectors. Each batch will be marked with a different color.
    """
    sampled_batches = []
    labels = []

    np.random.seed(0)

    # Randomly sample from each batch if max_samples_per_batch is specified
    for i, batch in enumerate(batch_list):
        if max_samples_per_batch is not None:
            # Randomly sample from the current batch
            sampled_batch = batch[np.random.choice(batch.shape[0], min(max_samples_per_batch, batch.shape[0]), replace=False)]
        else:
            sampled_batch = batch
        sampled_batches.append(sampled_batch)
        labels.extend([i] * len(sampled_batch))

    # Concatenate all sampled batches into a single matrix
    all_vectors = np.concatenate(sampled_batches, axis=0)
    labels = np.array(labels)

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(all_vectors)
    
    # Create a scatter plot of the 2D projection
    plt.figure(dpi=300)  # High DPI for better resolution

    # Define a colorblind-friendly colormap and assign different colors to each batch
    unique_labels = np.unique(labels)
    custom_colors = ['#FF7F7F', '#ADD8E6', '#FFD580', '#90EE90']  # Hex codes for light red, light blue, light orange, light green
    cmap = ListedColormap(custom_colors)  # Create a custom colormap
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', 'x', '+', 'X']  # Different markers
    fontsize = 20
    label_fontsize = 25

    # Plot each batch with a different color and marker
    for i, label in enumerate(unique_labels):
        batch_data = reduced_data[labels == label]
        plt.scatter(
            batch_data[:, 0], batch_data[:, 1], 
            label=f'{label_names[i]}', 
            color=cmap(i / len(unique_labels)),  # Normalize color mapping
            marker=markers[i % len(markers)],  # Cycle through markers
            s=100,  # Adjust marker size
            edgecolor='k',  # Add black border to markers
            linewidth=0.5,  # Border thickness
            zorder=1000*(len(unique_labels) - i)
        )

    # Add labels and title with larger font sizes
    plt.xlabel('t-SNE Component 1', fontsize=label_fontsize, fontweight='bold')
    plt.ylabel('t-SNE Component 2', fontsize=label_fontsize, fontweight='bold')
    # plt.title('2D t-SNE Projection of Data', fontsize=16, fontweight='bold')

    # Add gridlines for better readability
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tick_params(axis="x", labelsize=fontsize-2)
    plt.tick_params(axis="y", labelsize=fontsize-2)

    # Adjust legend placement and style
    handles, labels = plt.gca().get_legend_handles_labels()
    # plt.legend(
    #     # title='Batch Labels', 
    #     # title_fontsize=12, 
    #     fontsize=fontsize-2,
    #     loc='lower center', 
    #     bbox_to_anchor=(0.5, 1.01),  # Move legend outside the plot
    #     ncol=len(unique_labels),
    #     frameon=True, 
    #     framealpha=1.0, 
    #     edgecolor='k'
    # )

    # Save the figure in high quality
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, f"{name}.pdf"), 
        format='pdf', 
        bbox_inches='tight',  # Ensure no part of the plot is cut off
        dpi=300
    )
    plt.close()

    return handles, labels


