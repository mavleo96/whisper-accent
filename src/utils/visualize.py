import numpy as np
import seaborn as sns
import umap
from adjustText import adjust_text


def visualize_embedding_similarity(embeddings, labels, title, ax):
    # Cosine similarity
    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    cosine_sim = normalized @ normalized.T

    sns.heatmap(
        cosine_sim,
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis",
        cbar_kws={"label": "Cosine Similarity"},
        ax=ax,
    )

    # Set tick labels and title
    ax.tick_params(axis="x", rotation=90)
    ax.tick_params(axis="y", rotation=0)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title(title, fontweight="medium")


def visualize_umap_embeddings(embeddings, labels, title, ax):
    # Fit UMAP model and get 2D embeddings
    reducer = umap.UMAP(
        n_neighbors=5,
        n_components=2,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        n_epochs=500,
        n_jobs=1,
    )
    emb_2d = reducer.fit_transform(embeddings)
    x, y = emb_2d[:, 0], emb_2d[:, 1]

    sns.scatterplot(x=x, y=y, color="teal", edgecolor="black", linewidth=0.8, ax=ax)

    # Set text labels and adjust text
    texts = [ax.text(xi, yi, label, fontsize=9) for xi, yi, label in zip(x, y, labels, strict=True)]
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    # Set title, labels, and grid
    ax.set_title(title, fontweight="medium")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(alpha=0.3, linestyle="--")


__all__ = ["visualize_embedding_similarity", "visualize_umap_embeddings"]
