import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def draw_dataset(X, y, mask=None):
    if mask is None:
        mask = np.ones(X.shape[0], dtype=bool)
    y = np.float32(y)
    # 1. Define your class-to-color mapping
    color_map = {"0.0": "blue", "1.0": "red"}

    # Convert your categories into lists of colors for both groups
    classes_unmasked = [str(val) for val, m in zip(y[:, 0], mask) if not m]
    classes_masked = [str(val) for val, m in zip(y[:, 0], mask) if m]

    colors_unmasked = [
        color_map.get(c, "gray") for c in classes_unmasked
    ]  # 'gray' is a fallback
    colors_masked = [color_map.get(c, "gray") for c in classes_masked]

    # Scale features (important for t-SNE)
    X_scaled = StandardScaler().fit_transform(X)

    # Run t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=30,  # try 5–50; ~sqrt(n) is a good start
        learning_rate="auto",
        init="pca",
        random_state=42,
        # n_iter=1000
    )
    X_tsne = tsne.fit_transform(X_scaled)

    # 2. Setup the figure
    plt.figure(figsize=(8, 6))

    # 3. Plot background/unmasked items (Fade effect)
    plt.scatter(
        x=X_tsne[~mask, 0],
        y=X_tsne[~mask, 1],
        c=colors_unmasked,  # Colors represent the class
        marker="o",  # Circular marker
        s=10,
        alpha=0.3,  # Faded opacity
    )

    # 4. Plot foreground/masked items (Focus effect)
    plt.scatter(
        x=X_tsne[mask, 0],
        y=X_tsne[mask, 1],
        c=colors_masked,  # Colors represent the class
        marker="o",  # Distinct symbol shape
        s=25,
        alpha=1.0,  # Full opacity
    )

    # 5. Final layout
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("t-SNE Class Colors (Red/Blue) with Mask Opacity")
