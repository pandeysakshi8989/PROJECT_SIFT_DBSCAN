import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def build_blocks(image, block_size=8, step=1):
    """
    Splits an image into overlapping blocks.

    Returns:
        blocks: List of flattened block arrays
        positions: List of top-left coordinates for each block
    """
    h, w = image.shape[:2]
    blocks = []
    positions = []

    for y in range(0, h - block_size + 1, step):
        for x in range(0, w - block_size + 1, step):
            block = image[y:y + block_size, x:x + block_size]
            blocks.append(block.flatten())
            positions.append((x, y))

    return np.array(blocks), np.array(positions)

if __name__ == "__main__":
    # === Load image ===
    image_path = "Inputs/original.png"
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"[ERROR] Image not found at: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # === Generate blocks ===
    blocks, positions = build_blocks(gray, block_size=8, step=4)
    print(f"[INFO] Extracted {len(blocks)} blocks.")
    print("[INFO] First block position:", positions[0] if len(positions) > 0 else "No blocks")

    # === Create visualization ===
    num_blocks_to_show = min(10, len(blocks))
    if num_blocks_to_show == 0:
        print("[WARNING] No blocks to display.")
    else:
        fig, axs = plt.subplots(2, 5, figsize=(10, 5))
        fig.suptitle("First 10 Extracted Blocks")

        for i in range(10):
            if i < num_blocks_to_show:
                block_img = blocks[i].reshape(8, 8)
                axs[i // 5, i % 5].imshow(block_img, cmap='gray')
            axs[i // 5, i % 5].axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()

        # === Save visualization ===
        output_dir = os.path.join(os.path.dirname(__file__), "Outputs")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "sample_blocks.png")
        fig.savefig(output_path)
        print(f"[INFO] Sample block visualization saved at: {output_path}")
