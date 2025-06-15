import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def calculate_hu_moments(image):
    """
    Calculate log-scaled Hu Moments from a grayscale image.
    """
    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    for i in range(0, 7):
        if hu_moments[i] != 0:
            hu_moments[i] = -1 * np.sign(hu_moments[i]) * np.log10(abs(hu_moments[i]))
    
    return hu_moments

def get_threshold_distribution(images):
    """
    Compute Hu moments for a list of grayscale images.
    """
    hu_distributions = []
    for img in images:
        hu = calculate_hu_moments(img)
        hu_distributions.append(hu)
    
    return np.array(hu_distributions)

if __name__ == "__main__":
    # Define paths
    image_paths = ["Inputs/original.png", "Inputs/forged.png"]
    output_dir = os.path.join(os.path.dirname(__file__), "Outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Load grayscale images
    grayscale_images = []
    image_titles = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            grayscale_images.append(img)
            image_titles.append(os.path.basename(path))
        else:
            print(f"Failed to load image: {path}")

    # Calculate and print Hu Moments
    if grayscale_images:
        distribution = get_threshold_distribution(grayscale_images)
        print("Log-scaled Hu Moments Distribution:")
        for i, hu in enumerate(distribution):
            print(f"{image_titles[i]}: {hu}")

        # Show and save images with matplotlib
        plt.figure(figsize=(10, 4))
        for i in range(len(grayscale_images)):
            plt.subplot(1, len(grayscale_images), i+1)
            plt.imshow(grayscale_images[i], cmap='gray')
            plt.title(image_titles[i])
            plt.axis('off')
        
        plt.suptitle("Grayscale Input Images for Hu Moments")
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        # Save figure
        save_path = os.path.join(output_dir, "hu_moments_images.png")
        plt.savefig(save_path)
        plt.show()
        print(f"\nComparison image saved at: {save_path}")
