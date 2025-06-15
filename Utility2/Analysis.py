import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_forgery(original_image_path, forged_image_path):
    # Read the original and forged images
    original = cv2.imread(original_image_path)
    forged = cv2.imread(forged_image_path)

    if original is None or forged is None:
        raise FileNotFoundError("Check if both original and forged image paths are correct.")

    # Convert images to grayscale
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    forged_gray = cv2.cvtColor(forged, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference between images
    diff = cv2.absdiff(original_gray, forged_gray)

    # Threshold the difference to get a binary image of the altered region
    _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    """
    # Display images using OpenCV
    cv2.imshow("Original Image", original)
    cv2.imshow("Forged Image", forged)
    cv2.imshow("Detected Difference", diff)
    cv2.imshow("Binary Forgery Mask", mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    # Define the output directory relative to the script location
    output_dir = os.path.join(os.path.dirname(__file__), "Outputs")

    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the output images
    cv2.imwrite(os.path.join(output_dir, "original_image.png"), original)
    cv2.imwrite(os.path.join(output_dir, "forged_image.png"), forged)
    cv2.imwrite(os.path.join(output_dir, "detected_difference.png"), diff)
    cv2.imwrite(os.path.join(output_dir, "binary_forgery_mask.png"), mask)

    print(f"Output images saved in '{output_dir}' folder.")

    
    # OR if you prefer matplotlib:

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(forged, cv2.COLOR_BGR2RGB))
    plt.title("Forged Image")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(mask, cmap='gray')
    plt.title("Forgery Detected")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    

# Example usage
if __name__ == "__main__":
    original_path ="Inputs/original.png"
    forged_path = "Inputs/forged.png"
    analyze_forgery(original_path, forged_path)
