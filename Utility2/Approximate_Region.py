import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def approximate_region(image, keypoints, radius=15):
    """
    Draws circular regions around given keypoints in the image.

    Parameters:
    - image: input BGR image (numpy array)
    - keypoints: list of cv2.KeyPoint
    - radius: radius of circle to draw around keypoints

    Returns:
    - result_img: image with circles drawn
    """
    result_img = image.copy()

    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(result_img, (x, y), radius, (0, 0, 255), 2)

    return result_img

# Example usage
if __name__ == "__main__":
    # Load image
    image_path = "Inputs/original.png"
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError("Image not found. Please check the path:", image_path)

    # Detect keypoints using ORB
    orb = cv2.ORB_create()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints = orb.detect(gray, None)

    # Draw keypoint regions
    region_img = approximate_region(image, keypoints)

    # Create Outputs directory relative to script location
    output_dir = os.path.join(os.path.dirname(__file__), "Outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Save output image
    output_path = os.path.join(output_dir, "keypoint_regions.png")
    cv2.imwrite(output_path, region_img)
    print(f"Output saved to: {output_path}")

    # Show image using matplotlib
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB))
    plt.title("Keypoint Region Approximation")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
