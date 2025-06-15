import cv2
import numpy as np
from sklearn.cluster import KMeans

def extract_keypoints_and_descriptors(image, detector='ORB'):
    """
    Extracts keypoints and descriptors using a specified feature detector.

    Parameters:
    - image: Grayscale image (numpy array)
    - detector: Feature detector type - 'ORB', 'SIFT', or 'BRISK'

    Returns:
    - keypoints: List of cv2.KeyPoint objects
    - descriptors: numpy array of descriptors
    """
    if detector == 'ORB':
        extractor = cv2.ORB_create()
    elif detector == 'SIFT':
        extractor = cv2.SIFT_create()
    elif detector == 'BRISK':
        extractor = cv2.BRISK_create()
    else:
        raise ValueError("Unsupported detector. Use 'ORB', 'SIFT', or 'BRISK'.")

    keypoints, descriptors = extractor.detectAndCompute(image, None)
    return keypoints, descriptors


def cluster_keypoints(descriptors, num_clusters=5):
    """
    Cluster feature descriptors using KMeans.

    Parameters:
    - descriptors: numpy array of feature descriptors
    - num_clusters: Number of clusters to form

    Returns:
    - labels: Cluster labels for each descriptor
    """
    if descriptors is None or len(descriptors) < num_clusters:
        print("Not enough descriptors to form clusters.")
        return None

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(descriptors)
    return labels


# Example usage
if __name__ == "__main__":
    image_path = "Inputs/original.png"  # Replace with your actual image path
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: Unable to load image from path: {image_path}")
    else:
        keypoints, descriptors = extract_keypoints_and_descriptors(img, detector='ORB')
        print(f"Extracted {len(keypoints)} keypoints.")

        if descriptors is not None:
            cluster_labels = cluster_keypoints(descriptors, num_clusters=5)
            if cluster_labels is not None:
                print("Cluster labels:", cluster_labels)
