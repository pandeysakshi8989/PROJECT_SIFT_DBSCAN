import cv2
import numpy as np
import os

from BlockBuilder import build_blocks
from Approximate_Region import approximate_region
from HuThresholdDistribution import calculate_hu_moments
from Distance import distance_matrix as compute_distance_matrix
from KeypointCluster import cluster_keypoints
from SimilarityMeasure import DistanceCalculator
from Metrics import calculate_metrics

# ========== STEP 0: SETUP ==========
image_path = "Inputs/forged.png"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"[ERROR] Image not found at path: {image_path}")
print(f"[INFO] Loaded image from: {image_path}")

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# STEP 1: BLOCK-BASED CMFD PREPROCESSING (OPTIONAL/ANALYSIS)
blocks, raw_positions = build_blocks(image_gray, block_size=8, step=4)

# Convert numpy array positions (e.g., [x y]) to tuple (x, y)
positions = [tuple(pos) for pos in raw_positions]

print(f"[INFO] Blocks generated: {len(blocks)}")

# ========== STEP 2: SIFT Keypoints ==========
sift = cv2.SIFT_create()
kp, desc = sift.detectAndCompute(image_gray, None)
print(f"[INFO] Keypoints extracted: {len(kp)}")

# ========== STEP 3: REGION APPROXIMATION ==========
result_img = approximate_region(image, kp, radius=15)
cv2.imwrite("Outputs/approx_region_result.png", result_img)
print("[INFO] Region approximation image saved to Outputs/approx_region_result.png")

# ========== STEP 4: HU MOMENTS ==========
kp_mask = np.zeros_like(image_gray)
for point in kp:
    x, y = int(point.pt[0]), int(point.pt[1])
    cv2.circle(kp_mask, (x, y), radius=3, color=255, thickness=-1)

hu_features = calculate_hu_moments(kp_mask)
print("[INFO] Hu Moments extracted:", hu_features.shape)

# ========== STEP 5: DISTANCE MATRIX ==========
dist_matrix = compute_distance_matrix(hu_features)

# ========== STEP 6: KEYPOINT CLUSTERING ==========
clusters = cluster_keypoints(dist_matrix)
num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
print(f"[INFO] Found {num_clusters} clusters.")

# Ensure clusters are passed as Python list of ints
clusters = [int(c) for c in clusters]

# ========== STEP 7: SIMILARITY MEASURE ==========
# DistanceCalculation needs image, blocks, and a dict with cluster-wise positions
# Create a dictionary: cluster_id -> list of positions
cluster_positions = {}
for idx, cluster_id in enumerate(clusters):
    if cluster_id == -1:
        continue  # Skip noise
    if cluster_id not in cluster_positions:
        cluster_positions[cluster_id] = []
    cluster_positions[cluster_id].append(positions[idx])

DC = DistanceCalculator(image, blocks, cluster_positions)
similarity_score = DC.Calculate("HuMoments", desc)
print(f"[INFO] Similarity Score: {similarity_score:.4f}")

# ========== STEP 8: METRICS ==========

# Add dummy labels for now (if real labels are not available)
true_labels = [1, 0, 1, 1, 0, 1, 0]
predicted_labels = [1, 0, 1, 0, 0, 1, 1]

metrics = calculate_metrics(true_labels, predicted_labels)

for key, value in metrics.items():
    print(f"[INFO] {key}: {value}")
