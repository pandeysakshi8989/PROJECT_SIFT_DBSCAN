import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity

class DistanceCalculator:
    def __init__(self, positions, blocks, descriptors, delta=None, gamma=None):
        self.positions = positions
        self.blocks = blocks
        self.descriptors = descriptors
        self.delta = delta
        self.gamma = gamma

        # Detect the number of clusters safely
        if isinstance(blocks, dict):
            self.cluster_indices = blocks.keys()
        else:
            self.cluster_indices = range(len(blocks))

    def Calculate(self, method="Euclidean", desc=None):
        total_similarity = 0
        valid_clusters = 0

        print("[INFO] Calculating similarity using", method)

        for cluster_Pos in self.cluster_indices:
            try:
                cluster_blocks = self.blocks[cluster_Pos]
                cluster_positions = self.positions[cluster_Pos]
                cluster_descriptors = self.descriptors[cluster_Pos]
            except (KeyError, IndexError) as e:
                continue  # Skip if the cluster index is not valid

            if len(cluster_blocks) < 2 or len(cluster_positions) < 2:
                continue

            similarities = []

            for i in range(len(cluster_descriptors)):
                for j in range(i + 1, len(cluster_descriptors)):
                    desc1 = cluster_descriptors[i]
                    desc2 = cluster_descriptors[j]

                    if method == "Euclidean":
                        sim = euclidean(desc1, desc2)
                    elif method == "Cosine":
                        sim = cosine_similarity([desc1], [desc2])[0][0]
                    elif method == "HuMoments":
                        sim = np.linalg.norm(np.array(desc1) - np.array(desc2))
                    else:
                        raise ValueError("Unsupported similarity method")

                    similarities.append(sim)

            if similarities:
                avg_sim = sum(similarities) / len(similarities)
                total_similarity += avg_sim
                valid_clusters += 1

        if valid_clusters == 0:
            print("[WARNING] No valid clusters found for similarity comparison.")
            return None

        average_similarity = total_similarity / valid_clusters
        print(f"[INFO] Average similarity score: {average_similarity}")
        return average_similarity
