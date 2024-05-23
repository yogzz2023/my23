import numpy as np
from itertools import product
from scipy.stats import chi2

# Define the measurement and track parameters
state_dim = 3  # 3D state (e.g., x, y, z)

# Predefined tracks and reports in 3D
tracks = np.array([
    [6, 6, 10],
    [15, 15, 15],
    [20, 20, 20]
])

reports = np.array([
    [10, 10, 10],
    [17, 17, 10],
    [12, 12, 10],
    [18, 18, 10],
    [16, 16, 10],
    [50, 55, 50]
])

# Chi-squared gating threshold for 95% confidence interval
chi2_threshold = chi2.ppf(0.95, df=state_dim)
print(f"Chi-squared threshold: {chi2_threshold}")

def mahalanobis_distance(x, y, cov_inv):
    delta = x - y
    return np.sqrt(np.dot(np.dot(delta, cov_inv), delta))

# Covariance matrix of the measurement errors (assumed to be identity for simplicity)
cov_matrix = np.eye(state_dim)
cov_inv = np.linalg.inv(cov_matrix)

# Perform residual error check using Chi-squared gating
association_list = []
for i, track in enumerate(tracks):
    for j, report in enumerate(reports):
        distance = mahalanobis_distance(track, report, cov_inv)
        print(f"Distance between track {i} and report {j}: {distance}")
        if distance < (chi2_threshold):
            association_list.append((i, j))

# Print association list
print("Association List (Track Index, Report Index):")
for assoc in association_list:
    print(assoc)

# Clustering reports and tracks based on associations
clusters = []
while association_list:
    cluster_tracks = set()
    cluster_reports = set()
    stack = [association_list.pop(0)]
    while stack:
        track_idx, report_idx = stack.pop()
        cluster_tracks.add(track_idx)
        cluster_reports.add(report_idx)
        new_assoc = [(t, r) for t, r in association_list if t == track_idx or r == report_idx]
        for assoc in new_assoc:
            if assoc not in stack:
                stack.append(assoc)
        association_list = [assoc for assoc in association_list if assoc not in new_assoc]
    clusters.append((list(cluster_tracks), list(cluster_reports)))

# Print clusters
print("\nClusters (Tracks, Reports):")
for cluster in clusters:
    print(cluster)

# Hypothesis generation for each cluster
def generate_hypotheses(num_tracks, num_reports):
    # Include the possibility of missed detections (track has no corresponding report)
    report_indices = list(range(num_reports)) + [None]

    hypotheses = []
    for assignment in product(report_indices, repeat=num_tracks):
        # Ensure no report is assigned to more than one track
        if len(set([a for a in assignment if a is not None])) == len([a for a in assignment if a is not None]):
            hypotheses.append(assignment)

    return hypotheses

# Calculate probabilities for each hypothesis
def calculate_probabilities(hypotheses, tracks, reports, cov_inv):
    probabilities = []
    for hypothesis in hypotheses:
        prob = 1.0
        for track_idx, report_idx in enumerate(hypothesis):
            if report_idx is not None:
                distance = mahalanobis_distance(tracks[track_idx], reports[report_idx], cov_inv)
                prob *= np.exp(-0.5 * distance**2)
            else:
                prob *= 0.01  # Probability of a missed detection or false alarm
        probabilities.append(prob)
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()  # Normalize
    return probabilities

# Process each cluster and generate hypotheses
for track_idxs, report_idxs in clusters:
    cluster_tracks = tracks[track_idxs]
    cluster_reports = reports[report_idxs]
    num_tracks = len(cluster_tracks)
    num_reports = len(cluster_reports)
    hypotheses = generate_hypotheses(num_tracks, num_reports)
    probabilities = calculate_probabilities(hypotheses, cluster_tracks, cluster_reports, cov_inv)
    
    print("\nCluster Hypotheses and Probabilities:")
    for hypothesis, probability in zip(hypotheses, probabilities):
        # Represent hypothesis with track and report indices
        hypothesis_representation = [(track_idxs[track_idx], report_idxs[report_idx] if report_idx is not None else None) for track_idx, report_idx in enumerate(hypothesis)]
        print(f"Hypothesis: {hypothesis_representation}, Probability: {probability:.4f}")
