import numpy as np
from scipy.stats import chi2

# Define the measurement and track parameters
state_dim = 3  # 3D state (e.g., x, y, z)

# Predefined tracks and reports in 3D
tracks = np.array([
    [10, 10, 10],
    [20, 20, 20],
    [30, 30, 30],
    [40, 40, 40],
    [50, 50, 50]
])

reports = np.array([
    [12, 10, 10],
    [18, 22, 20],
    [29, 32, 30],
    [41, 39, 40],
    [50, 55, 50],
    [60, 60, 60],
    [70, 70, 70],
    [80, 80, 80]
])

# Chi-squared gating threshold for 95% confidence interval
chi2_threshold = chi2.ppf(0.95, df=state_dim)

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
        if distance < np.sqrt(chi2_threshold):
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
def generate_hypotheses(tracks, reports):
    if tracks.size == 0 or reports.size == 0:
        return []

    hypotheses = []
    num_tracks = len(tracks)
    num_reports = len(reports)

    def is_valid_hypothesis(hypothesis):
        report_set = set()
        for report_idx in hypothesis:
            if report_idx in report_set:
                return False
            report_set.add(report_idx)
        return True

    for hypothesis in np.ndindex(*(num_reports + 1 for _ in range(num_tracks))):
        if is_valid_hypothesis(hypothesis):
            hypotheses.append(hypothesis)

    print("\nGenerated Hypotheses:")
    for hypothesis in hypotheses:
        print(hypothesis)

    return hypotheses

# Calculate probabilities for each hypothesis
def calculate_probabilities(hypotheses, tracks, reports):
    probabilities = []
    for hypothesis in hypotheses:
        prob = 1.0
        for track_idx, report_idx in enumerate(hypothesis):
            if report_idx < len(reports):
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
    hypotheses = generate_hypotheses(cluster_tracks, cluster_reports)
    probabilities = calculate_probabilities(hypotheses, cluster_tracks, cluster_reports)
    # print("\nCluster Hypotheses and Probabilities:")
    # for hypothesis, probability in zip(hypotheses, probabilities):
    #     print(f"Hypothesis: {hypothesis}, Probability: {probability:.4f}")
