import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2
from itertools import product

# Define functions for chi-squared gating, hypothesis generation, and probability calculation

def mahalanobis_distance(track, report, cov_inv):
    delta = track - np.array([report[0], report[1], report[2], 0, 0, 0])
    return np.sqrt(np.dot(np.dot(delta, cov_inv), delta))


def generate_hypotheses(num_tracks, num_reports):
    report_indices = list(range(num_reports)) + [None]
    hypotheses = []
    for assignment in product(report_indices, repeat=num_tracks):
        if len(set([a for a in assignment if a is not None])) == len([a for a in assignment if a is not None]):
            hypotheses.append(assignment)
    return hypotheses

def calculate_probabilities(hypotheses, tracks, measurements, cov_inv):
    probabilities = []
    for hypothesis in hypotheses:
        prob = 1.0
        for track_idx, report_idx in enumerate(hypothesis):
            if report_idx is not None:
                if 0 <= report_idx < len(measurements):
                    report = measurements[report_idx][:3]  # Extract (x, y, z) from the measurement
                    distance = mahalanobis_distance(tracks[track_idx], report, cov_inv)
                    prob *= np.exp(-0.5 * distance**2)
                else:
                    # If report_idx is out of bounds, assume a small probability
                    prob *= 0.01  # Probability of a missed detection or false alarm
            else:
                # If report_idx is None, assume a small probability
                prob *= 0.01  # Probability of a missed detection or false alarm
        probabilities.append(prob)
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()  # Normalize
    return probabilities




class CVFilter:
    def __init__(self):
        # Initialize filter parameters
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.pf = np.eye(6)  # Filter state covariance matrix
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        # Initialize filter state
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time

    def predict_step(self, current_time):
        # Predict step
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sf = np.dot(Phi, self.Sf)
        self.pf = np.dot(np.dot(Phi, self.pf), Phi.T) + Q

    def update_step(self, measurements, chi2_threshold):
        # Update step with JPDA
        association_list = []
        for i, track in enumerate(self.Sf):
            for j, report in enumerate(measurements):
                distance = mahalanobis_distance(track, report, np.linalg.inv(self.pf))
                if distance < np.sqrt(chi2_threshold):
                    association_list.append((i, j))

        # Clustering
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

        # Hypothesis generation and probability calculation
        hypotheses_probabilities = []
        for cluster in clusters:
            cluster_tracks, cluster_reports = cluster
            num_tracks = len(cluster_tracks)
            num_reports = len(cluster_reports)
            hypotheses = generate_hypotheses(num_tracks, num_reports)
            probabilities = calculate_probabilities(hypotheses, cluster_tracks, cluster_reports, np.linalg.inv(self.pf))
            hypotheses_probabilities.append((hypotheses, probabilities))

        # Perform update for each hypothesis
        for hypotheses, probabilities in hypotheses_probabilities:
            max_prob_index = np.argmax(probabilities)
            max_prob_hypothesis = hypotheses[max_prob_index]
            max_prob = probabilities[max_prob_index]

            for track_idx, report_idx in enumerate(max_prob_hypothesis):
                if report_idx is not None:
                    measurement = measurements[report_idx]
                    Inn = np.array(measurement) - np.dot(self.H, self.Sf)
                    S = np.dot(self.H, np.dot(self.pf, self.H.T)) + self.R
                    K = np.dot(np.dot(self.pf, self.H.T), np.linalg.inv(S))
                    self.Sf = self.Sf + np.dot(K, Inn)
                    self.pf = np.dot(np.eye(6) - np.dot(K, self.H), self.pf)

# Function to convert spherical coordinates to Cartesian coordinates
def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

# Function to convert Cartesian coordinates to spherical coordinates
def cart2sph(x, y, z):
    r = math.sqrt(x**2 + y**2 + z**2)
    el = math.degrees(math.atan2(z, math.sqrt(x**2 + y**2)))
    az = math.degrees(math.atan2(y, x))

    if x > 0.0:
        az = 90 - az
    else:
        az = 270 - az

    return r, az, el

# Function to read measurements from CSV file
def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            # Adjust column indices based on CSV file structure
            mr = float(row[10])  # MR column
            ma = float(row[11])  # MA column
            me = float(row[12])  # ME column
            mt = float(row[13])  # MT column
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            measurements.append((x, y, z, mt))
    return measurements

# Create an instance of the CVFilter class
kalman_filter = CVFilter()

# Define the path to your CSV file containing measurements
csv_file_path = 'data_test.csv'  #
# Read measurements from CSV file
measurements = read_measurements_from_csv(csv_file_path)

# Calculate Chi-squared threshold
state_dim = 6
chi2_threshold = chi2.ppf(0.95, df=state_dim)
print(f"Chi-squared threshold: {chi2_threshold}")

# Iterate through measurements
for i, (x, y, z, mt) in enumerate(measurements):
    if i == 0:
        # Initialize filter state with the first measurement
        kalman_filter.initialize_filter_state(x, y, z, 0, 0, 0, mt)
    elif i == 1:
        # Initialize filter state with the second measurement and compute velocity
        prev_x, prev_y, prev_z = measurements[i-1][:3]
        dt = mt - measurements[i-1][3]
        vx = (x - prev_x) / dt
        vy = (y - prev_y) / dt
        vz = (z - prev_z) / dt
        kalman_filter.initialize_filter_state(x, y, z, vx, vy, vz, mt)
    else:
        # Predict step
        kalman_filter.predict_step(mt)

        # Perform update step with JPDA
        kalman_filter.update_step(measurements[:i+1], chi2_threshold)

# Plot the results
# Plot range (r) vs. time
plt.figure(figsize=(12, 6))
plt.plot([m[3] for m in measurements], [math.sqrt(m[0]**2 + m[1]**2 + m[2]**2) for m in measurements], color='lime', linewidth=2, label='Measured Range')
plt.plot([m[3] for m in measurements], [math.sqrt(kalman_filter.Sf[0][0]**2 + kalman_filter.Sf[1][0]**2 + kalman_filter.Sf[2][0]**2) for m in measurements], color='red', linestyle='--', label='Filtered Range')
plt.xlabel('Time')
plt.ylabel('Range (r)')
plt.title('Range vs. Time')
plt.grid(True)
plt.legend()
plt.show()

# Plot azimuth (az) vs. time
plt.figure(figsize=(12, 6))
plt.plot([m[3] for m in measurements], [cart2sph(m[0], m[1], m[2])[1] for m in measurements], color='lime', linewidth=2, label='Measured Azimuth')
plt.plot([m[3] for m in measurements], [cart2sph(kalman_filter.Sf[0][0], kalman_filter.Sf[1][0], kalman_filter.Sf[2][0])[1] for m in measurements], color='red', linestyle='--', label='Filtered Azimuth')
plt.xlabel('Time')
plt.ylabel('Azimuth (az)')
plt.title('Azimuth vs. Time')
plt.grid(True)
plt.legend()
plt.show()

# Plot elevation (el) vs. time
plt.figure(figsize=(12, 6))
plt.plot([m[3] for m in measurements], [cart2sph(m[0], m[1], m[2])[2] for m in measurements], color='lime', linewidth=2, label='Measured Elevation')
plt.plot([m[3] for m in measurements], [cart2sph(kalman_filter.Sf[0][0], kalman_filter.Sf[1][0], kalman_filter.Sf[2][0])[2] for m in measurements], color='red', linestyle='--', label='Filtered Elevation')
plt.xlabel('Time')
plt.ylabel('Elevation (el)')
plt.title('Elevation vs. Time')
plt.grid(True)
plt.legend()
plt.show()
