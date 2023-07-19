import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import label
from skimage.measure import regionprops
from skimage.morphology import binary_erosion, square
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def preprocess_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

def convert_to_grayscale(image):
    # Convert the RGB image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return grayscale_image

def apply_threshold(image_gray, threshold_value):
    # Apply thresholding to segment the image
    _, binary_image = cv2.threshold(image_gray, threshold_value, 255, cv2.THRESH_BINARY)
    return binary_image

def calculate_fractal_dimension(binary_image):
    # Calculate fractal dimension using box-counting method
    num_scales = 80 #more = more accurate but slower
    box_counts = []

    for scale in range(1, num_scales + 1):
        eroded_image = binary_erosion(binary_image, footprint=square(scale))
        labeled_array, num_features = label(eroded_image)
        box_counts.append(num_features)

    # Remove zero values from box_counts
    box_counts = np.array(box_counts)
    non_zero_indices = np.nonzero(box_counts)
    scales = np.arange(1, num_scales + 1)[non_zero_indices]
    non_zero_box_counts = box_counts[non_zero_indices]

    if len(scales) == 0:
        # If all box_counts are zero, return NaN for the fractal dimension
        fractal_dimension = np.nan
    else:
        # Fit a line to the log-log plot of box counts vs. scale
        coeffs = np.polyfit(np.log(scales), np.log(non_zero_box_counts), 1)
        fractal_dimension = coeffs[0]

    return fractal_dimension

def analyze_regions(binary_image):
    # Perform region analysis on the binary image
    labeled_image, num_regions = label(binary_image)

    # Calculate area distribution of the segmented regions
    region_areas = []
    for region in regionprops(labeled_image):
        region_areas.append(region.area)

    return region_areas, num_regions

def plot_area_distribution(region_areas):
    # Plot the area distribution of segmented regions
    plt.figure(figsize=(8, 6))
    counts, bins, _ = plt.hist(region_areas, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel("Region Area")
    plt.ylabel("Number of Regions")
    plt.title("Area Distribution of Segmented Regions")
    plt.grid(True)

    # Add values to the nonzero bars
    for count, bin_val in zip(counts, bins):
        if count > 0:
            plt.text(bin_val, count, str(int(count)), ha='center', va='bottom')

    plt.show()

def analyze_connectivity(binary_image):
    # Perform connectivity analysis on the binary image
    labeled_image, num_regions = label(binary_image)

    # Create a connectivity matrix
    connectivity_matrix = np.zeros((num_regions, num_regions), dtype=int)

    for region in regionprops(labeled_image):
        min_row, min_col, max_row, max_col = region.bbox
        for row in range(min_row, max_row):
            for col in range(min_col, max_col):
                if labeled_image[row, col] > 0:
                    neighbors = labeled_image[max(0, row - 1):min(row + 2, labeled_image.shape[0]),
                                              max(0, col - 1):min(col + 2, labeled_image.shape[1])]
                    unique_neighbors = np.unique(neighbors[neighbors > 0])
                    if len(unique_neighbors) > 1:
                        for i in range(len(unique_neighbors)):
                            for j in range(i + 1, len(unique_neighbors)):
                                connectivity_matrix[unique_neighbors[i] - 1, unique_neighbors[j] - 1] += 1
                                connectivity_matrix[unique_neighbors[j] - 1, unique_neighbors[i] - 1] += 1

    return connectivity_matrix

def plot_connectivity_matrix(connectivity_matrix):
    # Plot the connectivity matrix of segmented regions
    plt.figure(figsize=(8, 8))
    plt.imshow(connectivity_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label="Number of Connections")
    plt.xlabel("Region Index")
    plt.ylabel("Region Index")
    plt.title("Connectivity Matrix of Segmented Regions")
    plt.show()

def analyze_cluster_size(region_areas):
    # Calculate cluster size distribution of the segmented regions
    cluster_sizes = {}
    for area in region_areas:
        cluster_sizes[area] = cluster_sizes.get(area, 0) + 1

    return cluster_sizes

def plot_cluster_size_distribution(cluster_sizes):
    # Plot the cluster size distribution of segmented regions
    plt.figure(figsize=(8, 6))
    plt.xscale('log')  # Set the x-axis to be logarithmic
    bars = plt.bar(cluster_sizes.keys(), cluster_sizes.values(), edgecolor='black', alpha=0.7)
    plt.xlabel("Cluster Size (Log Scale)")
    plt.ylabel("Number of Clusters")
    plt.title("Cluster Size Distribution of Segmented Regions")
    plt.grid(True)

    # Add values to the nonzero bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width() / 2, height, str(int(height)), ha='center', va='bottom')

    plt.show()

def plot_connected_clusters(binary_image_optimal):
    # Plot the connected clusters in the image
    labeled_image, num_regions = label(binary_image_optimal)
    plt.figure(figsize=(8, 6))
    plt.imshow(labeled_image, cmap='tab20', aspect='auto')
    plt.colorbar(label="Cluster Index")
    plt.xlabel("Column Index")
    plt.ylabel("Row Index")
    plt.title("Connected Clusters in the Image")
    plt.show()

def select_image():
    global file_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        image = Image.open(file_path)
        image = image.resize((400, 400), Image.ANTIALIAS)
        image_tk = ImageTk.PhotoImage(image)
        image_label.configure(image=image_tk)
        image_label.image = image_tk
        analyze_button["state"] = tk.NORMAL  # Enable the analyze button

def analyze_image():
    loading_label = tk.Label(frame, text="Analyzing Image, Please Wait...", font=("Arial", 12))
    loading_label.pack(pady=10)
    root.update()
    # Step 1: Preprocess the image
    image_path = file_path  # Get the selected image path
    image_rgb = preprocess_image(image_path)

    # Step 2: Convert to grayscale
    image_gray = convert_to_grayscale(image_rgb)

    # Step 3: Perform Scaling Analysis
    start_threshold = 10  # Starting threshold value
    end_threshold = 150   # Ending threshold value (inclusive)
    step_size = 5        # Step size between thresholds

    threshold_values = np.arange(start_threshold, end_threshold + 1, step_size)
    fractal_dimensions = []

    for threshold_value in threshold_values:
        binary_image = apply_threshold(image_gray, threshold_value)
        fractal_dimension = calculate_fractal_dimension(binary_image)
        fractal_dimensions.append(fractal_dimension)

    # Step 4: Plot Fractal Dimension vs. Threshold Value
    plt.figure(figsize=(10, 6))
    plt.plot(threshold_values, fractal_dimensions, marker='o', linestyle='-')
    plt.xlabel("Threshold Value")
    plt.ylabel("Fractal Dimension")
    plt.title("Fractal Dimension vs. Threshold Value")
    plt.grid(True)
    # Step 5: Get the threshold value with maximum fractal dimension
    max_fractal_dimension = max(fractal_dimensions)
    max_dimension_index = fractal_dimensions.index(max_fractal_dimension)
    optimal_threshold = threshold_values[max_dimension_index]
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', label='Optimal Threshold')
    plt.legend()
    plt.text(150, max(fractal_dimensions) - 0.1, f"Max Fractal Dimension: {max(fractal_dimensions):.2f}", color='blue')
    plt.scatter(optimal_threshold, max_fractal_dimension, color='red', label='Maximum Fractal Dimension', zorder=5)
    plt.text(optimal_threshold + 5, max_fractal_dimension - 0.1, f"Max Fractal Dimension: {max_fractal_dimension:.2f}", color='blue')

    plt.show()


    # Step 6: Generate the binary image with the optimal threshold
    binary_image_optimal = apply_threshold(image_gray, optimal_threshold)

    # Display the original image and binary image (optional)
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(binary_image_optimal, cmap='gray')
    plt.title(f"Binary Image (Optimal Threshold: {optimal_threshold})")
    plt.axis('off')
    plt.show()

    # Step 7: Perform Additional Analyses
    #region_areas, num_regions = analyze_regions(binary_image_optimal)

    # Step 8: Plot Area Distribution
    #plot_area_distribution(region_areas)

    # Step 9: Perform Connectivity Analysis
    #connectivity_matrix = analyze_connectivity(binary_image_optimal)

    # Step 10: Plot Connectivity Matrix
    #plot_connectivity_matrix(connectivity_matrix)

    # Step 11: Analyze Cluster Size Distribution
    #cluster_sizes = analyze_cluster_size(region_areas)

    # Step 12: Plot Cluster Size Distribution
    #plot_cluster_size_distribution(cluster_sizes)

    # Step 13: Plot Connected Clusters
    plot_connected_clusters(binary_image_optimal)
    loading_label.pack_forget()
# Create the main Tkinter window
root = tk.Tk()
root.title("Fractal Analyzer V1.0 GitHub.com/AbhinavM2000")

# Create a frame to hold the image label and button
frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

# Create a label to display the image
image_label = tk.Label(frame)
image_label.pack()

# Create a button to select the image
select_button = tk.Button(frame, text="Select Image", command=select_image)
select_button.pack(side=tk.LEFT, padx=5, pady=5)

# Create a button to analyze the image
analyze_button = tk.Button(frame, text="Analyze Image", command=analyze_image, state=tk.DISABLED)
analyze_button.pack(side=tk.RIGHT, padx=5, pady=5)

# Run the main Tkinter event loop
root.mainloop()
