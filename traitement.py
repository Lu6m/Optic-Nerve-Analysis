import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu, gaussian, laplace
from skimage.feature import canny
from skimage.morphology import opening, closing, square, white_tophat, disk
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import os

# Créez le dossier Results s'il n'existe pas déjà
results_path = 'Results'
if not os.path.exists(results_path):
    os.makedirs(results_path)

# répertoire de travail actuel
current_dir = os.getcwd()

# Vérifier si le fichier image existe dans le répertoire courant
image_path = os.path.join(current_dir, 'LC001.jpg')  # Remplacez par le chemin de votre image
if not os.path.isfile(image_path):
    raise FileNotFoundError(f"Image not found at path: {image_path}")

# Charger l'image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError(f"Image not found or cannot be opened at path: {image_path}")

# I. Introduction
# (Introduction text, no code needed here)

# II. Pré-traitement de l’image

# 1. Elimination des rectangles noirs
def remove_black_rectangles(image):
    image[image == 0] = 128
    return image

# 2. Augmentation du contraste
def manual_contrast(image, factor):
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def histogram_equalization(image):
    return cv2.equalizeHist(image)

def top_hat(image):
    selem = disk(12)
    return white_tophat(image, selem)

# III. Méthodes de segmentation
def otsu_threshold(image):
    thresh = threshold_otsu(image)
    return image > thresh

def hysteresis_threshold(image):
    return canny(image, low_threshold=10, high_threshold=50)

# IV. Amélioration de l’image
def apply_opening(image):
    return opening(image, square(3))

def apply_closing(image):
    return closing(image, square(3))

def sequential_filtering(image):
    return opening(closing(opening(closing(image, square(3)), square(3)), square(3)), square(3))

# V. Détection de contours
def laplacian_of_gaussian(image):
    return laplace(gaussian(image, sigma=3))

#  workflow
image = remove_black_rectangles(image)
manual_contrast_image = manual_contrast(image, 1.5)
equalized_image = histogram_equalization(image)
tophat_image = top_hat(image)
binary_otsu = otsu_threshold(equalized_image)
edges = hysteresis_threshold(equalized_image)
opened_image = apply_opening(binary_otsu)
closed_image = apply_closing(opened_image)
filtered_image = sequential_filtering(binary_otsu)
log_image = laplacian_of_gaussian(equalized_image)

# Display and save results
plt.figure(figsize=(12, 12))

plt.subplot(331), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(332), plt.imshow(manual_contrast_image, cmap='gray'), plt.title('Manual Contrast')
plt.subplot(333), plt.imshow(equalized_image, cmap='gray'), plt.title('Histogram Equalization')
plt.subplot(334), plt.imshow(tophat_image, cmap='gray'), plt.title('Top Hat')
plt.subplot(335), plt.imshow(binary_otsu, cmap='gray'), plt.title('Otsu Threshold')
plt.subplot(336), plt.imshow(edges, cmap='gray'), plt.title('Hysteresis Threshold')
plt.subplot(337), plt.imshow(opened_image, cmap='gray'), plt.title('Opening')
plt.subplot(338), plt.imshow(closed_image, cmap='gray'), plt.title('Closing')
plt.subplot(339), plt.imshow(filtered_image, cmap='gray'), plt.title('Sequential Filtering')

# Sauvegarder la figure résultante
result_image_path = os.path.join(results_path, 'result_image.png')
plt.savefig(result_image_path)

plt.show()

# Additional analysis and results
regions = regionprops(label(filtered_image))
areas = [region.area for region in regions]

# Display and save histogram of pore areas
plt.figure(figsize=(8, 6))
plt.hist(areas, bins=40, color='blue')
plt.title('Distribution of Pore Areas')
plt.xlabel('Area')
plt.ylabel('Count')

# Sauvegarder l'histogramme
histogram_path = os.path.join(results_path, 'histogram.png')
plt.savefig(histogram_path)

plt.show()

# Statistical Summary
mean_area = np.mean(areas)
median_area = np.median(areas)
std_area = np.std(areas)

print(f"Mean Area: {mean_area:.2f}")
print(f"Median Area: {median_area:.2f}")
print(f"Standard Deviation of Area: {std_area:.2f}")

# Sauvegarder les statistiques dans un fichier texte
stats_path = os.path.join(results_path, 'statistics.txt')
with open(stats_path, 'w') as f:
    f.write(f"Mean Area: {mean_area:.2f}\n")
    f.write(f"Median Area: {median_area:.2f}\n")
    f.write(f"Standard Deviation of Area: {std_area:.2f}\n")
