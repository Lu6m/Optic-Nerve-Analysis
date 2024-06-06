import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu, gaussian, laplace
from skimage.feature import canny
from skimage.morphology import opening, closing, square, white_tophat, disk
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import os

# Créez le dossier 'Results' s'il n'existe pas déjà
results_path = 'Results_Chloe'
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Créez un sous-dossier spécifique pour les résultats de cette exécution
result_folder = os.path.join(results_path, 'Run_Results')
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

# Vérifier si le fichier image existe dans le répertoire courant
image_path = os.path.join('Lame_criblee', 'LC001.jpg')  # Remplacez par le chemin de votre image
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

# Fonction pour sauvegarder les images
def save_image(image, filename):
    filepath = os.path.join(result_folder, filename)
    plt.imsave(filepath, image, cmap='gray')

# Exemple de flux de travail
original_image = image
manual_contrast_image = manual_contrast(image, 1.5)
equalized_image = histogram_equalization(image)
tophat_image = top_hat(image)
binary_otsu = otsu_threshold(equalized_image)
edges = hysteresis_threshold(equalized_image)
opened_image = apply_opening(binary_otsu)
closed_image = apply_closing(opened_image)
filtered_image = sequential_filtering(binary_otsu)
log_image = laplacian_of_gaussian(equalized_image)

# Sauvegarder les images traitées
save_image(original_image, 'original_image.png')
save_image(manual_contrast_image, 'manual_contrast_image.png')
save_image(equalized_image, 'equalized_image.png')
save_image(tophat_image, 'tophat_image.png')
save_image(binary_otsu, 'binary_otsu.png')
save_image(edges, 'edges.png')
save_image(opened_image, 'opened_image.png')
save_image(closed_image, 'closed_image.png')
save_image(filtered_image, 'filtered_image.png')
save_image(log_image, 'log_image.png')

# Comparaison des images traitées avec l'image originale
def plot_comparison(original, processed, title_processed, filename):
    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(original, cmap='gray'), plt.title('Original Image')
    plt.subplot(122), plt.imshow(processed, cmap='gray'), plt.title(title_processed)
    comparison_path = os.path.join(result_folder, filename)
    plt.savefig(comparison_path)
    plt.close()

# Comparer et sauvegarder les résultats
plot_comparison(original_image, manual_contrast_image, 'Manual Contrast', 'comparison_manual_contrast.png')
plot_comparison(original_image, equalized_image, 'Histogram Equalization', 'comparison_equalized_image.png')
plot_comparison(original_image, tophat_image, 'Top Hat', 'comparison_tophat_image.png')
plot_comparison(original_image, binary_otsu, 'Otsu Threshold', 'comparison_binary_otsu.png')
plot_comparison(original_image, edges, 'Hysteresis Threshold', 'comparison_edges.png')
plot_comparison(original_image, opened_image, 'Opening', 'comparison_opened_image.png')
plot_comparison(original_image, closed_image, 'Closing', 'comparison_closed_image.png')
plot_comparison(original_image, filtered_image, 'Sequential Filtering', 'comparison_filtered_image.png')
plot_comparison(original_image, log_image, 'Laplacian of Gaussian', 'comparison_log_image.png')

# Additional analysis and results
regions = regionprops(label(filtered_image))
areas = [region.area for region in regions]

# Display and save histogram of pore areas
plt.figure(figsize=(8, 6))
plt.hist(areas, bins=40, color='blue')
plt.title('Distribution of Pore Areas')
plt.xlabel('Area')
plt.ylabel('Count')
histogram_path = os.path.join(result_folder, 'histogram.png')
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
stats_path = os.path.join(result_folder, 'statistics.txt')
with open(stats_path, 'w') as f:
    f.write(f"Mean Area: {mean_area:.2f}\n")
    f.write(f"Median Area: {median_area:.2f}\n")
    f.write(f"Standard Deviation of Area: {std_area:.2f}\n")
