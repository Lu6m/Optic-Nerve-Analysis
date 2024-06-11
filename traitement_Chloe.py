import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu, gaussian, laplace
from skimage.feature import canny
from skimage.morphology import opening, closing, square, white_tophat, disk
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, mean_squared_error
import os

# Créez le dossier 'Results' s'il n'existe pas déjà
results_path = 'Results_Chloe'
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Créez un sous-dossier spécifique pour les résultats de cette exécution
result_folder = os.path.join(results_path, 'Run_Results')
if not os.path.exists(result_folder):
    os.makedirs(result_folder)


# Charger l'image
image_path = 'Lame_criblee/LC001.jpg'  # Remplacez par le chemin de votre image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Appliquer le seuillage d'Otsu pour générer un masque binaire
thresh = threshold_otsu(image)
binary_mask = (image > thresh).astype(np.uint8) * 255

# Sauvegarder le masque binaire
mask_path = 'Terrain/LC001_VT.png'  # Remplacez par le chemin souhaité pour sauvegarder le masque
cv2.imwrite(mask_path, binary_mask)

# Afficher le masque généré
plt.imshow(binary_mask, cmap='gray')
plt.title('Generated Binary Mask')
plt.show()
if not os.path.isfile(image_path):
    raise FileNotFoundError(f"Image not found at path: {image_path}")
if not os.path.isfile(mask_path):
    raise FileNotFoundError(f"Mask not found at path: {mask_path}")

# Charger l'image et le masque de vérité terrain
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
ground_truth = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Image not found or cannot be opened at path: {image_path}")
if ground_truth is None:
    raise FileNotFoundError(f"Mask not found or cannot be opened at path: {mask_path}")

# Convertir le masque de vérité terrain en 0 et 1
ground_truth = (ground_truth > 128).astype(np.uint8)

# II. Pré-traitement de l’image

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

# Binarisation en utilisant Otsu
def binarize(image):
    thresh = threshold_otsu(image)
    binary = image > thresh
    return (binary * 255).astype(np.uint8)

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

# Générer un masque binaire à partir de l'image originale
mask_binary = binarize(original_image)

# Binarisation des images traitées
manual_contrast_binary = binarize(manual_contrast_image)
equalized_binary = binarize(equalized_image)
tophat_binary = binarize(tophat_image)
binary_otsu_binary = binarize(binary_otsu)
edges_binary = binarize(edges)
opened_binary = binarize(opened_image)
closed_binary = binarize(closed_image)
filtered_binary = binarize(filtered_image)
log_binary = binarize(log_image)

# Convertir les images binaires en 0 et 1
manual_contrast_binary = (manual_contrast_binary > 128).astype(np.uint8)
equalized_binary = (equalized_binary > 128).astype(np.uint8)
tophat_binary = (tophat_binary > 128).astype(np.uint8)
binary_otsu_binary = (binary_otsu_binary > 128).astype(np.uint8)
edges_binary = (edges_binary > 128).astype(np.uint8)
opened_binary = (opened_binary > 128).astype(np.uint8)
closed_binary = (closed_binary > 128).astype(np.uint8)
filtered_binary = (filtered_binary > 128).astype(np.uint8)
log_binary = (log_binary > 128).astype(np.uint8)

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

# Sauvegarder les images binaires
save_image(mask_binary, 'mask_binary.png')
save_image(manual_contrast_binary, 'manual_contrast_binary.png')
save_image(equalized_binary, 'equalized_binary.png')
save_image(tophat_binary, 'tophat_binary.png')
save_image(binary_otsu_binary, 'binary_otsu_binary.png')
save_image(edges_binary, 'edges_binary.png')
save_image(opened_binary, 'opened_binary.png')
save_image(closed_binary, 'closed_binary.png')
save_image(filtered_binary, 'filtered_binary.png')
save_image(log_binary, 'log_binary.png')

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

# Comparer et sauvegarder les images binaires avec le masque binaire généré
plot_comparison(mask_binary, manual_contrast_binary, 'Manual Contrast Binary', 'comparison_manual_contrast_binary.png')
plot_comparison(mask_binary, equalized_binary, 'Histogram Equalization Binary', 'comparison_equalized_binary.png')
plot_comparison(mask_binary, tophat_binary, 'Top Hat Binary', 'comparison_tophat_binary.png')
plot_comparison(mask_binary, binary_otsu_binary, 'Otsu Threshold Binary', 'comparison_binary_otsu_binary.png')
plot_comparison(mask_binary, edges_binary, 'Hysteresis Threshold Binary', 'comparison_edges_binary.png')
plot_comparison(mask_binary, opened_binary, 'Opening Binary', 'comparison_opened_binary.png')
plot_comparison(mask_binary, closed_binary, 'Closing Binary', 'comparison_closed_binary.png')
plot_comparison(mask_binary, filtered_binary, 'Sequential Filtering Binary', 'comparison_filtered_binary.png')
plot_comparison(mask_binary, log_binary, 'Laplacian of Gaussian Binary', 'comparison_log_binary.png')

# Calcul des métriques d'évaluation
def evaluate_segmentation(predicted, ground_truth):
    predicted_flat = predicted.flatten()
    ground_truth_flat = ground_truth.flatten()

    precision = precision_score(ground_truth_flat, predicted_flat, pos_label=1)
    recall = recall_score(ground_truth_flat, predicted_flat, pos_label=1)
    f1 = f1_score(ground_truth_flat, predicted_flat, pos_label=1)
    jaccard = jaccard_score(ground_truth_flat, predicted_flat, pos_label=1)
    mse = mean_squared_error(ground_truth_flat, predicted_flat)

    return precision, recall, f1, jaccard, mse

# Évaluation des différentes méthodes
performance_metrics = {
    'Manual Contrast': evaluate_segmentation(manual_contrast_binary, ground_truth),
    'Histogram Equalization': evaluate_segmentation(equalized_binary, ground_truth),
    'Top Hat': evaluate_segmentation(tophat_binary, ground_truth),
    'Otsu Threshold': evaluate_segmentation(binary_otsu_binary, ground_truth),
    'Hysteresis Threshold': evaluate_segmentation(edges_binary, ground_truth),
    'Opening': evaluate_segmentation(opened_binary, ground_truth),
    'Closing': evaluate_segmentation(closed_binary, ground_truth),
    'Sequential Filtering': evaluate_segmentation(filtered_binary, ground_truth),
    'Laplacian of Gaussian': evaluate_segmentation(log_binary, ground_truth)
}

# Sauvegarder les statistiques dans un fichier texte
stats_path = os.path.join(results_path, 'performance_metrics.txt')
with open(stats_path, 'w') as f:
    for method, metrics in performance_metrics.items():
        precision, recall, f1, jaccard, mse = metrics
        f.write(f"{method}:\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"  Recall: {recall:.4f}\n")
        f.write(f"  F1 Score: {f1:.4f}\n")
        f.write(f"  Jaccard Index: {jaccard:.4f}\n")
        f.write(f"  Mean Squared Error: {mse:.4f}\n\n")

print("Performance metrics have been saved to performance_metrics.txt")
