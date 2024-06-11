import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu, gaussian, laplace
from skimage.morphology import white_tophat, disk
import os

# Créez le dossier 'Results' s'il n'existe pas déjà
results_path = 'Results_Chloe'
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Créez un sous-dossier spécifique pour les résultats de cette exécution
result_folder = os.path.join(results_path, 'Mask_Results')
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

# Vérifier si le fichier image existe dans le répertoire courant
image_path = os.path.join('Lame_criblee', 'LC002.jpg')  # Remplacez par le chemin de votre image
if not os.path.isfile(image_path):
    raise FileNotFoundError(f"Image not found at path: {image_path}")

# Charger l'image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError(f"Image not found or cannot be opened at path: {image_path}")

# II. Pré-traitement de l’image

# 2. Augmentation du contraste
def manual_contrast(image, factor):
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)

def histogram_equalization(image):
    return cv2.equalizeHist(image)

def top_hat(image):
    selem = disk(12)
    return white_tophat(image, selem)

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
log_image = laplacian_of_gaussian(image)

# Générer un masque binaire à partir de l'image originale
mask_binary = binarize(original_image)

# Binarisation des images traitées
manual_contrast_binary = binarize(manual_contrast_image)
equalized_binary = binarize(equalized_image)
tophat_binary = binarize(tophat_image)
log_binary = binarize(log_image)

# Sauvegarder les images traitées
save_image(original_image, 'original_image.png')
save_image(manual_contrast_image, 'manual_contrast_image.png')
save_image(equalized_image, 'equalized_image.png')
save_image(tophat_image, 'tophat_image.png')
save_image(log_image, 'log_image.png')

# Sauvegarder les images binaires
save_image(mask_binary, 'mask_binary.png')
save_image(manual_contrast_binary, 'manual_contrast_binary.png')
save_image(equalized_binary, 'equalized_binary.png')
save_image(tophat_binary, 'tophat_binary.png')
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
plot_comparison(original_image, log_image, 'Laplacian of Gaussian', 'comparison_log_image.png')

# Comparer et sauvegarder les images binaires avec le masque binaire généré
plot_comparison(mask_binary, manual_contrast_binary, 'Manual Contrast Binary', 'comparison_manual_contrast_binary.png')
plot_comparison(mask_binary, equalized_binary, 'Histogram Equalization Binary', 'comparison_equalized_binary.png')
plot_comparison(mask_binary, tophat_binary, 'Top Hat Binary', 'comparison_tophat_binary.png')
plot_comparison(mask_binary, log_binary, 'Laplacian of Gaussian Binary', 'comparison_log_binary.png')
