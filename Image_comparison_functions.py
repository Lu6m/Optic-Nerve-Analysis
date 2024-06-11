import os
import cv2
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import colors
from skimage.morphology import disk, reconstruction, binary_opening, binary_closing


###########################################################################
####### Charger et ouvrir
###########################################################################
def loadImage(image_path, nb):
    im = None
    im_name = f"LC00{nb}"
    if nb == 10 : 
        im_name = "LC010"
    if os.path.exists(os.path.join(image_path, f"{im_name}.jpg")): 
        im = cv2.imread(f'{image_path}/{im_name}.jpg')
    elif os.path.exists(os.path.join(image_path, f"{im_name}.png")):
        im = cv2.imread(f'{image_path}/{im_name}.png')
    else : 
        print("The file path does not exist")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im_name, im

def loadMask(im_name):
    # Charger le .mat
    mat = scipy.io.loadmat(f'Terrain/{im_name}_VT.mat')
    # Récupérer juste la matrice et la mettre dnas la varibale mask (car binaire)
    mask = mat['seeds']
    plt.imsave(f'Terrain/{im_name}_VT.png', mask, cmap=plt.cm.gray)
    return mask

###########################################################################
####### Afficher images 
###########################################################################
def draw1Plot(name, nb, image1):
    plt.figure()
    plt.axis('off')
    plt.imsave(f"Results/{name}_{nb}.png", image1)

def drawOverlapPlot(name, nb, image1, image2, alpha, name1='', colormap1=plt.cm.gray, colormap2=plt.cm.gray):
    plt.figure()
    plt.imshow(image1, cmap=colormap1)
    plt.imshow(image2, cmap=colormap2, alpha = alpha)
    plt.axis('off')
    plt.title(name1)
    plt.savefig(f"Results/{name}_{nb}.png", bbox_inches='tight')
    plt.close()

def draw2Plots(name, nb, image1, image2, name1 = '', name2='', colormap1=plt.cm.gray, colormap2=plt.cm.gray):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image1, cmap=colormap1)
    axes[0].set_title(name1)
    axes[1].imshow(image2, cmap=colormap2)
    axes[1].set_title(name2)
    for ax in axes:
        ax.axis('off')
    plt.savefig(f"Results/{name}_{nb}.png", bbox_inches='tight')
    plt.close()

def draw3Plots(name, nb, image1, image2, image3, name1='', name2='', name3='',colormap1=plt.cm.gray, colormap2=plt.cm.gray, colormap3=plt.cm.gray):
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(image1, cmap=colormap1)
    axes[0].set_title(name1)
    axes[1].imshow(image2, cmap=colormap2)
    axes[1].set_title(name2)
    axes[2].imshow(image3, cmap=colormap3)
    axes[2].set_title(name3)
    for ax in axes:
        ax.axis('off')
    plt.savefig(f"Results/{name}_{nb}.png", bbox_inches='tight')
    plt.close()


###########################################################################
######## Pré-traitement image 
###########################################################################
def deleteRectangles(im):
    _, binary = cv2.threshold(im, 10, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect_mask = np.ones(im.shape, dtype=np.uint8) * 255

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (w > 20 and h > 20) and (x < 10 or x + w > im.shape[1] - 10):
            cv2.drawContours(rect_mask, [contour], -1, 0, -1)

    rect_free_im = im.copy()
    rect_free_im[rect_mask == 0] = 128
    return rect_free_im

def imageBrightness(im, contrast_coeff, brightness_coeff): #coeff entre 1 et 3 (en dessous de 1, diminue le contrast)
    new_image = im*contrast_coeff + brightness_coeff
    new_image = np.clip(new_image, 0, 255)  # Limit values to the range [0, 255]
    return new_image

###########################################################################
######## Traitement image 
###########################################################################
def threshold(im, thresh):
    blur_im = cv2.GaussianBlur(im,(5,5),0)
    _, seuil_im = cv2.threshold(blur_im, thresh, 255, cv2.THRESH_BINARY_INV)
    seuil_im = cv2.bitwise_not(seuil_im)
    filled_seuil_im = fillHoles(seuil_im)
    return filled_seuil_im

def openClose(im, thresh):
    SE7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    SE9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))

    opening3 = cv2.morphologyEx(im, cv2.MORPH_OPEN, SE7)
    closing3 = cv2.morphologyEx(opening3, cv2.MORPH_CLOSE, SE9)
    _, close_seuil_im = cv2.threshold(closing3, thresh, 255, cv2.THRESH_BINARY_INV)
    close_seuil_im = cv2.bitwise_not(close_seuil_im)
    filled_close_seuil_im = fillHoles(close_seuil_im)
    return filled_close_seuil_im

def topHat(bin_im):
    k = 30
    change = True
    init_tophat = bin_im
    while change : 
        k+=1
        SE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
        tophat_im = cv2.morphologyEx(init_tophat, cv2.MORPH_TOPHAT, SE)
        filled_tophat_im = fillHoles(tophat_im)
        if np.array_equal(filled_tophat_im, init_tophat):
            change = False
        init_tophat = filled_tophat_im
    print(k)
    return filled_tophat_im


def FAS(bin_im, MAX_TAILLE):
    tailles = range(1, MAX_TAILLE + 1)

    SE = [disk(r) for r in tailles]
    fas = bin_im.copy()

    # Débruitage par FAS
    for se in SE:
        openR = reconstruction(binary_opening(fas, se), fas)  # Ouverture par reconstruction
        fas = 1 - reconstruction(1 - binary_closing(openR, se), 1 - openR)  # Fermeture par reconstruction
    return fas


###########################################################################
######## Amélioration traitement image  
###########################################################################

def ReconstructionDilatation(F, IC, SE):
    """
    Parameters:
    F (ndarray): Marker image
    IC (ndarray): Mask image (complement of original image)
    SE (ndarray): Structuring element
    """
    prev_F = np.zeros_like(F)
    while not np.array_equal(F, prev_F):
        prev_F = F.copy()
        F = cv2.dilate(F, SE)
        F = np.minimum(F, IC)
    return F

# Fill the holes in the white structures 
def fillHoles(im):
    im = (im > 0).astype(np.uint8)

    # Dimensions of the source image
    H, W = im.shape

    # Structuring element
    SE = np.ones((3, 3), np.uint8)

    # Create the marker image F
    F = np.zeros((H, W), dtype=np.uint8)
    F[:, 0] = 1 - im[:, 0]
    F[:, -1] = 1 - im[:, -1]
    F[0, :] = 1 - im[0, :]
    F[-1, :] = 1 - im[-1, :]

    IC = 1 - im
    HC = ReconstructionDilatation(F, IC, SE)
    H = 1 - HC
    return H

###########################################################################
######## Contours 
###########################################################################

def imfilter(im,noyau):
    # Creation du bordage en miroir 
    h,w = np.shape(noyau)
    hbis = int(np.floor(h/2))
    wbis = int(np.floor(w/2))
    im_bis = np.pad(im, ((hbis,hbis),(wbis,wbis)),'symmetric')
    [Hbis,Wbis] = im_bis.shape
    # Delete le bordage 
    conv = sp.signal.convolve2d(im_bis, noyau)
    conv_bis = conv[hbis:Hbis-hbis, wbis:Wbis-wbis]
    return conv_bis

# Filtre dérivatuer, passe haut 
def passeHaut(im, nb): 
    noyau_horiz = np.array([[-1, 0, 1]])
    noyau_vert = np.array([[-1], [0], [1]])

    im_horiz = imfilter(im,noyau_horiz)
    im_vert = imfilter(im,noyau_vert)
    im_contour = np.sqrt(im_horiz**2 + im_vert**2)
    return im_contour

###########################################################################
###### Define the quantifications to determine which image is better
###########################################################################
def quantification(im, mask):
    false_positives = np.sum((im == 1) & (mask == 0))
    false_negatives = np.sum((im == 0) & (mask == 1))
    true_positives = np.sum((im == 1) & (mask == 1))
    true_negatives = np.sum((im == 0) & (mask == 0))
    false_positives += np.sum((im == 255) & (mask == 0))
    false_negatives += np.sum((im == 0) & (mask == 255))
    true_positives += np.sum((im == 255) & (mask == 255))
    true_negatives += np.sum((im == 0) & (mask == 0))
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return [false_positives, false_negatives, true_positives, true_negatives, precision, recall, f1_score]

def contoursCaracteristics(bin_im):
    edges = bin_im.copy()
    shape_count = 0
    total_area = 0
    total_perimeter = 0
    edges = edges.astype(np.uint8)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Approximation du contour
        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.01 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        # Filtrer les formes en fonction de leurs caractéristiques
        area = cv2.contourArea(contour)
        if area > 100:  # Filtrer par taille
            shape_count += 1
            total_area += area
            total_perimeter +=perimeter
            cv2.drawContours(edges, [approx], -1, (255, 0, 0), 2)
            cv2.circle(edges, (cx, cy), 5, (255, 0, 0), -1)
    mean_area = total_area / shape_count if shape_count > 0 else 0
    mean_perimeter = total_perimeter / shape_count if shape_count > 0 else 0
    return edges, [shape_count, mean_area, mean_perimeter]

def name_exists(df, name):
    return (df['Filter'] == name).any()

def fillDataframe(name, im, bin_im, df, mask, nb):
    name = f'{name}_{nb}'
    edges, contour = contoursCaracteristics(bin_im)
    new_name = f'Contours_{name}'
    drawOverlapPlot(f'{new_name}_bis', nb, im, edges, 0.4, colormap2=colors.ListedColormap(['white', 'red']), name1 = 'Detected edges over original image')
    draw2Plots(new_name, nb, mask, edges, name1 = 'Vérité Terrain ', name2=f'Contours Image seuillée avec {name}')
    quantif = quantification(bin_im, mask)
    quantif = quantif + contour
    if name_exists(df, name):
        df.drop(df[df['Filter'] == name].index, inplace = True)
    quantif.insert(0, name)
    df.loc[len(df.index)] = quantif
    return df