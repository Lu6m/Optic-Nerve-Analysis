import os
import cv2
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

def loadImage(image_path, nb):
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

def test():
    print("Test")

def loadMask(im_name):
    # Charger le .mat
    mat = scipy.io.loadmat(f'Terrain/{im_name}_VT.mat')
    # Récupérer juste la matrice et la mettre dnas la varibale  mask (car binaire) 
    mask = mat['seeds']
    plt.imsave(f'Terrain/{im_name}_VT.png', mask, cmap=plt.cm.gray)
    return mask

def draw1Plot(name, nb, image1):
    plt.figure()
    plt.imsave(f"Results/{name}_{nb}.png", image1)

def drawOverlapPlot(name, nb, image1, image2, alpha, colormap1=plt.cm.gray, colormap2=plt.cm.gray):
    plt.figure()
    plt.imshow(image1, cmap=colormap1)
    plt.imshow(image2, cmap=colormap2, alpha = alpha)
    plt.savefig(f"Results/{name}_{nb}.png", bbox_inches='tight')
    plt.close()

def draw2Plots(name, nb, image1, image2, colormap1=plt.cm.gray, colormap2=plt.cm.gray):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image1, cmap=colormap1)
    axes[1].imshow(image2, cmap=colormap2)
    plt.savefig(f"Results/{name}_{nb}.png", bbox_inches='tight')
    plt.close()

def draw3Plots(name, nb, image1, image2, image3, colormap1=plt.cm.gray, colormap2=plt.cm.gray, colormap3=plt.cm.gray):
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    axes[0].imshow(image1, cmap=colormap1)
    axes[1].imshow(image2, cmap=colormap2)
    axes[2].imshow(image3, cmap=colormap3)
    plt.savefig(f"Results/{name}_{nb}.png", bbox_inches='tight')
    plt.close()

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

def adaptBrightness(im):
    hist = cv2.calcHist([im], [0], None, [256], [0,256])
    hist = cv2.calcHist([new_image], [0], None, [256], [0,256])
    med_im = np.median(hist)
    med_im = np.median(hist)
    while 0 < med_im:
        if 0>1:
            new_image = imageBrightness(im, contrast_coeff, brightness_coeff)
            return 

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
