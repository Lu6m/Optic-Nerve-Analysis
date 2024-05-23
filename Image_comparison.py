from Image_comparison_functions import *

import matplotlib.pyplot as plt
from matplotlib import colors
import PIL 

# All the path 
projet_path = os.getcwd()
print(projet_path)
image_path = os.path.join(projet_path, "Lame_criblee")
mask_path = os.path.join(projet_path, "Terrain")
result_path = os.path.join(projet_path, "Results")
if not os.path.exists(result_path):
    os.makedirs(result_path)

# Main function 
def imagesPrinting(image_path):
    for nb in range(1,11):
        im_name, im = loadImage(image_path, nb)
        mask = loadMask(im_name)
        drawOverlapPlot("Overlap", nb, im, mask, 0.3, colormap2=colors.ListedColormap(['white', 'red']))

        rect_free_im = deleteRectangles(im)
        
        egal_im = cv2.equalizeHist(rect_free_im)
        draw2Plots("Egalisation", nb, im, egal_im)
   
        brighter_im = imageBrightness(egal_im, 1.8, 0)
        draw2Plots("Brightness", nb, im, brighter_im)

        im_contour = passeHaut(brighter_im, nb)
        draw3Plots("Contours", nb, egal_im, im_contour, mask)

        thresh = 30
        _, seuil_im = cv2.threshold(brighter_im, thresh, 255, cv2.THRESH_BINARY_INV)
        draw2Plots("Threshold", nb, mask, seuil_im)

        closing = cv2.morphologyEx(seuil_im, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
        draw2Plots("OpenClose", nb, mask, opening)


imagesPrinting(image_path)
