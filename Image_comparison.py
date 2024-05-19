from Image_comparison_class import *
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

        egal_im = cv2.equalizeHist(im)
        draw2Plots("Egalisation", nb, im, egal_im)
   
        brighter_im = imageBrightness(egal_im, 1.8, 0)
        draw2Plots("Brightness", nb, im, brighter_im)



        im_contour = passeHaut(egal_im, nb)
        draw3Plots("Contours", nb, egal_im, im_contour, mask)

        # 

        ret, seuil_im = cv2.threshold(egal_im, 10, 255, cv2.THRESH_BINARY_INV)
        draw2Plots("Seuillage", nb, mask, seuil_im)


imagesPrinting(image_path)
