from Image_comparison_functions import *
from Image_comparison_thresh import *

import matplotlib.pyplot as plt

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
    columns = ['Filter', 'FP', 'FN', 'TP', 'TN', 'precision', 'recall', 'f1_score', 'shape_count', 'mean_area', 'mean_perimeter']
    df = pd.DataFrame(columns = columns)

    for nb in range(1,11):
        im_name, im = loadImage(image_path, nb)
        mask = loadMask(im_name)
        # drawOverlapPlot("Overlap", nb, im, mask, 0.3, colormap2=colors.ListedColormap(['white', 'red']))

        rect_free_im = deleteRectangles(im)
        im_inverted = cv2.bitwise_not(rect_free_im)
        
        egal_im = cv2.equalizeHist(im_inverted)
        draw2Plots("Egalisation", nb, im, egal_im, name1 = 'Image Originale', name2='Image Egalisée')
   
        contrast_coeff = 1.8
        brightness_coeff = 0
        brighter_im = imageBrightness(egal_im, contrast_coeff, brightness_coeff)
        draw2Plots("Brightness", nb, im, brighter_im, name1 = 'Image Originale', name2='Image Eclaircie')

        #im_contour = passeHaut(brighter_im, nb)
        #draw3Plots("Contours", nb, egal_im, im_contour, mask)

        name = 'Threshold'
        thresh = 230
        filled_seuil_im = threshold(egal_im, thresh)
        draw2Plots(name, nb, mask, filled_seuil_im, name1 = 'Vérité Terrain ', name2=f'Image seuillée à {thresh}')
        df = fillDataframe(name, im, filled_seuil_im, df, mask, nb)
        
        name = 'OpenClose'
        filled_close_seuil_im = openClose(egal_im, thresh)
        draw2Plots(name, nb, mask, filled_close_seuil_im, name1 = 'Vérité Terrain ', name2=f'Image seuillée avec ouverture & fermeture')
        df = fillDataframe(name, im, filled_close_seuil_im, df, mask, nb)

        name = 'TopHat'
        filled_tophat_im = topHat(filled_seuil_im)
        draw2Plots(name, nb, mask, filled_tophat_im, name1 = 'Vérité Terrain ', name2=f'Image seuillée ')
        df = fillDataframe(name, im, filled_tophat_im, df, mask, nb)

        name = 'FiltreAlternesSeq'
        MAX_TAILLE = 3
        fas = FAS(filled_seuil_im, MAX_TAILLE)
        draw2Plots(name, nb, mask, fas, name1 = 'Vérité Terrain ', name2=f'Image seuillée avec FAS')
        df = fillDataframe(name, im, fas, df, mask, nb)

    # Save results to CSV
    print(os.path.join(result_path,'Method_quantif.csv'))
    df.to_csv(os.path.join(result_path,'Method_quantif.csv'), index=False)  

imagesPrinting(image_path)

def findBestParam():
    Highestf1 = 0
    HighestMaxTaille = 0
    HighestThresh = 0 
    for MAX_TAILLE in range (10):
        for thresh in range(230, 255, 5):
            for nb in range(1,11):
                print(MAX_TAILLE, thresh, nb)
                im_name, im = loadImage(image_path, nb)
                mask = loadMask(im_name)
                rect_free_im = deleteRectangles(im)
                im_inverted = cv2.bitwise_not(rect_free_im)
                egal_im = cv2.equalizeHist(im_inverted)

                filled_seuil_im = threshold(egal_im, thresh)
                name = 'FiltreAlternesSeq'
                MAX_TAILLE
                fas = FAS(filled_seuil_im, MAX_TAILLE)
                [false_positives, false_negatives, true_positives, true_negatives, precision, recall, f1_score] = quantification(fas, mask)
                if f1_score > Highestf1 : 
                    Highestf1 = f1_score
                    HighestMaxTaille = MAX_TAILLE
                    HighestThresh = thresh
                    df = [false_positives, false_negatives, true_positives, true_negatives, precision, recall, f1_score]
    return HighestMaxTaille, HighestThresh, df

#HighestMaxTaille, HighestThresh, df = findBestParam()
#print(HighestMaxTaille, HighestThresh, df)