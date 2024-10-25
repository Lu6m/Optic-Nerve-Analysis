from Image_comparison_functions import *
import matplotlib.pyplot as plt

#############################################################################
#### Run the code, it should work automatically : py Image_comparison.py
#############################################################################

# All the path 
home = os.getcwd()
project_path = os.path.join(projet_path, "Datasets")
print(projet_path)
image_path = os.path.join(projet_path, "Lame_criblee")
mask_path = os.path.join(projet_path, "Terrain")
result_dir_path = os.path.join(home, "Results")
if not os.path.exists(result_dir_path):
    os.makedirs(result_dir_path)

# Main function 
def imagesPrinting(image_path):
    columns = ['Filter', 'FP', 'FN', 'VP', 'VN', 'precision', 'rappel', 'f1_score', 'shape_count', 'mean_area', 'area_var', 'area_std_dev', 'mean_perimeter']
    df = pd.DataFrame(columns = columns)

    for nb in range(1,11):
        print(nb)
        im_name, im = loadImage(image_path, nb)

        result_path = os.path.join(result_dir_path, im_name)
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        mask = loadMask(im_name)
        drawOverlapPlot(result_path, "Overlap", nb, im, mask, 0.3, colormap2=colors.ListedColormap(['white', 'red']))

        name = "Rectangle_free"
        rect_free_im = deleteRectangles(im)
        draw1Plot(result_path, name, nb, rect_free_im)

        name = "Invert"
        im_inverted = cv2.bitwise_not(rect_free_im)
        draw1Plot(result_path, name, nb, im_inverted)
        
        name = "Egalisation"
        egal_im = cv2.equalizeHist(im_inverted)
        draw1Plot(result_path, name, nb, egal_im)
        draw2Plots(result_path, f'{name}_bis', nb, im, egal_im, name1 = 'Image Originale', name2='Image Egalisée')
   
        contrast_coeff = 1.8
        brightness_coeff = 0
        brighter_im = imageBrightness(egal_im, contrast_coeff, brightness_coeff)
        # draw2Plots(result_path, "Brightness", nb, im, brighter_im, name1 = 'Image Originale', name2='Image Eclaircie')

        #im_contour = passeHaut(brighter_im, nb)
        #draw3Plots(result_path, "Contours", nb, egal_im, im_contour, mask)

        name = 'Threshold'
        thresh = 230
        seuil_im, filled_seuil_im = threshold(egal_im, thresh)
        draw1Plot(result_path, f'Holed_{name}', nb, seuil_im)
        draw1Plot(result_path, f'Filled_{name}', nb, filled_seuil_im)
        draw2Plots(result_path, name, nb, mask, filled_seuil_im, name1 = 'Vérité Terrain ', name2=f'Image seuillée à {thresh}')
        df = fillDataframe(name, filled_seuil_im, df, mask, nb)
        
        name = 'OpenClose'
        opening, closing, close_seuil_im, filled_close_seuil_im = openClose(egal_im, thresh)
        draw1Plot(result_path, f'{name}_bis', nb, filled_close_seuil_im)
        draw2Plots(result_path, name, nb, mask, filled_close_seuil_im, name1 = 'Vérité Terrain ', name2=f'Image seuillée avec ouverture & fermeture')
        df = fillDataframe(name, filled_close_seuil_im, df, mask, nb)

        name = 'TopHat'
        se_size = 59
        thresh = 230
        SE = cv2.getStructuringElement(cv2.MORPH_RECT, (se_size, se_size))
        top_hat = cv2.morphologyEx(egal_im, cv2.MORPH_TOPHAT, SE)
        draw1Plot(result_path, f'{name}_tophat', nb, top_hat)
        draw2Plots(result_path, name, nb, egal_im, top_hat, name1 = 'Image Originale ', name2=f'Image tophat')
        _, seuil_tophat_im = cv2.threshold(top_hat, thresh, 255, cv2.THRESH_BINARY)
        draw1Plot(result_path, f'Holed_ {name}_', nb, seuil_tophat_im)
        filled_seuil_tophat_im = fillHoles(seuil_tophat_im)
        draw2Plots(result_path, f'{name}_bis', nb, mask, filled_seuil_tophat_im, name1 = 'Vérité Terrain ', name2=f'Image seuillée ')
        draw1Plot(result_path, f'Filled_{name}', nb, filled_seuil_tophat_im)
        df = fillDataframe(name, filled_seuil_tophat_im, df, mask, nb)

        #name = 'FiltreAlternesSeq'
        #max_taille = 1
        #fas = FAS(egal_im, max_taille)
        #draw2Plots(result_path, name, nb, mask, fas, name1 = 'Vérité Terrain', name2=f'Image avec FAS')
        #df = fillDataframe(name, fas, df, mask, nb)

        min_area = 90
        name = f'Contours_Threshold'
        edges, filled_contours, contour_values = contoursCaracteristics(filled_seuil_im, min_area)
        drawOverlapPlot(result_path, f'{name}_bis', nb, im, edges, 0.4, colormap2=colors.ListedColormap(['white', 'red']), name1 = 'Detected edges over original image')
        draw1Plot(result_path, f'{name}_bis2', nb, edges)
        draw2Plots(result_path, name, nb, mask, edges, name1 = 'Vérité Terrain ', name2=name)
        df = fillDataframeShapes(f'{name}_{min_area}', filled_contours, df, mask, nb, contour_values)

        name = f'Contours_OpenClose'
        edges, filled_contours, contour_values = contoursCaracteristics(filled_close_seuil_im, min_area)
        drawOverlapPlot(result_path, f'{name}_bis', nb, im, edges, 0.4, colormap2=colors.ListedColormap(['white', 'red']), name1 = 'Detected edges over original image')
        draw1Plot(result_path, f'{name}_bis2', nb, edges)
        draw2Plots(result_path, name, nb, mask, edges, name1 = 'Vérité Terrain ', name2=name)
        df = fillDataframeShapes(f'{name}_{min_area}', filled_contours, df, mask, nb, contour_values)

        name = 'Contours_TopHat'
        edges, filled_contours, contour_values = contoursCaracteristics(filled_seuil_tophat_im, min_area)
        drawOverlapPlot(result_path, f'{name}_bis', nb, im, edges, 0.4, colormap2=colors.ListedColormap(['white', 'red']), name1 = 'Detected edges over original image')
        draw1Plot(result_path, f'{name}_bis2', nb, edges)
        draw2Plots(result_path, name, nb, mask, edges, name1 = 'Vérité Terrain ', name2=name)
        df = fillDataframeShapes(f'{name}_{min_area}', filled_contours, df, mask, nb, contour_values)

        #name = f'Contours_FAS'
        #edges, filled_contours, contour_values = contoursCaracteristics(fas, min_area)
        #drawOverlapPlot(result_path, f'{name}_bis', nb, im, edges, 0.4, colormap2=colors.ListedColormap(['white', 'red']), name1 = 'Detected edges over original image')
        #draw2Plots(result_path, name, nb, mask, edges, name1 = 'Vérité Terrain ', name2=name)
        #df = fillDataframeShapes(f'{name}_{min_area}', filled_contours, df, mask, nb, contour_values)

        min_area = 0
        name = f'Verite_terrain'
        edges, filled_contours, contour_values = contoursCaracteristics(mask, min_area)
        draw2Plots(result_path, name, nb, mask, filled_contours, name1 = 'Vérité Terrain Init', name2=name)
        df = fillDataframeShapes(f'{name}_{min_area}', filled_contours, df, mask, nb, contour_values)

    # Save results to CSV
    df.to_csv(os.path.join(result_dir_path,'Method_quantif.csv'), index=False)  

imagesPrinting(image_path)



def findBestParamFAS():
    Highf1, HighMaxTaille, HighThresh = [], [], []
    for nb in range(1,11):
        Highestf1 = 0
        HighestMaxTaille = 0
        HighestThresh = 0 
        for max_taille in range (10):
            for thresh in range(230, 255, 5):
                print(nb, max_taille, thresh)
                im_name, im = loadImage(image_path, nb)
                mask = loadMask(im_name)
                rect_free_im = deleteRectangles(im)
                im_inverted = cv2.bitwise_not(rect_free_im)
                egal_im = cv2.equalizeHist(im_inverted)

                _, fas_bin = FAS(egal_im, max_taille, thresh)
                [_, _, _, _, _, _, f1_score] = quantification(fas_bin, mask)
                if f1_score > Highestf1 : 
                    Highestf1 = f1_score
                    HighestMaxTaille = max_taille
                    HighestThresh = thresh
        Highf1.append(Highestf1)
        HighMaxTaille.append(HighestMaxTaille)
        HighThresh.append(HighestThresh)
    return Highf1, HighMaxTaille, HighThresh


#Highf1, HighMaxTaille, HighThresh = findBestParamFAS()
#print(Highf1, HighMaxTaille, HighThresh)


def findBestParamTopHat():
    Highf1, HighSEsize, HighThresh = [], [], []
    for nb in range(1,11):
        Highestf1, HighestSEsize, HighestThresh = 0, 0, 0
        for se_size in range (31,61,2):
            for thresh in range(230, 255, 5):
                print(nb, se_size, thresh)
                im_name, im = loadImage(image_path, nb)
                mask = loadMask(im_name)
                rect_free_im = deleteRectangles(im)
                im_inverted = cv2.bitwise_not(rect_free_im)
                egal_im = cv2.equalizeHist(im_inverted)

                SE = cv2.getStructuringElement(cv2.MORPH_RECT, (se_size, se_size))
                top_hat = cv2.morphologyEx(egal_im, cv2.MORPH_TOPHAT, SE)
                _, seuil_tophat_im = cv2.threshold(top_hat, thresh, 255, cv2.THRESH_BINARY)
                filled_seuil_tophat_im = fillHoles(seuil_tophat_im)
                [_, _, _, _, _, _, f1_score] = quantification(filled_seuil_tophat_im, mask)
                if f1_score > Highestf1 : 
                    Highestf1 = f1_score
                    HighestSEsize = se_size
                    HighestThresh = thresh
        Highf1.append(Highestf1)
        HighSEsize.append(HighestSEsize)
        HighThresh.append(HighestThresh)
    return Highf1, HighSEsize, HighThresh

#Highf1, HighSEsize, HighThresh = findBestParamTopHat()
#print(Highf1, HighSEsize, HighThresh)

def findBestParamThresh():
    Highf1, HighArea, HighThresh = [], [], []
    for nb in range(1,11):
        Highestf1, HighestArea, HighestThresh = 0, 0, 0
        for min_area in range (5, 50, 2):
            for thresh in range(230, 255, 5):
                print(nb, min_area, thresh)
                im_name, im = loadImage(image_path, nb)
                mask = loadMask(im_name)
                rect_free_im = deleteRectangles(im)
                im_inverted = cv2.bitwise_not(rect_free_im)
                egal_im = cv2.equalizeHist(im_inverted)

                _, filled_seuil_im = threshold(egal_im, thresh)
                edges, filled_contours, contour_values = contoursCaracteristics(filled_seuil_im, min_area)
                [_, _, _, _, _, _, f1_score] = quantification(filled_seuil_im, mask)
                
                if f1_score > Highestf1 : 
                    Highestf1 = f1_score
                    HighestArea = min_area
                    HighestThresh = thresh
        Highf1.append(Highestf1)
        HighArea.append(HighestArea)
        HighThresh.append(HighestThresh)
    return Highf1, HighArea, HighThresh

#Highf1, HighArea, HighThresh = findBestParamThresh()
#print(Highf1, HighArea, HighThresh)