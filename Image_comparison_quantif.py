import matplotlib.pyplot as plt
import numpy as np

def hysteresis_thresholding(im, lowThresh, highThresh): #high is >0, low is <1
    H, W = im.shape
    
    # Apply thresholding
    im1 = im >= lowThresh
    im2 = im >= highThresh
    
    # Hysteresis thresholding
    imc = im2.copy()
    imc0 = im2.copy()
    imdiff = np.ones_like(imc)
    
    iter = 0
    while np.sum(imdiff) > 0:
        iter += 1
        x, y = np.where((im >= lowThresh) & (im < highThresh))
        
        for k in range(len(x)):
            if 1 < x[k] < H-1 and 1 < y[k] < W-1:
                if np.any(imc[x[k]-1:x[k]+2, y[k]-1:y[k]+2]):
                    imc[x[k], y[k]] = 1
        
        imdiff = imc > 0 & imc0 == 0      
        imc0 = imc.copy()
    return imc