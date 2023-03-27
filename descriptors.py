import mahotas.features as ft
from skimage.feature import graycomatrix,graycoprops
from BiT import biodiversity,taxonomy
import numpy as np

def haralick(data):
    all_statistics = ft.haralick(data)
    return all_statistics.ravel()


def haralick_mean(data):
    all_statistics = ft.haralick(data).mean(0)
    return all_statistics.tolist()

def bitdesc(data):
    bio = biodiversity(data)
    #print(bio)
    taxo = taxonomy(data)
    #print(taxo)
    all_statistics = bio + taxo #concatenacion
    return all_statistics

def glcm(data):
    glcm = graycomatrix(data, distances=[2], angles=[0,90],levels=256,symmetric=True,normed=True)
    diss= graycoprops(glcm,prop='dissimilarity')[0,0]
    cont= graycoprops(glcm,prop='contrast')[0,0]
    corr= graycoprops(glcm,prop='correlation')[0,0]
    ener= graycoprops(glcm,prop='energy')[0,0]
    homo= graycoprops(glcm,prop='homogeneity')[0,0]
    all_features = [diss,cont,corr,ener,homo]
    return all_features