import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np

def decompose(image):
    y = img.imread(image)

    #taille de l'image
    nbligne,nbcolonne = np.shape(y)

    y = np.transpose(y)
    image_vecteur=[]

    for i in range (nbcolonne):
        image_vecteur.extend(y[i])

    return [image_vecteur,nbligne,nbcolonne]