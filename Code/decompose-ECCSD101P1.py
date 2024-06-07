import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np

def decompose(image):
    y = img.imread(image)
    # y = np.array(y)
    print(np.shape(y))

    # nbligne = np.shape(y)[0]
    # nbcolonne = np.shape(y)[1]

    y = np.transpose(y)

    [nbcolonne,nbligne]= np.shape(y)
    image_vecteur=[]

    #taille de l'image

    #pour les images en couleur
    #permet de ne prendre que la 1ere couche de couleurs.
    #nbcolonne=nbcolonne/3;

    for i in range (nbcolonne):
        image_vecteur.extend(y[i])

    image_vecteur = np.transpose(image_vecteur)

    # image_vecteur=double(image_vecteur);
    # image_vecteur=image_vecteur-mean(image_vecteur);
    # image_vecteur=image_vecteur/std(image_vecteur);

    return [image_vecteur,nbligne,nbcolonne]

