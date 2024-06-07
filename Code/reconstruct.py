import numpy as np

def reconstruct(vecteur,nbligne,nbcolonne): 
    image_matrice = []
    for i in range(nbcolonne):
        z=vecteur[i*nbligne: (i+1)*nbligne]
        image_matrice.append(z)
    
    image_matrice = np.transpose(image_matrice)
    
    return image_matrice