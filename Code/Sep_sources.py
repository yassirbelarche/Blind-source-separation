# Separation de source avec le premier algorithme MeR 2004
# Version Python

import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import math as m
from decompose import decompose
from reconstruct import reconstruct
from compute_gradient import compute_gradient
from correl_coef_composante_nb import correl_coef_composante_nb


#Decompose transforme une matrice en un vecteur 
#Construction des deux signaux sous forme vectorielle
#Calcul du nombre de ligne et de colonne dont on se servira 
# pour reconstruire les signaux mélangés et séparés

[image_vecteur_s1,nbligne_s1,nbcolonne_s1]=decompose('test1_gray.png')
image_dep1= img.imread('test1_gray.png')
plt.title('image source 1')
plt.imshow(image_dep1, cmap='gray')
plt.show()

[image_vecteur_s2,nbligne_s2,nbcolonne_s2]=decompose('test2_gray.png')
image_dep2= img.imread('test2_gray.png')
plt.title('image source 2')
plt.imshow(image_dep2, cmap='gray')
plt.show()
 


#Création de la matrice de melange X = A* S,A est l'opérateur de mélange 
# dans la "vraie vie", il est inconnu, ici pour image/tester l'algorithme 
# on le simule

A=np.array([[0.6, 0.4], [0.4,0.6]])

# les signaux sources (ici les deux images test1 et test2)
# avec lesquelles on va tester l'algorithme
# inconnues dans la "vraie vie", elles vont nous donner
# les deux sorties mélangées(x1,x2)
s1=image_vecteur_s1
s2=image_vecteur_s2

Mat_cor = correl_coef_composante_nb(s1,s2)
print(Mat_cor)

# on normalise
s1=s1/np.std(s1, ddof=1)
s2=s2/np.std(s2, ddof=1)

s = np.array([s1,s2])
x1=np.dot(A[0],s)
x2=np.dot(A[1],s)

# reconstruction des matrices pour retrouver les images "melangées" associées"
# vecteur --> matrice
# calcul préliminaire
xx1=(x1*np.std(x1, ddof=1))
xx2=(x2*np.std(x2, ddof=1))
X1=max(abs(xx1))
X2=max(abs(xx2))
xx1=xx1/X1*255
xx2=xx2/X2*255

# reconstruction de nos images melangées
image_matrice_s1=reconstruct(xx1,nbligne_s1,nbcolonne_s1)
# title
plt.imshow(image_matrice_s1, cmap='gray')
plt.title('image melangee 1')
plt.show()

image_matrice_s2=reconstruct(xx2,nbligne_s2,nbcolonne_s2)
# title
plt.imshow(image_matrice_s2, cmap='gray')
plt.title('image melangee 2')
plt.show()

# normalisation des deux sorties mélangées
x1=x1/np.std(x1, ddof=1)
x2=x2/np.std(x2, ddof=1)

#création de la matrice de séparation
nb_iter=200
B=np.eye(2)
mu=0.05   
lam=1.
y1=np.copy(x1)
y2=np.copy(x2)

RSBB1 = []
RSBB2 = []

for i in range(nb_iter):
    C=np.transpose(B)

    [GradIM,Gradpen] = compute_gradient(y1,y2,x1,x2)
    
    B=B-mu*(-np.dot(C,GradIM)-np.eye(2))-mu*lam*np.dot(C,Gradpen)


    #calcul de la separation
    X=np.array([x1,x2])
    y1=np.dot(B[0],X)
    y2=np.dot(B[1],X)

    #calcul fictif pour recuperer la moyenne et la variance
    XX=np.array([xx1,xx2])
    yy1=np.dot(B[0],XX)
    yy2=np.dot(B[1],XX)

    m_yy1=np.mean(yy1, axis=0)
    m_yy2=np.mean(yy2, axis=0)
    e_yy1=np.std(yy1, ddof=1)
    e_yy2=np.std(yy2, ddof=1)

    y1est = y1              #pour vérifier les écarts types
    y2est = y2              #rappel : on cherche les sources 
                            #ayant un écart type =1

    #calcul des rapports signal/bruit(diaphonie)
    BA=np.dot(B,A)

    # ModuleBruit1 = sum((BA[0][1]*s2)**2) + sum((BA[1][1]*s2)**2)
    # ModuleBruit2 = sum((BA[0][0]*s1)**2) + sum((BA[1][0]*s1)**2)


    Bruit1 = BA[0][1]*s2
    Bruit2 = BA[1][0]*s1
    # # BA[1][0]*s1

    RSBB1.append(10*m.log10(np.mean(y1est**2)/np.mean(Bruit1**2)))
    RSBB2.append(10*m.log10(np.mean(y2est**2)/np.mean(Bruit2**2)))


#Calcul des matrice de correlation 
#rappel : lorsque deux signaux sont indépendants ou décorrélés
#Cette matrice est la matrice identité

# [Mat_cor] = correl_coef_composante_nb(x1,x2)
Mat_cor = correl_coef_composante_nb(y1est,y2est)
 
#reconstruction des matrices pour obtenir les images séparées
#cette reconstruction est obtenu avec l'aide de la variance et 
#la moyenne des images de mélanges
y1=y1*e_yy1
y2=y2*e_yy2

V1=y1/max(y1)*255
V2=y2/max(y2)*255

imagematrices1=reconstruct(V1,nbligne_s1,nbcolonne_s1)
YY1=np.uint8(imagematrices1)
plt.imshow(YY1, cmap='gray')
plt.title('image separée 1')
plt.show()

imagematrices2=reconstruct(V2,nbligne_s2,nbcolonne_s2)
YY2=np.uint8(imagematrices2)
plt.imshow(YY2, cmap='gray')
plt.title('image separée 2')
plt.show() 

#calcul des rapports signal/bruit



# BA=np.dot(B,A)

# Bruit1 = BA[0][1]*s2
# Bruit2 = BA[1][0]*s1

# RSBB1=10*m.log10(np.mean(y1est)**2/np.mean(Bruit1)**2)
# RSBB2=10*m.log10(np.mean(y2est)**2/np.mean(Bruit2)**2)

# print("RSN1 =", RSBB1)
# print("RSN2 =", RSBB2)


#Tracer les courbes des RSN en foction de nombre d'itérations

Iteration = list(range(nb_iter))

print()

plt.subplot(2,1,1)
plt.plot(Iteration, RSBB1)
plt.subplot(2,1,2)
plt.plot(Iteration, RSBB2)
# # plt.subplot(4,1,3)
# # plt.plot(BB3)
# # plt.subplot(4,1,4)
# # plt.plot(BB4)

plt.show()