#Attention ligne 11;27
#pour ligne11 : moy_1 = np.mean(A, axis=0)

import numpy as np
from column import Colonne

def correl_coef_composante_nb(im1,im2):
   N = len(im1)

   moy_1 = sum(im1)/N
   moy_2 = sum(im2)/N

   Mat_cor = np.zeros((2,2))

   ec_1 = np.std(im1, ddof=1)
   ec_2 = np.std(im2, ddof=1)

   I2=np.ones((2,1))

   ec = I2
   ec[0] = ec_1
   ec[1] = ec_2


   moy = I2
   moy[0] = moy_1
   moy[1] = moy_2

   ima = list(zip(im1, im2))
   
   for i in range(len(ima)) :
      ima[i]=list(ima[i])

   Mat_cor=np.zeros((2,2))

   for i in range(2):

      for j in range(2):
         IN = np.ones((1,N))
         N2 = N*ec[i]*ec[j]
         Vecteur = (Colonne(ima,i) - moy[i]*IN) * (Colonne(ima,j) - moy[j]*IN)

         v = sum(sum(Vecteur)) / N2

         Mat_cor[i][j] = v
         
   return Mat_cor
      
   



