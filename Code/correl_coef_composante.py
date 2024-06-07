import numpy as np
def correl_coef_composante(im1_R,im1_G,im1_B,im2_R,im2_G,im2_B):
   N = len(im1_R)

   moy_1R = sum(im1_R)/len(im1_R)
   moy_1G = sum(im1_G)/len(im1_G)
   moy_1B = sum(im1_B)/len(im1_B)
   moy_2R =sum(im2_R)/len(im2_R)
   moy_2G = sum(im2_G)/len(im2_G)
   moy_2B = sum(im2_B)/len(im2_B)

   Mat_cor_R = np.zeros(2,2)
   Mat_cor_G = np.zeros(2,2)
   Mat_cor_B = np.zeros(2,2)

   ec_1R = np.std(im1_R)
   ec_1G = np.std(im1_G)
   ec_1B = np.std(im1_B)
   ec_2R = np.std(im2_R)
   ec_2G = np.std(im2_G)
   ec_2B = np.std(im2_B)

   ec_R = [ec_1R,ec_2R]
   ec_G = [ec_1G,ec_2G]
   ec_B = [ec_1B,ec_2B]

   moy_R = [moy_1R,moy_2R]
   moy_G = [moy_1G,moy_2G]
   moy_B = [moy_1B,moy_2B]

   ima_R = [[im1_R[:]],[im2_R[:]]]
   ima_G = [[im1_G[:]],[im2_G[:]]]
   ima_B = [[im1_B[:]],[im2_B[:]]]
   
   Mat_cor_B=[]
   Mat_cor_G=[]
   Mat_cor_R=[]

   
   for i in 12 :

      for j in 12 :

        Mat_cor_R[i][j] = (1/(N*ec_R[i]*ec_R[j]))*sum(  ( ima_R[:][i] - moy_R[i] )*( ima_R[:][j] - moy_R[j] ) )
        Mat_cor_G[i][j] = (1/(N*ec_G[i]*ec_G[j]))*sum(  ( ima_G[:][i] - moy_G[i] )*( ima_G[:][j] - moy_G[j] ) )     
        Mat_cor_B[i][j] = (1/(N*ec_B[i]*ec_B[j]))*sum(  ( ima_B[:][i] - moy_B[i] )*( ima_B[:][j] - moy_B[j] ) )    
      

   return [Mat_cor_R,Mat_cor_G,Mat_cor_B] 
