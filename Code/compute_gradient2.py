  
#Cette fonction calcul le gradient du critère pénalisé : 
#information mutuelle + lambda pénalisation
#où lambda est un hyperparamètre de pénalisation
#l'information mutuelle caractérise l'indépendance
#la pénalisation normalise le vecteur et le force à avoir 
#un écart-type = 1    
#biblio
          
import numpy as np
import numpy.linalg as nl

# Estimation polynomiale de la fonction score marginale (Psi) par moindres carrés
#Thèse Massoud Babaie Zadeh page 46   
def compute_gradient(y1,y2,x1,x2):
    m_y1=sum(y1)/len(y1)
    m_y12=sum(y1**2)/len(y1)
    m_y13=sum(y1**3)/len(y1)
    m_y14=sum(y1**4)/len(y1)
    m_y15=sum(y1**5)/len(y1)
    m_y16=sum(y1**6)/len(y1)
    m_y2=sum(y2**1)/len(y2)
    m_y22=sum(y2**2)/len(y2)
    m_y23=sum(y2**3)/len(y2)
    m_y24=sum(y2**4)/len(y2)
    m_y25=sum(y2**5)/len(y2)
    m_y26=sum(y2**6)/len(y2)

    #Calcul des différents parametres (approximation des fonctions scores
    # par moindres carrées
    
    K1=[1,m_y1,m_y12,m_y13]
    K2=[1,m_y2,m_y22,m_y23]
    M1=[1,m_y1,m_y12,m_y13,m_y1,m_y12,m_y13,m_y14,m_y12,m_y13,m_y14,m_y15,m_y13,m_y14,m_y15,m_y16]
    M2=[1,m_y2,m_y22,m_y23,m_y2,m_y22,m_y23,m_y24,m_y22,m_y23,m_y24,m_y25,m_y23,m_y24,m_y25,m_y26]
    P1=[0,1,2*m_y1,3*(m_y12)]
    P2=[0,1,2*m_y2,3*(m_y22)]
    w1=-nl.inv(M1)*P1
    w2=-nl.inv(M2)*P2
          
    #calcul du Psi
    Psi_y1=w1(1)+w1(2)*y1+w1(3)*y1**2+w1(4)*y1**3
    Psi_y2=w2(1)+w2(2)*y2+w2(3)*y2**2+w2(4)*y2**3
    Psi_y=[Psi_y1,Psi_y2]
    
   
    #Calcul de la Jacobienne de l'information mutuelle
    M_Psi11=sum(Psi_y1*x1)/len(x1)
    M_Psi12=sum(Psi_y1*x2)/len(x2)
    M_Psi21=sum(Psi_y2*x1)/len(x1)
    M_Psi22=sum(Psi_y2*x2)/len(x2)
    Sep = [M_Psi11,M_Psi12,M_Psi21,M_Psi22]
    
    #J'enlève la moyenne
    y1=y1-(sum(y1)/len(y1))
    y2=y2-(sum(y2)/len(y2))
    
    #Calcul de la Jacobienne de la normalisation (penalisation) 
    # papier Mohammed El Rhabi 2004
    temp1=4*(sum(y1**2)/len(y1)-1).dot(y1)
    temp2=4*(sum(y2**2)/len(y2)-1).dot(y2)
    m_y1x1=sum(temp1*x1)/len(x1)
    m_y1x2=sum(temp1*x2)/len(x2)
    m_y2x1=sum(temp2*x1)/len(x1)
    m_y2x2=sum(temp2*x2)/len(x2)
    pen=[[m_y1x1, m_y1x2],[m_y2x1, m_y2x2]]
