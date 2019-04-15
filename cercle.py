# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.spatial import Delaunay
from math import pi, cos, sin, acos

###############################################################################
# METHODES
###############################################################################

# Genere les points formant le contour du cercle
def genererContourCercle(resolution):
  """
  Entree: resolution (nbre de pts de la discretisation du cercle)
  Sortie: numpy.array contenant l'ensemble points de la
  discretisation du cercle

  On genere les points du contour sur un quadrant de facon reguliere, 
  puis on place les symetriques en manipulant les coordonnees:
            _____
          /   |   \ 
        /     |     \ 
      /       |       \ 
    /         |         \ 
   | (-y,x)   |   (x,y)  |
   |----------+----------|
   | (-x,-y)  | (y,-x)   |
    \         |         /
      \       |       / 
        \     |     / 
          \ __|__ /  
  C'est juste pour economiser du temps de calcul sur les cos() et sin(),
  c'est peut-etre pas utile
  """
  
  # On initialise le tableau qui contiendra les points
  ptsContourQuadrant = np.zeros(shape=(resolution*4, 2))

  # On va faire varier l'angle entre 0 et pi/4 (45 deg)
  increment = (pi/2) / resolution
  angle = -increment
  for i in range (resolution):
    angle += increment
    x = cos(angle)
    y = sin(angle)
    ptsContourQuadrant[i] = [x, y]                #top-right
    ptsContourQuadrant[resolution+i] = [-y, x]    #top-left
    ptsContourQuadrant[2*resolution+i] = [-x, -y] #bottom-left
    ptsContourQuadrant[3*resolution+i] = [y, -x]  #bottom-right

  return ptsContourQuadrant

# Sous-fonction de genererInterieurCercle, pour un seul point
def genererPointInterieur(angleMin, angleMax):
  """
  Entree: intervalle d'angle pour des coordonnees polaires
  Sortie: coordonnees d'un point quelconque sur une portion du disque unite
  """
  angle = random.uniform(angleMin, angleMax)
  distance = random.random()
  return [distance * cos(angle), distance * sin(angle)]

# Genere des points aleatoirement dans un disque
def genererInterieurCercle(nbPointsInterieurs):
  # On initialise le tableau qui contiendra les points
  ptsInterieurs = np.zeros(shape=(nbPointsInterieurs, 2))

  for i in range(nbPointsInterieurs):
    ptsInterieurs[i] = genererPointInterieur(0, 2*pi)

  return ptsInterieurs

# Sous-fonction pour qualiteTriangle
def calculerAngle(a, b, c):
  """
  Entree: 3 points A[xa, ya], B[xb, yb], C[xc, yc]
  Sortie: angle (AB, AC) en radians

  - On utilise les coordonnees de facon vectorielle: 
  AB = B - A = [xb-xa, yb-ya] etc.
  - On utilise la formule du produit scalaire avec cosinus:
  AB.AC = ||AB||.||AC||.cos(AB,AC) => (AB,AC) = arccos(AB.AC/||AB||.||AC||)
  """
  ab = b - a
  ac = c - a
  prodScal = np.dot(ab,ac)
  prodNorm = np.linalg.norm(ab) * np.linalg.norm(ac)
  return acos(prodScal/prodNorm)


# Qualité d'un seul triangle
def qualiteTriangle(a, b, c):
  """
  Entree: 3 angles a, b, et c en radians
  Sortie: float dans [0,1] en fonction du max(angle - angle_droit)
  """
  ecartAngle1 = abs(calculerAngle(a,b,c) - pi/2)
  ecartAngle2 = abs(calculerAngle(b,c,a) - pi/2)
  ecartAngle3 = abs(calculerAngle(c,a,b) - pi/2)
  ecartMax = max(ecartAngle1, ecartAngle2, ecartAngle3)

  return ecartMax / (pi/2)

# Qualité d'un ensemble de triangles 
def qualiteMaillageTri(triangles, points):
  """
  Entree: 
    - triangles de la forme Delaunay(points).simplices
    - np.array des points du maillage
  Sortie: float dans [0,1]

  Calcule la qualite de tous les triangles et en fait la moyenne arithmetique
  """
  sommeQualite = 0.0

  for triangle in triangles:
    sommeQualite += qualiteTriangle(points[triangle[0]], points[triangle[1]], points[triangle[2]])

  return sommeQualite / len(triangles)

def croisementMaillages(parent1, parent2):
  """
  Entree: 2 ensembles de points correspondant a 2 maillages
  Sortie: 1 ensemble de points correspondant a un maillage

  - on prend le point le plus a gauche (x min) et le plus a droite (x max),
  on calcule le barycentre (milieu)
  - on prend tous les points de gauche pour parent1, les autres pour
  parent2
  - on fait l'union, et on ajoute/supprime des points random pour conserver 
  le meme nombre que les parents

  On peut faire le même decoupage verticalement

  Une autre façon de croiser c'est de prendre 1 point sur 2 dans chaque maillage
  """
  # On cherche le point le plus a gauche
  frontiere = (np.amin(parent1, axis=0) + np.amax(parent1, axis=0)) / 2 
  xFrontiere = frontiere[0]

  enfant = parent1
  i = 0

  for point in parent1:
    if point[0] <= xFrontiere and i < len(enfant):
      enfant[i] = point
      i = i + 1

  for point in parent2:
    if point[0] > xFrontiere and i < len(enfant):
      enfant[i] = point
      i = i + 1
  
  return enfant




###############################################################################
# EXECUTION
###############################################################################

# Nombre de points du contour du cercle par quadrant
resolutionQuadrant = 5
contourCercle = genererContourCercle(resolutionQuadrant)

# Nombre de points interieurs au cercle
nbPtsDansCercle = 25
interieurCercle = genererInterieurCercle(nbPtsDansCercle)
interieurC2 = genererInterieurCercle(nbPtsDansCercle)

# Tous les points regroupes dans un tableau
pointsCercle = np.concatenate((contourCercle, interieurCercle), axis=0)
ptsC2 = np.concatenate((contourCercle, interieurC2), axis=0)

pointsCercle = croisementMaillages(pointsCercle, ptsC2)

# Creation du maillage
maillageTriCercle = Delaunay(pointsCercle)

print "Qualite du maillage : "
print qualiteMaillageTri(maillageTriCercle.simplices, pointsCercle)

###############################################################################
# VISUEL
###############################################################################
plt.triplot(pointsCercle[:,0], pointsCercle[:,1], maillageTriCercle.simplices.copy())
plt.plot(pointsCercle[:,0], pointsCercle[:,1], 'o')
plt.show()
