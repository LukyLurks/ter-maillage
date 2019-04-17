# -*- coding: utf-8 -*-

###############################################################################
# IMPORTS
###############################################################################

import matplotlib.pyplot as plt       # voir/tracer les figures
import numpy as np                    # manipuler les tableaux de points
import random                         # générer les points dans les figures
from scipy.spatial import Delaunay    # générer les maillages

###############################################################################
# METHODES
###############################################################################

## POLYGONE CONVEXE ##

#Fonction créant un point aléatoirement sur un segment
#Sous-fonction de genererContourPoly et genererCentrePoly
def randomPointSegment(A,B,points):
	#Vérifie si on ne divise pas par zéro pour le coefficient directeur du segment
	coefficient = 0.0 + (points[B][0]-points[A][0])
	if coefficient == 0:
		coefficient = 1
		C=points[A][0]
		D=random.uniform(points[A][1],points[B][1])
	else :
		coefficient = (points[B][1]-points[A][1])/coefficient
		b = points[A][1] - coefficient*points[A][0]
		C=random.uniform(points[A][0],points[B][0])
		D = coefficient*C+b
	return [C,D]

#Fonction de génération des points sur les côtés
def genererContourPoly(nbPointTour,points):
	#Contour
	j=0
	nbPoint = len(points)
	while j < nbPoint:
		if ( j == (nbPoint-1)):
			i = 0
			while i < nbPointTour:
				points = np.concatenate((points,[randomPointSegment(0,j,points)]), axis=0)
				i = i + 1
		else:
			i = 0
			while i < nbPointTour:
				points = np.concatenate((points,[randomPointSegment(j,j+1,points)]), axis=0)
				i = i + 1
		j=j+1
	return points

#Fonction de génération des points 
def genererCentrePoly(nbPointCentre, pointsContour):
  # Ancienne fonction, on la garde au cas où
  """  
  def genererCentrePoly(nbPointCentre, fichierPoints):
    #centre
    points = lireFichierPoints(fichierPoints)
    k=0
    while k < nbPointCentre:
      A= random.randint(0,len(points)-1)
      B= random.randint(0,len(points)-1)
      points = np.concatenate((points,[randomPointSegment(A,B,points)]), axis=0)
      k=k+1
    ecrireFichierPoints(points, fichierPoints)
  """
  ptsCentre = np.zeros(shape=(nbPointCentre, 2))
  for k in range(nbPointCentre):
		A= random.randint(0,len(pointsContour)-1)
		B= random.randint(0,len(pointsContour)-1)
		ptsCentre[k] = [A, B]

  return ptsCentre

def afficherMaillage(points):
	#Code pour le afficherMaillage
	tri = Delaunay(points)
	plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
	plt.plot(points[:,0], points[:,1], 'o')
	plt.show()

###############################################################################
# Exemple de tracé
###############################################################################

## Paramètres ##
#points est initialisé avec les sommets de la figure
points = np.array([[-20.0, -10.0], [-30.0, 40.0],[-10.0,60.0],[50.0, 10.0],[30.0, -15.0]])

#nombre de points générés sur chaque segment 
nbPointTour = 5

#nombre de points générés dans la figure
nbPointCentre = 20

## Exécution ##
contourPoly = genererContourPoly(nbPointTour,points)
centrePoly = genererCentrePoly(nbPointCentre, contourPoly)
pointsPoly = np.concatenate((contourPoly, centrePoly), axis=0)
afficherMaillage(pointsPoly)

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

###############################################################################

from math import pi, cos, sin, acos   # générer les points du cercle

###############################################################################
# METHODES
###############################################################################

## CERCLE DISCRET ##

# Génère les points formant le contour du cercle
def genererContourCercle(resolution):
  """
  Entrée: résolution (nbre de pts de la discrétisation du cercle)
  Sortie: numpy.array contenant l'ensemble points de la
  discrétisation du cercle

  On génère les points du contour sur un quadrant de facon régulière, 
  puis on place les symétriques en manipulant les coordonnées:
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
  C'est juste pour économiser du temps de calcul sur les cos() et sin(),
  peut-être pas un gain significatif
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
  Entrée: intervalle d'angle pour des coordonnées polaires
  Sortie: coordonnées d'un point quelconque sur une portion du disque unité
  """
  angle = random.uniform(angleMin, angleMax)
  distance = random.random()
  return [distance * cos(angle), distance * sin(angle)]

# Génère des points aléatoirement dans un disque
def genererInterieurCercle(nbPointsInterieurs):
  # On initialise le tableau qui contiendra les points avec des zéros
  ptsInterieurs = np.zeros(shape=(nbPointsInterieurs, 2))

  for i in range(nbPointsInterieurs):
    ptsInterieurs[i] = genererPointInterieur(0, 2*pi)

  return ptsInterieurs

###############################################################################
# Exemple de tracé
###############################################################################

## Paramètres ##
# Nombre de points du contour du cercle par quadrant
resolutionQuadrant = 5
# Nombre de points intérieurs au cercle
nbPtsDansCercle = 25

## Exécution ##
contourCercle = genererContourCercle(resolutionQuadrant)
centreCercle = genererInterieurCercle(nbPtsDansCercle)
pointsCercle = np.concatenate((contourCercle, centreCercle), axis=0)
afficherMaillage(pointsCercle)

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

###############################################################################
# METHODES
###############################################################################

# Sous-fonction pour qualiteTriangle
def calculerAngle(a, b, c):
  """
  Entrée: 3 points A[xa, ya], B[xb, yb], C[xc, yc]
  Sortie: angle (AB, AC) en radians

  - On utilise les coordonnées de facon vectorielle: 
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
  Entrée: 3 angles a, b, et c en radians
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
  Entrée: 
    - triangles de la forme Delaunay(points).simplices
    - np.array des points du maillage
  Sortie: float dans [0,1]

  Calcule la qualité de tous les triangles et en fait la moyenne arithmétique
  """
  sommeQualite = 0.0

  for triangle in triangles:
    sommeQualite += qualiteTriangle(points[triangle[0]], points[triangle[1]], points[triangle[2]])

  return sommeQualite / len(triangles)

###############################################################################
# TESTS SUR LES EXEMPLES
###############################################################################


maillageExemplePoly = Delaunay(pointsPoly)
maillageExempleCercle = Delaunay(pointsCercle)

print "Qualite du maillage pour le polygone : "
print qualiteMaillageTri(maillageExemplePoly.simplices, pointsPoly)

print "Qualité du maillage pour le cercle : "
print qualiteMaillageTri(maillageExempleCercle.simplices, pointsCercle)

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################