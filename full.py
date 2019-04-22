# -*- coding: utf-8 -*-


## Polygone convexe
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

# Génère un polygone avec tous les points
def genererPoly(nbPointTour, nbPointCentre, sommets):
    contourPoly = genererContourPoly(nbPointTour, sommets)
    centrePoly = genererCentrePoly(nbPointCentre, contourPoly)
    pointsPoly = np.concatenate((contourPoly, centrePoly), axis=0)
    return pointsPoly

def afficherMaillage(points):
	#Code pour le afficherMaillage
	tri = Delaunay(points)
	plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
	plt.plot(points[:,0], points[:,1], 'o')
	plt.show()

"""
###############################################################################
# Exemple de tracé
###############################################################################

## Paramètres ##
#sommetsPoly est initialisé avec les sommets de la figure
sommetsPoly = np.array([[-20.0, -10.0], [-30.0, 40.0],[-10.0,60.0],[50.0, 10.0],[30.0, -15.0]])

#nombre de points générés sur chaque segment 
nbPointTour = 5

#nombre de points générés dans la figure
nbPointCentre = 20

## Exécution ##
pointsPoly = genererPoly(nbPointTour, nbPointCentre, sommetsPoly)
afficherMaillage(pointsPoly)
"""

## Cercle
###############################################################################
# IMPORTS
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

# Génere le cercle avec tous ses points
def genererCercle(resolutionQuadrant, nbPointsInterieurs):
    contourCercle = genererContourCercle(resolutionQuadrant)
    centreCercle = genererInterieurCercle(nbPointsInterieurs)
    pointsCercle = np.concatenate((contourCercle, centreCercle), axis=0)
    return pointsCercle
    
"""
###############################################################################
# Exemple de tracé
###############################################################################

## Paramètres ##
# Nombre de points du contour du cercle par quadrant
resolutionQuadrant = 5
# Nombre de points intérieurs au cercle
nbPtsDansCercle = 25

## Exécution ##
pointsCercle = genererCercle(resolutionQuadrant, nbPtsDansCercle)
afficherMaillage(pointsCercle)
"""

## Qualité
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
def qualiteMaillage(triangles, points):
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

"""
###############################################################################
# TESTS SUR LES EXEMPLES
###############################################################################


maillageExemplePoly = Delaunay(pointsPoly)
maillageExempleCercle = Delaunay(pointsCercle)

print "Qualite du maillage pour le polygone : "
print qualiteMaillage(maillageExemplePoly.simplices, pointsPoly)

print "Qualité du maillage pour le cercle : "
print qualiteMaillage(maillageExempleCercle.simplices, pointsCercle)
"""

## Croisement
# Croisement gauche-droite
def croisemtGD(pointsParent1, pointsParent2):
  """
  Entrée: 2 ensembles de points correspondant à 2 maillages
  Sortie: 1 ensemble de points correspondant à un maillage
  
  L'enfant aura le même nombre de points que les parents.

  - on prend le point le plus a gauche (x min) et le plus a droite (x max),
  on calcule le barycentre (milieu)
  - on prend tous les points de gauche pour parent1, les autres pour
  parent2
  - on fait l'union, et on ajoute/supprime des points random pour conserver 
  le meme nombre que les parents
  """

  # Calcul de la frontière
  frontiere = (np.amin(pointsParent1, axis=0) + np.amax(pointsParent1, axis=0)) / 2 
  abscisseFrontiere = frontiere[0]

  # On initialise l'enfant à un parent pour être sûr 
  # d'avoir le bon nombre de points
  pointsEnfant = pointsParent1
  i = 0

  for point in pointsParent1:
    if point[0] <= abscisseFrontiere and i < len(pointsEnfant):
      pointsEnfant[i] = point
      i = i + 1

  for point in pointsParent2:
    if point[0] > abscisseFrontiere and i < len(pointsEnfant):
      pointsEnfant[i] = point
      i = i + 1
  
  return pointsEnfant

"""
###############################################################################
# TESTS SUR LES EXEMPLES
###############################################################################

## Polygones convexes ##
# On génère 2 parents
#sommetsCarre = np.array([[-10.0, -10.0], [-10.0, 10.0],[10.0,10.0],[10.0, -10.0]])
sommetsCarre = np.array([[-20.0, -10.0], [-30.0, 40.0],[-10.0,60.0],[50.0, 10.0],[30.0, -15.0]])

pointsCarre1 = genererPoly(nbPointTour, nbPointCentre, sommetsCarre)
maillageCarre1 = Delaunay(pointsCarre1)

pointsCarre2 = genererPoly(nbPointTour, nbPointCentre, sommetsCarre)
maillageCarre2 = Delaunay(pointsCarre2)

# On calcule leur qualité
print "Qualité parent carré 1 : ",qualiteMaillage(maillageCarre1.simplices,pointsCarre1)
print "Qualité parent carré 2 : ",qualiteMaillage(maillageCarre2.simplices,pointsCarre2)

# On les croise et on observe l'enfant
enfantCarre = croisemtGD(pointsCarre1, pointsCarre2)
maillageEnfantCarre = Delaunay(enfantCarre)
print "Qualité enfant carré : ",qualiteMaillage(maillageEnfantCarre.simplices,enfantCarre)
afficherMaillage(enfantCarre)

## Cercles ##
# On génère les 2 parents
pointsCerc1 = genererCercle(resolutionQuadrant, nbPtsDansCercle)
maillageCerc1 = Delaunay(pointsCerc1)

pointsCerc2 = genererCercle(resolutionQuadrant, nbPtsDansCercle)
maillageCerc2 = Delaunay(pointsCerc2)


# On calcule leur qualité
print "Qualité parent cercle 1 : ",qualiteMaillage(maillageCerc1.simplices,pointsCerc1)
print "Qualité parent cercle 2 : ",qualiteMaillage(maillageCerc2.simplices,pointsCerc2)

# On les croise et on observe l'enfant
enfantCerc = croisemtGD(pointsCerc1, pointsCerc2)
maillageEnfantCerc = Delaunay(enfantCerc)
print "Qualité enfant cercle : ",qualiteMaillage(maillageEnfantCerc.simplices,enfantCerc)
afficherMaillage(enfantCerc)
"""

## Classe et conservation des données
###############################################################################
# CLASSE
###############################################################################

## Définition ##
class Maillage(object):
    def __init__(self, points):
        self.points = points
        self.tri = Delaunay(points).simplices
        self.qualite = qualiteMaillage(self.tri, points)
        
    ## Méthodes ##
    # Ecrit les points du maillage dans un fichier .txt
    def ecrireFichierPoints(self, fichier):
	w=0
	while w<len(self.points):
		temp0 = str(self.points[w][0])
		temp1 = str(self.points[w][1])
		point = "["+temp0+","+temp1+"]\n"
		fichier.write(point)
		w=w+1
        
    # Ecrit la qualité du maillage dans un fichier .txt
    def ecrireFichierQualite(self, fichier):
        fichier.write(str(self.qualite)+'\n')
        
###############################################################################
# METHODES
###############################################################################

    
# Lit un fichier .txt contenant des points et les met dans un np.array
# On utilisera cette fonction pour exploiter les résultats
def lireFichierPoints(filename, taillePop):
	points = np.array([[0.0, 0.0]])
	fichier = open(filename, "r")
	for ligne in fichier:
		temp0 = ""
		temp1 = ""
		indice = 0
		for c in ligne:
			if (indice == 0 and (c == "-" or c == "." or c.isdigit() == True )):
				temp0 = temp0+c
			elif c == ",":
				indice = 1
			elif (indice == 1 and (c == "-" or c == "." or c.isdigit() == True)):
				temp1 = temp1+c
		points = np.concatenate((points,[[float(temp0),float(temp1)]]), axis=0)
	points = np.delete(points,0,0)
	fichier.close()
	return np.split(points,taillePop,axis=0)

def lireFichierQualite(filename):
    fichier = open(filename, "r")
    qualites = [q for q in fichier]
    fichier.close()
    return qualites

"""
###############################################################################
# EXEMPLE
###############################################################################


taillePopulation = 10

# On génère une population de maillages dans une liste
exemplePopulation = [
    Maillage(points=genererCercle(resolutionQuadrant, nbPtsDansCercle)) 
    for i in range(taillePopulation)
]

# On la trie par ordre décroissant de qualité (1er élément: meilleur)
exemplePopulation.sort(key=lambda maillage: maillage.qualite, reverse=True)

# On écrit les résultats dans un fichier
testPoints = open("testPoints.txt", "w")
testQualite = open("testQualite.txt", "w")
for maillage in exemplePopulation:
    maillage.ecrireFichierPoints(testPoints)
    maillage.ecrireFichierQualite(testQualite)
testPoints.close()
testQualite.close()

# On peut lire ces résultats
variablePoints = lireFichierPoints("testPoints.txt", taillePopulation)
testQualite = lireFichierQualite("testQualite.txt")

# Points et qualité du 3e maillage, par exemple
print variablePoints[2]
print testQualite[2]
"""

## Algo complet
###############################################################################
# PARAMETRES
###############################################################################

## Algorithme ##
# Nombre de générations
nbGen = 500
# Taille de la population par figure
taillePop = 10

## Figures ##
# Polygone
ptsParCote = 8
ptsDansPoly = 30
sommets = np.array([[-20.0, -10.0], [-30.0, 40.0],[-10.0,60.0],[50.0, 10.0],[30.0, -15.0]])
ptsParPoly = ptsParCote * len(sommets) + ptsDansPoly

# Cercle
resQuart = 10
ptsDansCercle = 50
ptsParCercle = 4 * resQuart + ptsDansCercle


###############################################################################
# ALGORITHME EVOLUTIONNAIRE
###############################################################################

# On initialise les populations
popPoly = [
    Maillage(points=genererPoly(ptsParCote, ptsDansPoly, sommets)) 
    for i in range(taillePop)
]

popCercle = [
    Maillage(points=genererCercle(resQuart, ptsDansCercle)) 
    for i in range(taillePop)
]

# Début algo
for g in range(nbGen):
    # Trie la population
    popPoly.sort(key=lambda x: x.qualite, reverse=True) #bug après 1ere iteration
    popCercle.sort(key=lambda x: x.qualite, reverse=True)
    # Garde les meilleurs
    popPoly = popPoly[:taillePop]
    popCercle = popCercle[:taillePop]
    
    # On aura 1 fichier par génération, donc on formate bien les noms 
    nomPtsPoly = "ptsPoly_top" + str(taillePop) + "_gen" + str(g) + ".txt"
    nomQualPoly = "qualPoly_top" + str(taillePop) + "_gen" + str(g) + ".txt"
    nomPtsCercle = "ptsCercle_top" + str(taillePop) + "_gen" + str(g) + ".txt"
    nomQualCercle = "qualCercle_top" + str(taillePop) + "_gen" + str(g) + ".txt"
    # Ouvre les fichiers
    ptsPoly = open(nomPtsPoly, "w")
    qualPoly = open(nomQualPoly, "w")
    ptsCercle = open(nomPtsCercle, "w")
    qualCercle = open(nomQualCercle, "w")
    # Ecrit dans les fichiers
    for poly in popPoly:
        poly.ecrireFichierPoints(ptsPoly)
        poly.ecrireFichierQualite(qualPoly)
    for cercle in popCercle:
        cercle.ecrireFichierPoints(ptsCercle)
        cercle.ecrireFichierQualite(qualCercle)
    # Ferme les fichiers
    ptsPoly.close()
    qualPoly.close()
    ptsCercle.close()
    qualCercle.close()
        
    # Croise et ajoute les enfants à la population
    for i in range(0, taillePop, 2):
        tempPtsPoly = croisemtGD(popPoly[i].points, popPoly[i+1].points)
        tempPoly = Maillage(points=tempPtsPoly)
        popPoly.append(tempPoly)
        tempPtsCercle = croisemtGD(popCercle[i].points, popCercle[i+1].points)
        tempCercle = Maillage(points=tempPtsCercle)
        popCercle.append(tempCercle)
        
# Affiche le meilleur maillage de la 1ere et la dernière génération

# 1ere gen
ptsPolyOld = lireFichierPoints("ptsPoly_top10_gen0.txt", taillePop)
qPolyOld = lireFichierQualite("qualPoly_top10_gen0.txt")
ptsCercleOld = lireFichierPoints("ptsCercle_top10_gen0.txt", taillePop)
qCercleOld = lireFichierQualite("qualCercle_top10_gen0.txt")
# dernière gen
ptsPolyNew = lireFichierPoints("ptsPoly_top10_gen99.txt", taillePop)
qPolyNew = lireFichierQualite("qualPoly_top10_gen99.txt")
ptsCercleNew = lireFichierPoints("ptsCercle_top10_gen99.txt", taillePop)
qCercleNew = lireFichierQualite("qualCercle_top10_gen99.txt")

afficherMaillage(ptsPolyOld[0])
print "Qualité meilleur poly gen 0 : ",qPolyOld[0]
afficherMaillage(ptsCercleOld[0])
print "Qualité meilleur cercle gen 0 : ",qCercleOld[0]

afficherMaillage(ptsPolyNew[0])
print "Qualité meilleur poly gen 99 : ",qPolyNew[0]
afficherMaillage(ptsCercleNew[0])
print "Qualité meilleur cercle gen 99 : ",qCercleNew[0]