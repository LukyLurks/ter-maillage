import matplotlib.pyplot as plt
import numpy as np
import random
#points est initialise avec les sommets de la figure
points = np.array([[-20.0, -10.0], [-30.0, 40.0],[-10.0,60.0],[50.0, 10.0],[30.0, -15.0]])

#nombre de point genere sur chaque segment 
nbPointTour = 5 
#nombre de point genere dans la figure
nbPointCentre = 20

#Fonction creant le point sur le milieu d'un segment
def Milieu(A,B):
	Cx = points[A][0]+points[B][0]
	Cx = Cx/2
	Cy = points[A][1]+points[B][1]
	Cy = Cy/2
	return [Cx,Cy]

#Fonction Test 	
def DoCarre(nbPointTour, nbPointCentre,points):
	i = 0
	while i < nbPointTour:
		j = 0
		taille = len(points)
		print "taille ",taille
		while j < taille:
			print "j ",j
			if ( j == taille-1):
				points = np.concatenate((points,[Milieu(0,j)]), axis=0)
			else:
				points = np.concatenate((points,[Milieu(j,j+1)]), axis=0)
			j=j+1
		i=i+1
		print "Points : ",points
	return points

#Fonction creant un point aleatoirement sur un segment
def RandomPoint(A,B,points):
	#Verifie si on ne divise pas par zero pour le coefficiant directeur du segment
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

#Fonction de generation des points 
def Generation(points,nbPointTour,nbPointCentre):
	#Contour
	j=0
	nbPoint = len(points)
	while j < nbPoint:
		if ( j == (nbPoint-1)):
			i = 0
			while i < nbPointTour:
				points = np.concatenate((points,[RandomPoint(0,j,points)]), axis=0)
				i = i + 1
		else:
			i = 0
			while i < nbPointTour:
				points = np.concatenate((points,[RandomPoint(j,j+1,points)]), axis=0)
				i = i + 1
		j=j+1
	#centre
	k=0
	print points
	while k < nbPointCentre:
		A= random.randint(0,len(points)-1)
		B= random.randint(0,len(points)-1)
		print "len : ",len(points), "  A: ",A,"  B:", B
		points = np.concatenate((points,[RandomPoint(A,B,points)]), axis=0)
		k=k+1
	return points
		



#Le Main de mon code
points = Generation(points,nbPointTour,nbPointCentre)


#Zone de test
#points = DoCarre(nbPointTour, nbPointCentre,points)

#print random.uniform(7.5,6.6)

#Code pour le visuel
from scipy.spatial import Delaunay
tri = Delaunay(points)
plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
plt.plot(points[:,0], points[:,1], 'o')
plt.show()









