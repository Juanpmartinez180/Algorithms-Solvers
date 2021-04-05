import numpy as np 
import matplotlib.pyplot as plt

#------------- LECTURA DATOS-------------------------
#---NODOS---
file = 'nodos.txt'   #Abro el archivo de datos

data = np.loadtxt(file, delimiter='\t', skiprows=2)     #Asigno los datos a una matriz

x = data[:,1]   #Coordenadas X-nodo en vector
y = data[:,2]   #Coordenadas Y-nodo en vector

nnod = len(x)   #Numero de nodos

#---ELEMENTOS---
file = 'elementos.txt'   #Abro el archivo de datos

data = np.loadtxt(file, delimiter='\t', skiprows=2)     #Asigno los datos a una matriz

N1 = data[:,1]      #Nodo 1 elemento i
N2 = data[:,2]      #Nodo 2 elemento i
N3 = data[:,3]      #Nodo 3 elemento i
Nmat = data[:,4]     #Material elemento i
nelem = len(N1)     #Numero de elementos

N = np.matrix([N1,N2,N3])


#---CONDICIONES DE DIRICHLET---
file = 'dirichlet_bc.txt'   #Abro el archivo de datos
data = np.loadtxt(file, delimiter='\t', skiprows=2)     #Asigno los datos a una matriz

if data.size == 0 :     #Si no hay valores en el archivo
    ndir = 0                #Numero de nodos con cond. de Dirichlet
else:
    N_dir = data[:,0]          #Nodo con cond. de Dirichlet
    R_dir = data[:,1]          #Cond. Dirichlet nodo nd
    ndir = len(N_dir)          #Numero de nodos con cond. de Dirichlet

#---CONDICIONES DE NEUMANN---
file = 'neumann_bc.txt'   #Abro el archivo de datos
data = np.loadtxt(file, delimiter='\t', skiprows=2)     #Asigno los datos a una matriz
  
if data.size == 0 :         #Si no hay valores en el archivo
    nneu = 0                    #numero de lados con cond. de Neumann
else:
    if data.size <= 3:
        N_neu1 = data[0]          #Nodo1 con cond. de Neumann
        N_neu2 = data[1]          #Nodo2 con cond. de Neumann
        R_neu = data[2]           #Condicion de Neumann en los nodos N1-N2   
        nneu = 1         #Numero de lados con cond. de Neumann                 
    else:
        N_neu1 = data[:,0]          #Nodo1 con cond. de Neumann
        N_neu2 = data[:,1]          #Nodo2 con cond. de Neumann
        R_neu = data[:,2]           #Condicion de Neumann en los nodos N1-N2
        nneu = len(N_neu1)          #Numero de lados con cond. de Neumann
    N_neu = np.matrix([N_neu1,N_neu2])  #Matriz de nodos

#---FUENTE DE CALOR Q---
file = 'fuente_Q.txt'   #Abro el archivo de datos
data = np.loadtxt(file, delimiter='\t', skiprows=2)     #Asigno los datos a una matriz

if data.size == 0:      #Si no hay valores en el archivo
    nq = 0                  #Numero de elementos con fuente de calor
else:
    E_q = data[:,0]         #Elemento con fuente de calor
    R_q = data[:,1]         #Fuente de calor sobre elemento i
    nq = len(E_q)           #Numero de elementos con fuente de calor

#---MATERIALES ELEMENTOS---
file = 'materiales.txt' #Abro el archivo de datos
data = np.loadtxt(file, delimiter='\t', skiprows=2)     #Asigno los datos a una matriz

if data.size <= 2:      #Si hay un solo material
    rk_x = [data[0]]          #Constante K en x material i
    rk_y = [data[1]]          #Constante K en y material i
else:
    rk_x = data[:,0]        #Constante K en x material i
    rk_y = data[:,1]        #Constante K en y material i   
#-----------------------------------------------------------------------------#
#-----------------------------------------------------------------------------#
#----Cálculo de area elemental---
def area(x1,x2,x3,y1,y2,y3):
    
    M = [[1, x1, y1],           #Creo la matriz a operar
         [1, x2, y2],
         [1, x3, y3]]
    
    a = 0.5*np.linalg.det(M)    #Calculo área con el determinante
        
    return(a)
#----Cálculo de Bi--------------
def Bi(y1,y2,y3):  
     
    b = np.zeros((3))     #Creo vector de variable y calculo los valores del mismo
    b[0]=y2-y3
    b[1]=y3-y1
    b[2]=y1-y2
  
    return(b)
#----Cálculo de Ci-------------
def Ci(x1,x2,x3):
    c = np.zeros((3))     #Creo vector de variable y calculo los valores del mismo
    c[0]=x3-x2
    c[1]=x1-x3
    c[2]=x2-x1  
    
    return(c)
#-----------------------------------------
#----MATRICES ELEMENTALES Y ENSAMBLADO----
A = np.zeros((nnod, nnod))          #Vector de coeficientes 
KG = np.matrix(A)                   #Matriz de coeficientes Global
KE = np.matrix(np.zeros((3, 3)))    #Matriz de coeficientes elemental

for i in range (0, nelem):
    n1 = int(N[0,i])
    n2 = int(N[1,i])
    n3 = int(N[2,i])
    
    x1 = x[n1-1]        #Determino las coordenadas de los nodos del elemento i
    y1 = y[n1-1]
    x2 = x[n2-1]
    y2 = y[n2-1]
    x3 = x[n3-1]
    y3 = y[n3-1]
    
    Area =  area(x1,x2,x3,y1,y2,y3)     #Calculo el área del elemento i
    
    b = Bi(y1,y2,y3)        #Obtengo los coeficientes b y c
    c = Ci(x1,x2,x3)

    for j in range(0,3):        #Calculo la matriz local
        for k in range(0,3):
            KE[j,k] = ( b[j]*rk_x[int(Nmat[i])-1]*b[k] + c[j]*rk_y[int(Nmat[i])-1]*c[k] )/(4*Area) 
        
    for j in range(0,3):        #Ensamblo la matriz local en la matriz global
        for k in range(0,3):          
            KG[int(N[j,i])-1,int(N[k,i])-1] =  KG[int(N[j,i]-1),int(N[k,i])-1] + KE[j,k]   
      
#-----------------------------------------
#----TERMINO INDEPENDIENTE Y ENSAMBLADO----  
FG = np.zeros((nnod, 1))        #Vector de coeficientes Globales
FE = np.zeros((3,1))            #Vector de coeficientes elemental

#---Fuente de calor----
for i in range (0, nq):     #Itero todos los terminos con fuente
    iel = int(E_q[i])           #Indice elemento que contiene fuente
 
    n1 = int(N[0,iel-1])        #Nodos del elemento con fuente
    n2 = int(N[1,iel-1])
    n3 = int(N[2,iel-1])
    
    x1 = x[n1-1]                #Determino las coordenadas de los nodos del elemento i
    y1 = y[n1-1]
    x2 = x[n2-1]
    y2 = y[n2-1]
    x3 = x[n3-1]
    y3 = y[n3-1]
    
    Area =  area(x1,x2,x3,y1,y2,y3)     #Calculo el área del elemento i   
    
    for j in range(0,3):
        FE[j] = ( Area*R_q[i] ) / 3
        
    for j in range(0,3):
        a = int(N[j,iel-1])
        FG[int(N[j,iel-1])-1] = FG[int(N[j,iel-1])-1] + FE[j]
    
#---Condiciones de neumann---
for i in range (0, nneu):       #Itero para todos los lados con cond. de neumann
    n1 = int(N_neu[0,i])            #Nodos con cond. de Neum.
    n2 = int(N_neu[1,i])

    x1 = x[n1-1]                    #Determino las coordenadas de los nodos del lado con cond. de Neum.
    y1 = y[n1-1]
    x2 = x[n2-1]
    y2 = y[n2-1]

    longitud = np.sqrt( ((x2-x1)**2) + ((y2-y1)**2) )   #Longitud del elemento con cond. de neum.
    
    for j in range(0,2):                    #Ensamblo termino indep. elemental
        FE[j] = ( longitud*R_neu[i] ) / 2
        
    for j in range(0,2):                    #Ensamblo terminos indep. globales
        FG[int(N_neu[j,i])-1] = FG[int(N_neu[j,i])-1] + FE[j]


#---Condiciones de Dirichlet---
for i in range (0, ndir):       #Itero para todos los nodos con cond. de Dirichlet
    idir = int(N_dir[i])            #Nodo con cond. de Dir.
    KG[idir-1,idir-1] = 10**10      #Modifico matriz Global
    FG[idir-1] = R_dir[i]*(10**10)  #Termino indep. Globlal

#--------------------------------------------
#----Resolucion del sistema de ecuaciones---- 

T = np.zeros((nnod,1))

T = np.linalg.solve(KG, FG)

#-----POST PROCESADO DE LOS RESULTADOS------
print(T)

f_x = np.zeros((nelem, 3))
f_y = np.zeros((nelem, 3))

for i in range (0, nelem):
    n1 = int(N[0,i])
    n2 = int(N[1,i])
    n3 = int(N[2,i])
    x1 = x[n1-1]
    x2 = x[n2-1]
    x3 = x[n3-1]
    y1 = y[n1-1]
    y2 = y[n2-1]
    y3 = y[n3-1]
    
    Area = area(x1,x2,x3,y1,y2,y3)
    b = Bi(y1,y2,y3)        #Obtengo los coeficientes b y c
    c = Ci(x1,x2,x3)
    
    dT_x = b/(2*Area)
    dT_y = c/(2*Area)
    f_x[i,:] = -rk_x[int(Nmat[i])-1]*dT_x
    f_y[i,:] = -rk_y[int(Nmat[i])-1]*dT_y
    
s =1 