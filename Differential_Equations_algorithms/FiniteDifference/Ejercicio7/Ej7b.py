import numpy as np
import matplotlib.pyplot as plt

#---- Defino constantes del ejercicio -----
V = 1*(10**-6)  #Viscosidad [m2/s]
RO = 1000       #Densidad [kg/m3]
R1 = 0.0015     #Radio interior cilindro
R2 = 0.0079     #Radio cilindro exterior

L = R2-R1          #Longitud del dominio



#---- Condiciones de contorno -----
U0 = 0      #Condicion inicial del dominio (t=0)

#---- Discretizacion del dominio ----
#----------ESPACIAL-------------   
N = 40      #Numero de discretizaciones
h = L/N     #Longitud de la discretizacion (espacio de grilla)

#----------TEMPORAL-------------
t_inic = 0
t_fin = 20        #Tiempo final a calcular [s]
K_t = 2.5*(10**-4)    #Paso temporal
N_t = round(t_fin / K_t)   #Iteraciones a resolver 

#----- solucion analitica
def an_sol(x):
    
    sol = -166667*(x**2 - 0.02*x - 0.00204545)
    return(sol)

#----- Velocidades angulares ----
t = np.linspace(0,t_fin,N_t)    #Vector de tiempo
w = np.linspace(0,t_fin,N_t)    #Vector de velocidad angular vs tiempo

for i in range(0,len(t)):       #Defino las velocidades en funcion del tiempo
    
    if( t[i]<= 0.8 ):               #Para t<0.8s
        w[i] = 5*t[i]/2
        pass
    elif( t[i]>0.8 and t[i]<20 ):   #Para 0.8s<t<20s
        w[i] = 2
        pass
    else:                           #Para t>20s
        w[i] = 2 - 5*(t[i]-20)/2
        pass

w = w*2*np.pi
#--- MAIN --------------    
x = np.linspace(R1, R2, N+1)  #Puntos de la grilla a calcular
u_an = np.zeros((len(x)))   #Vector solucion analitica

C = np.zeros((N+1, N+1))    #Vector de coeficientes 
A = np.matrix(C)            #Matriz de coeficientes
B = np.zeros((N+1,1))       #Vector de variables independientes


##------------------------Temporal-------------------

D = np.zeros((N+1,1))       #Vector de solucion en t_i
D_1 = np.zeros((N+1,1))     #Vector de solucion en t_i+1

L_0 = np.matrix(C)          #Matriz de coeficientes 

#--- Cargo condiciones iniciales en el dominio---
for i in range (1,N):
    D[N] = U0

D[0] = w[0]*t[t_inic]   #Condicion inicial de borde0 
D[N] = w[0]*t[t_inic]   #Condicion inicial de bordeN

#--- Matriz de coeficientes 

for i in range (0, N+1):    #Lleno la matriz de coef.
    
    if i == 0:
        L_0[i,i] = 1                 #Coeficientes del metodo a resolver (D2 order 2)

        pass
    elif i == N:
        L_0[i,i] = 1 

        pass
    else:
        L_0[i,i] = 1/(K_t*V)             
        pass

#----- Resuelvo el problema temporal-----

values = [6]     #Valores a imprimir

s = 1
for k in range (0, len(values)):
     
    N_t = round(values[k] / K_t)
    

    for i in range(0,N_t): 
        
        D_1[0] = w[i]*R1
        
        D_1[N] = w[i]*R2
        
        for j in range(1,N):        #Variables dependientes en el dominio
            
            D_1[j] = ( D[j+1] * ( (h**-2) + (1/(2*x[j]*h)) )          
                    - D[j] * ( (2/(h**2)) + (x[j]**-2) - (1/(K_t*V)) ) 
                    + D[j-1] * ( (h**-2) - (1/(2*x[j]*h)) )   )
            
            
        P = np.linalg.solve( L_0, D_1 )     #Resuelvo el sistema (determino valores en t=tn+1)  
        
        a = np.amax(P - D)
        
        D = P
        
        error = np.abs( (a) /K_t) 
        
        if (error < 1*(10**-4) and s == 1):
            print("YESSS")
            print(t[i])
            

    plt.plot(x, D[:,0])

    
#--- DATA PRINT --------
plt.title("TP7-Ej7 - Finite difference method - Mixed boundary conditions")
plt.xlabel("Lenght [mm]")
plt.ylabel("Temperature [°C]")
#plt.plot(x, u_an, label = "Exact Solution")
#plt.plot(x, T[:,0], label = "Computed Solution")
#plt.plot(x, D[:,0], label = "Computed Solution")
#plt.legend()
plt.show()