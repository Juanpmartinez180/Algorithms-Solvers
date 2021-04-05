import numpy as np
import matplotlib.pyplot as plt

#---- Defino constantes del ejercicio -----
K = 30          #Coeficiente de transmision [W/mK]
Ro = 5*(10**6)  
Hc = 1100       #Convectividad [W/m2K]
Q1 = 1*(10**7)  #Fuente de calor [W/m3]
Q2 = 2*(10**7)  #Fuente de calor 2
L = 0.02          #Longitud del dominio

#---- Condiciones de contorno -----
T_inf = 250     #Temperatura del medio convectivo

#---- Discretizacion del dominio ----
#----------ESPACIAL-------------
h = 1*(10**-3)      #Longitud de la discretizacion (espacio de grilla)
N = round(L/h)      #Numero de discretizaciones

#----------TEMPORAL-------------
t_fin = 500        #Tiempo final a calcular [s]
#K_t = 0.04    #Paso temporal
K_t = 0.05
N_t = round(t_fin / K_t)   #Iteraciones a resolver 

#----- solucion analitica
def an_sol(x):
    
    sol = -166667*(x**2 - 0.02*x - 0.00204545)
    return(sol)


#--- MAIN --------------    
x = np.linspace(0, L, N+1)  #Puntos de la grilla a calcular
u_an = np.zeros((len(x)))   #Vector solucion analitica

C = np.zeros((N+1, N+1))    #Vector de coeficientes 
A = np.matrix(C)            #Matriz de coeficientes
B = np.zeros((N+1,1))       #Vector de variables independientes

#--- Matriz de coeficientes 

for i in range (0, N+1):    #Lleno la matriz de coef.
    
    if i == 0:
        A[i,i] = 2*(1+(Hc*h/K))                 #Coeficientes del metodo a resolver (D2 order 2)
        A[i,i+1] = -2
        
        B[i] = (Q1*(h**2) + 2*Hc*h*T_inf)/K          #Valor de la variable independiente
        pass
    elif i == N:
        A[i,i] = 2*(1+(Hc*h/K))  
        A[i,i-1] = -2
     
        B[i] = (Q1*(h**2) + 2*Hc*h*T_inf)/K       
        pass
    else:
        A[i,i] = -2               
        A[i,i-1] = 1
        A[i,i+1] = 1
     
        B[i] = -(h**2)*Q1/(K)          
        pass
    
u_an = an_sol(x)

T = np.linalg.solve(A,B)    #Resuelvo el sistema de ecuaciones

##------------------------Temporal-------------------

D = np.zeros((N+1,1))       #Vector de solucion en t_i
D_1 = np.zeros((N+1,1))     #Vector de solucion en t_i+1

L_0 = np.matrix(C)

D = T

W = (Ro*(h**2))/(K_t*K)


#--- Matriz de coeficientes 

for i in range (0, N+1):    #Lleno la matriz de coef.
    
    if i == 0:
        L_0[i,i] = W + (h*Hc/K)                 #Coeficientes del metodo a resolver (D2 order 2)
        L_0[i,i+1] = -1
        pass
    elif i == N:
        L_0[i,i] = W + (h*Hc/K)  
        L_0[i,i-1] = -1
              
        pass
    else:
        L_0[i,i] = 2*W              
        L_0[i,i+1] = -1  
        L_0[i,i-1] = -1  
        pass

#----- Resuelvo el problema temporal-----
G = t_fin/1
R = G
while R <= t_fin:
    
    N_t = round(G / K_t)
     
    for i in range(0,N_t): 
        
        D_1[0] = D[1] + D[0]*(W-2-(h*Hc/K)) + 2*h*Hc*T_inf/K + Q2*(h**2)/K
          
        D_1[N] = D[N-1] + D[N]*(W-2-(h*Hc/K)) + 2*Hc*h*T_inf/K + Q2*(h**2)/K
        
        for j in range(1,N):
            
            D_1[j] = D[j+1] +2* D[j]*(W-2) + D[j-1] + Q2*2*(h**2)/K
            
            
        D = np.linalg.solve( L_0, D_1 )               #Resuelvo el sistema (determino valores en t=tn+1)   

    plt.plot(x, D[:,0])
    
    R = R+G

#--- DATA PRINT --------
plt.title("TP7-Ej5 - Finite difference method - Mixed boundary conditions")
plt.xlabel("Lenght [mm]")
plt.ylabel("Temperature [°C]")
plt.plot(x, u_an, label = "Exact Solution")
plt.plot(x, T[:,0], label = "Computed Solution 1")
plt.plot(x, D[:,0], label = "Computed Solution 2")
plt.legend()
plt.show()
