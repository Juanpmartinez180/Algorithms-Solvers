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
N = 100     #Numero de discretizaciones
h = L/(N)   #Longitud de la discretizacion (espacio de grilla)

#----------TEMPORAL-------------




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


#--- DATA PRINT --------
plt.title("TP7-Ej5 - Finite difference method - Mixed boundary conditions")
plt.xlabel("Lenght [mm]")
plt.ylabel("Temperature [°C]")
plt.plot(x, u_an, label = "Exact Solution")
plt.plot(x, T[:,0], label = "Computed Solution")
plt.legend()
plt.show()