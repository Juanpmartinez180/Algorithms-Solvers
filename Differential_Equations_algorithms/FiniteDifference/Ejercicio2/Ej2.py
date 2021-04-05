
import numpy as np
import matplotlib.pyplot as plt
#--- Propiedades del material----
L = 1           #Largo de la vara
K = 235         #Coef de transmision [W/mK]
Ro = 5*(10**6)  #Densidad*calor especifico

#--- Condiciones iniciales y de bordes ----
T0 = 30     #Temperatura extremo inicial [°C]
Tn = 30     #Temperatura extremo final [°C]
T_t0 = 50   #Temperatura en el instante inicial t=0 [°C]

#--- Discretizacion espacial y temporal ---
t_fin = 500         #Tiempo final a calcular [s]
K_t = 0.5           #Paso temporal
h = 9.7*(10**-3)    #Longitud de la discretizacion (espacio de grilla)
N = round(L/h)      #Numero de discretizaciones
N_t = round(t_fin / K_t)   #Iteraciones a resolver 

W = (K_t * K )/(Ro*h**2)    #Constante

#--- Funcion analitica ---
def an_sol(x,t) : 
    N = 10 #Iteraciones de la serie
    C1 = K/Ro
    C2 = np.pi / L
    
    sol = 0
    for i in range(1,N):
        sol = sol + np.exp(-t*C1*(i*C2)**2)*((1-(-1)**i)/(i*np.pi))*np.sin(C2*i*x)

    sol = T0 + 2*(T_t0-T0)*sol
    
    return (sol)

#--- MAIN --------------
      
x = np.linspace(0, L, N+1)  #Puntos de la grilla a calcular
u_an = np.zeros((len(x)))   #Vector solucion analitica

C = np.zeros((N+1, N+1))    #Vector de coeficientes  
M_k1 = np.matrix(C)         #Matriz de coeficientes
M_k2 = np.matrix(C)         #Matriz de coeficientes
M_k12 = np.matrix(C)        #Matriz de coeficientes
T_n = np.zeros((N+1,1))     #Vector de variables independientes en tiempo t=tn

K1 = np.zeros((N+1,1))      #Vector valores de K1
K2 = np.zeros((N+1,1))      #Vector valores de K2

#-----Cargo condiciones iniciales y coef. en la matriz
for i in range(1, N):
    
    T_n[i] = T_t0       #Condiciones iniciales en en el dominio 
          
T_n[0] = T0       #Condicion de Dirichlet en T0
T_n[N] = Tn       #Condicion de Dirichlet en Tn


#""--- Lleno las matrices de coeficientes
#-----------------------------------------
#-M_K1 = Matriz de coef. para calcular K1
#-M_K2 = Matriz de coef. para calcular K2
#-M_K12 = Matriz de coef. para calcular K2(parametros independientes)
#-----------------------------------------

for i in range (0,N+1):
    if i == 0:
        M_k1[i,i] = 2       
        M_k1[i,i+1] = -5
        M_k1[i,i+2] = 4
        M_k1[i,i+3] = -1
        
        M_k2[i,i] = -1+(1/W)
        M_k2[i,i+1] = 5/2
        M_k2[i,i+2] = -2
        M_k2[i,i+3] = 1/2  
        
        M_k12[i,i] = (1/W)+2
        M_k12[i,i+1] = -7/2
        M_k12[i,i+2] = 3
        M_k12[i,i+3] = -3/2  
        pass
        
    elif i == N :
        M_k1[i,i] = 2
        M_k1[i,i-1] = -5
        M_k1[i,i-2] = 4
        M_k1[i,i-3] = -1
        
        M_k2[i,i] = -1+(1/W)
        M_k2[i,i-1] = 5/2
        M_k2[i,i-2] = -2
        M_k2[i,i-3] = 1/2 
        
        M_k12[i,i] = (1/W)+2
        M_k12[i,i-1] = -7/2
        M_k12[i,i-2] = 3
        M_k12[i,i-3] = -3/2         
        pass
    
    else:
        M_k1[i,i] = -2
        M_k1[i,i+1] = 1
        M_k1[i,i-1] = 1
        
        M_k2[i,i] = 2*((1/W)+1) 
        M_k2[i,i-1] = -1
        M_k2[i,i+1] = -1  
        
        M_k12[i,i] = 2*((1/W)-3) 
        M_k12[i,i-1] = 3
        M_k12[i,i+1] = 3 
        
        pass
    
#----- Resuelvo el problema temporal-----

for i in range(0,N_t):

    K1 = M_k1*T_n*W                         #Calculo K1
        
    K2 = np.linalg.solve(M_k2,(M_k12*K1))   #Calculo K2

    for l in range(1,N):
         T_n[l] = T_n[l] + 0.5*(K1[l]+K2[l])  #Metodo RK2
    
u_an = an_sol(x, t_fin)     #Solucion analitica

#--- DATA PRINT --------

plt.title("TP7-Ej2 Finite difference method - Transient Heat Conduction  - Dirichlet boundary conditions")
plt.xlabel("Lenght [cm]")
plt.ylabel("Temperature [°C]")
plt.plot(x, u_an, label = "Exact Solution ")
plt.plot(x, T_n[:,0], label = "Computed Solution")
plt.legend()
plt.show()