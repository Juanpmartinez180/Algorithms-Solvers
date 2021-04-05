import numpy as np
import matplotlib.pyplot as plt
#--- Propiedades del ejercicio----
H = 4*(10**-3)           #Separacion de las placas
Nu = 2.17*(10**-4)       #Coef de viscosidad cinematica
V_t = 40                 #Velocidad de la placa sumerior [m/s]

#--- Condiciones iniciales y de bordes ----
U0 = 0      #Velocidad extremo inicial [m/s]
Un = V_t    #Velocidad extremo final [m/s]
U_t0 = 0    #Velocidad en el instante inicial t=0 [m/s]

#--- Discretizacion espacial y temporal ---
t_fin = 0.005        #Tiempo final a calcular [s]
K_t = 2*(10**-5)    #Paso temporal
h = 1*(10**-4)      #Longitud de la discretizacion (espacio de grilla)
N = round(H/h)      #Numero de discretizaciones
N_t = round(t_fin / K_t)   #Iteraciones a resolver 

#--- Funcion analitica ---
def an_sol(x,t) : 
    # N = 10 #Iteraciones de la serie
    # C1 = K/Ro
    # C2 = np.pi / L
    
    # sol = 0
    # for i in range(1,N):
    #     sol = sol + np.exp(-t*C1*(i*C2)**2)*((1-(-1)**i)/(i*np.pi))*np.sin(C2*i*x)

    # sol = T0 + 2*(T_t0-T0)*sol
    sol = 1
    return (sol)

#--- MAIN --------------
      
x = np.linspace(0, H, N+1)  #Puntos de la grilla a calcular
u_an = np.zeros((len(x)))   #Vector solucion analitica

C = np.zeros((N+1, N+1))    #Vector de coeficientes  
A = np.matrix(C)            #Matriz de coeficientes
B = np.zeros((N+1,1))       #Vector de variables independientes
B_1 = np.zeros((N+1,1))     #Vector de solucion en t_i+1

#-----Cargo condiciones iniciales y coef. en la matriz
for i in range(1, N):
    
    B[i] = U_t0       #Condiciones iniciales en en el dominio 
          
B[0] = U0       #Condicion de Dirichlet en T0
B[N] = Un       #Condicion de Dirichlet en Tn
B_1[0] = U0     #Condicion de Dirichlet en T0
B_1[N] = Un     #Condicion de Dirichlet en Tn

for i in range (0, N+1):    #Lleno la matriz de coeficientes
    A[i,i] = 1    
    
#----- Resuelvo el problema temporal-----

W = K_t * Nu / (h**2)     #Constante
    
for i in range(0,N_t): 
    for j in range(1, N):
              
        B_1[j] = W*(B[j-1]-2*B[j]+B[j+1]) + B[j]    #Cargo el vector de variables independientes con los valores de en t = tn
    
    B = np.linalg.solve(A,B_1)                      #Resuelvo el sistema (determino valores en t=tn+1)      
    
#u_an = an_sol(x, t_fin)     #Solucion analitica

#--- DATA PRINT --------

plt.title("TP7-Ej4 Finite difference method - Transient Heat Conduction  - Dirichlet boundary conditions")
plt.xlabel("Lenght [cm]")
plt.ylabel("Temperature [°C]")
#plt.plot(x, u_an, label = "Exact Solution at stationary state")
plt.plot(x, B[:,0], label = "Computed Solution")
plt.legend()
plt.show()