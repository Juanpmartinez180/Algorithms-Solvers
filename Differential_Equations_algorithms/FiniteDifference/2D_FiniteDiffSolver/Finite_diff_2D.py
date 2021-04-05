import numpy as np
import matplotlib.pyplot as plt

#---- Dimensionamiento dominio -----------
L = 100     #Ancho de la placa
H = 100     #Alto de la placa

#---- Propiedades material ---------------
Kx = 235    #Coef de transmision [W/mK]
Ky = 235
Q = 10      #Fuente de calor (cte)

#---- Condiciones iniciales --------------
T1 = 100      #Temperatura extremo inf inicial [°C]
T2 = 0    #Temperatura extremo izq inicial [°C]
T3 = 0    #Temperatura extremo sup inicial [°C]
T4 = 0    #Temperatura extremo dcho inicial [°C]

#---- Creacion de la grilla --------------
Nx = 60     #Discretizaciones en X
Ny = 60     #Discretizaciones en Y
Hx = L/Nx   #Ancho grilla en X
Hy = H/Ny   #Ancho grilla en Y
N = (Nx+1)*(Ny+1) #Puntos de la grilla

#--- Funcion analitica ---
def an_sol(x,y) : 
    sol = 0
    for i in range(1,10):
        c = i*np.pi
        sol = sol + ((1-(-1)**i)/(c)) * (np.sinh(c*(H-y)/L)) * np.sin(c*x/L) / np.sinh(c*H/L)
    
    sol = sol*T1*2
    return (sol)
#--- Condiciones de contorno
def boundary_cond():
    
    return
#--- MAIN --------------
#Deseamos resolver el sistema AT = P     
C = np.zeros((N, N))        #Vector de coeficientes 
A = np.matrix(C)            #Matriz de coeficientes
P = np.zeros((N,1))         #Vector de variables independientes
T = np.zeros((Ny+1,Nx+1))   #Matriz de soluciones espaciales

#--- Defino condiciones de contorno en el vector de solucion---
for i in range(0, N-1-Nx, Nx+1):   #Contorno izquierdo
    P[i] = T2
    A[i,i] = 1
    
for i in range(Nx, N-1, Nx+1):  #Contorno derecho
    P[i] = T4
    A[i,i] = 1

for i in range(1, Nx, 1):       #Contorno superior
    P[i] = T3
    A[i,i] = 1

for i in range(N-1-Nx, N, 1):   #Contorno inferior
    P[i] = T1
    A[i,i] = 1    

#---- Defino los coeficientes de dif. finita en cada punto -----
for j in range(1, Nx):
    for i in range(Nx+1+j , N-1-(Nx+1), Nx+1):
        a = Kx/(Hx**2)
        b = Ky/(Hy**2)
        
        P[i] = -Q
        A[i,i] = -(2*a + 2*b)     #P[i,j]
        A[i,i+1] = 1*a            #P[i+1,j]
        A[i,i-1] = 1*a            #P[i-1,j]
        A[i,i+Nx+1] = 1*b         #P[i,j-1]    "Recurrencia magica"             
        A[i,i-Nx-1] = 1*b         #P[i,j+1]


#---- solucion analitica
x = np.linspace(0, L, Nx+1)
y = np.linspace(0, H, Ny+1)
xx,yy = np.meshgrid(x,y)

R = an_sol(xx,yy)
print(an_sol(25,50))        #Solucion analitica

#plt.matshow(A)
#plt.colorbar()
#plt.show()

X = np.linalg.solve(A,P)    #Resuelvo el sistema de ecuaciones

for i in range(0, Ny+1):        #Cargo la matriz con los valores calculados
    for j in range(0, Nx+1):
        T[i,j] = X[j + i*(Nx+1)] 


# plt.matshow(T);
# plt.colorbar()

#--- DATA PRINT --------

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(T)#, interpolation='nearest')
fig.colorbar(cax)

ax.set_xticklabels("x")
ax.set_yticklabels("y")

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(xx, yy, T, 100, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Temperature')
ax.set_title('Heat Distribution')

plt.show()
