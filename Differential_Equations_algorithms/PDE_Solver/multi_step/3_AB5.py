import numpy as np 
import matplotlib.pyplot as plt

#--- Condiciones iniciales y constantes --------

G = 6.672*(10**-11)
Me = 5.9742*(10**24) 
Re = 6378140
V0 = 6700
H0 = 772000 + Re

    #Coeficientes AB5
Q0 = 1901/720
Q1 = 2774/720
Q2 = 2616/720
Q3 = 1274/720
Q4 = 251/720

N = 4       #Numero de ecuaciones
t_inic = 0  #Tiempo inicial
k = 0.05     #Paso temporal
u0 = [ H0 , 0 , 0 , V0/H0 ]    #Condiciones iniciales

#---- Funcion F(tn,yn) ------------------------------------
def ecuation(t,y):              #Funcion en forma vectorial

    function = np.array([   [ y[1]] ,                               #Y1
                            [(y[0]*(y[3]**2)) - G*Me/(y[0]**2) ] ,  #Y2
                            [ y[3] ] ,                              #Y3
                            [(-2*y[1]*y[3])/y[0] ] ])               #Y4
    return(function[:,0])

#---- Runge Kutta 4 order method --------------------------
def RK4(u,steps,t):
   
    for i in range(0, steps-1):

        k1 = k*ecuation(t[i], u[:,i])                   #Calculo los coeficientes segun el metodo

        k2 = k*ecuation(t[i]+(k/2.0), u[:,i]+(1/2)*k1)
        
        k3 = k*ecuation(t[i]+(k/2.0), u[:,i]+(1/2)*k2)

        k4 = k*ecuation(t[i]+k, u[:,i]+k3)

        a = u[:,i] + (k/6.0)*(k1 + 2*k2 + 2*k3 + k4)    #Calculo el valor de la solucion en el punto i
        a = a[:,np.newaxis]                             #Como a me devuelve un vector de 4 variables, lo convierto en una matriz de 4x1
        t.append( t[i] + k )                            #Añado al tiempo el nuevo valor
        u = np.append(u, a, axis =1)                    #Añado el valor a la matriz solucion (4xi)
       
    return(t,u)

#---- Metodo Adams-Bashforth 4 pasos -----------------------
def AB4(k):

    t = [t_inic]            #Vector de tiempo
    u = np.zeros((N,1))     #Vector solucion de N filas y 1 columna (donde almaceno el valor inicial)
    u[:,0] = u0             #Cargo los valores iniciales en el vector solucion
    steps = 5               #Numero de soluciones requeridas para comenzar el metodo
    i = 0                   #Variable auxiliar para iterar el metodo

    while u[0,i] >= Re :

        if i < steps :                          #Si se calcularon menos de (steps) soluciones
            a = RK4(u, steps, t)                   #Determino las soluciones iniciales con el metodo RK4
            time = a[0]                            #Cargo los tiempos iniciales computados
            u = a[1]                            #Cargo las soluciones iniciales computadas
            i = steps -1                        #El proximo paso a resolver será el (steps-1)
    
        b = u[:,i] + k * (      ( Q0*ecuation( time[i]  , u[:,i]   ))      #Solucion mediante Adams Bashforth
                            -   ( Q1*ecuation( time[i-1], u[:,i-1] ))  
                            +   ( Q2*ecuation( time[i-2], u[:,i-2] )) 
                            -   ( Q3*ecuation( time[i-3], u[:,i-3] ))
                            +   ( Q4*ecuation( time[i-4], u[:,i-4] ))  )
                 
        b = b[:,np.newaxis]                 #Como a me devuelve un vector de 4 variables, lo convierto en una matriz de 4x1      
        u = np.append(u, b, axis =1)        #Añado el valor a la matriz solucion (4xi)
        time.append( t[i] + k )             #Añado al tiempo el nuevo valor

        i = i+1                             #Aumento un paso la variable auxiliar

    return(time, u)

#--- Calculo coeficiente de precision Q ---------------
def precision_quotient (k):

    v = AB4(k)            
    v1 = v[1]               #Soluciones para k
    v2 = AB4(k/2)[1]      #Soluciones para k/2
    v4 = AB4(k/4)[1]      #Soluciones para k/4

    time = len(v[0]) #Iteraciones de tiempo
    q = np.zeros((N, time))

    for i in range(0,time-10,1):
        
        q[0,i] = (v1[0,i] - v2[0,2*i]) / (v2[0,2*i] - v4[0,4*i])
        
        pass
  
    return(q)

#---------------- MAIN ----------------------------

a = AB4(k)          #Inicio el metodo Adams Bashforth de 4 pasos con paso temporal K
time = a[0]             #Vector de tiempos computados
solution = a[1]         #Vector de soluciones de interes

#q = precision_quotient(k)

angle =  (a[1])[2, len(time)-1]
radius = (a[1])[0, len(time)-1]

print ("Tiempo de vuelo= ", time[len(time)-1], " [seg]")
print ("Angulo de impacto= ", angle, " [rad]")

#---- PRINT --------------------
plt.subplot(2,1,1)
plt.title("Ej2 TP4 4-step Adams-Bashforth Method")
plt.xlabel("Time")
plt.ylabel("Altitude[km]")
plt.plot(time, (solution[0] - Re)/1000 )

plt.subplot(2,1,2)
plt.xlabel("Time")
plt.ylabel("Angle [rad]")
plt.plot(time, solution[2])

plt.show()