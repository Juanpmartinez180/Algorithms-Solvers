import numpy as np 
import matplotlib.pyplot as plt 

#--- Condiciones iniciales y constantes --------

Mu = 0.5

t_inic = 0          #Tiempo inicial
t_fin = 2           #Tiempo final
N = 1               #Numero de ecuaciones a resolver
k = 0.01            #Paso temporal
u0 = [ 1  ]         #Condiciones iniciales

#---- Funcion F(tn,yn) ----------------------
def ecuation(t,y):              #Funcion en forma vectorial

    function = np.array([   [ -y[0] + np.sin(2*np.pi*t)]  ])                      #Y1

    return(function[:,0])

#---- Solucion exacta -----------------------
def exact_solution(t):          #Solucion exacta en forma vectorial

    a = 1 / (1+4*(np.pi)**2)
    function = np.array([  [np.exp(-t)*(1+2*np.pi*a) + a*(np.sin(2*np.pi*t) - 2*np.pi*np.cos(2*np.pi*t))    ] ])

    return(function[:,0])

#---- Euler Explicit method --------------
def Euler(k):
    time = [t_inic]         #Creo vector de tiempos computados
    u = np.zeros((N,1))     #Matriz de soluciones
    u[:,0] = u0             #Cargo las condiciones iniciales

    error = np.zeros((1))   #Creo vector de errores computados
    sol = np.zeros((1))     #Creo vector de solucion exacta
    sol[0] = 1

    i = 0                   #Variable auxiliar

    while time[i] < t_fin :

        k1 = ecuation( time[i]     , u[:,i] )           #Calculo los coeficientes segun el metodo
        k2 = ecuation( time[i] + k/2 , u[:,i] + k*k1/2 )

        a = u[:,i] + k *  k2                            #Calculo el valor de la solucion en el punto i+1 
        a = a[:,np.newaxis]                             #Como a me devuelve un vector de N variables, lo convierto en una matriz de Nx1
        u = np.append(u, a, axis =1)                    #Añado el valor a la matriz solucion (Nxi)
        time.append(time[i]+k)                          #Añado el tiempo computado al vector tiempo


        sol = np.append(sol,exact_solution(time[i+1]))    #Calculo valor de la solucion exacta
        e = np.abs(u[0,i+1] - sol[i+1])                   #Calculo el error con el valor de la solucion exacta
        error = np.append(error, e)                     #Añado el error calculado al vector de errores
        
        i = i + 1                                       #Añado un paso a la variable auxiliar

    return(time, u[0], np.amax(error), sol, error)             #Devuelvo el valor del tiempo, las soluciones de interes y el maximo error

#--- Calculo coeficiente de precision Q ---------------
def precision_quotient (k):

    v = Euler(k)            
    v1 = v[1]               #Soluciones para k
    v2 = Euler(k/2)[1]      #Soluciones para k/2
    v4 = Euler(k/4)[1]      #Soluciones para k/4

    time = len(v[0]) #Iteraciones de tiempo
    q = np.zeros((time))

    for i in range(0,time,1):
        
        q[i] = (v1[i] - v2[2*i]) / (v2[2*i] - v4[4*i])
        
        pass
  
    return(q)

#--- Error vs K ---------------------------- (necesito solucion exacta!! )
def error_convergence():
    k_inic = 0.001          #Valor inicial de k
    k_fin = 0.1             #Valor final de k
    delta_k = 0.001         #Delta de k
    a = np.arange(k_inic, k_fin, delta_k)   #Vector con los valores de k a iterar
    error = np.zeros((len(a)))

    for i in range(0, len(a),1):
        e = Euler(a[i])[2]
        error[i] = e

        pass

    return(a, error)

#--- Solutions vs K ------------------------ (Graficar soluciones para distintos k)
def solutions_vs_k():

    a = [0.001, 0.01 , 0.1]             #Valores de k a iterar
    e = [0, 1, 2]                       #Vector de errores calculados

    for i in range (0, len(a), 1):
        method = Euler(a[i])            #Llamo el metodo para el K[i]
        results = method[1]                 #Asigno el resultado calculado
        times = method[0]                   #Asigno el tiempo iterado
        e[i] = method[4]                    #Asigno el error calculado

        plt.subplot(4,1,1)
        plt.xlabel("Time")
        plt.ylabel("Precition")
        plt.plot(times,results)         #Grafico soluciones vs K

        plt.subplot(4,1,2)
        plt.xlabel("Time")
        plt.ylabel("Error")
        plt.plot(times,e[i])            #Grafico errores(k) vs tiempo

        pass
    
    return()

#--------- MAIN ------------------------------

solution = Euler(k)             #Soluciones para paso k
t = solution[0]                     #Tiempo transcurrido           
y0 = solution[1]                    #Solucion 1 computada
exact = solution[3]                 #Solucion exacta         

q = precision_quotient(k)       #Coeficiente de precision Q

error_k = error_convergence()   #Error computado vs k
k_computed = error_k[0]             #Valores de k computados
error_computed = error_k[1]         #Valores de error computados

solutions_vs_k()                #Grafico soluciones para distintos K


#--- Graficas -----------------------

plt.subplot(4,1,2)
plt.xlabel("Time")
plt.ylabel("Position")
plt.title("Ej3 TP5 - Explicit Euler Method")
#plt.plot(t,y0)         #Grafico solucion computada 1
#plt.plot(t,exact)       #Grafico solucion exacta

plt.subplot(4,1,3)
plt.xlabel("K")
plt.ylabel("Error")
#plt.plot(t,y1)                     #Grafico solucion computada 2
plt.plot(k_computed,error_computed) #Grafico error vs K computado

plt.subplot(4,1,4)
plt.xlabel("Time")
plt.ylabel("Precition quotient")
plt.plot(t,q)           #Grafico coeficiente de precision

plt.show()