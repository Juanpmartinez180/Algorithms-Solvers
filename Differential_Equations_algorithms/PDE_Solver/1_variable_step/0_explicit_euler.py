import numpy as np 
import matplotlib.pyplot as plt 

#--- Condiciones iniciales y constantes --------
A = 10**3
Mu = 0.5

t_inic = 0              #Tiempo inicial
t_fin = 2*(np.pi)               #Tiempo final
N = 1                   #Numero de ecuaciones a resolver
P = 1                   #Orden del metodo
u0 = [ 1  ]           #Condiciones iniciales

e_min = 0.00001    #Cota de error minima
e_max = 0.000012    #Cota de error maxima
k0 = 0.1                #Paso temporal inicial
k_min = 0.0002           #Paso temporal minimo
k_max = 0.01             #Paso temporal maximo
beta = 0.84             #factor de seguridad stepsize

#---- Funcion F(tn,yn) ----------------------
def ecuation(t,y):              #Funcion en forma vectorial

    function = np.array([   [ -(A)*y[0]] + A*np.sin(t)  ])        #Y1                    

        #function = np.array([   [ y[1]] ,                               #Y1
        #                    [ 2*y[1] - y[0] + np.cos(t) ] ])        #Y2  
    return(function[:,0]) 

#---- Solucion exacta -----------------------
def exact_solution(t):

    a = 1 / (1+4*(np.pi)**2)
    function = np.array([  [np.exp(-t)*(1+2*np.pi*a) + a*(np.sin(2*np.pi*t) - 2*np.pi*np.cos(2*np.pi*t))    ] ])

    return(function[:,0])

#---- Euler Explicit method --------------
def Euler(time, k, u):

    a = u + k * ecuation(time,u)      #Calculo el valor de la solucion en el punto i+1   
    a = a[:,np.newaxis]               #Como a me devuelve un vector de N variables, lo convierto en una matriz de Nx1
    
    return(a)            #Devuelvo el valor del tiempo, las soluciones de interes y el maximo error

#--- Error decision function ----------------
def decision_error(v1, v2):

    e = error_computed(v1, v2)

    if e > e_min and e < e_max:     #Si el error está dentro de la cota propuesta
        a = 1                           #Devuelvo a = 1
        pass
    elif e > e_max:                 #Si el error es mayor al propuesto
        a = 0                           #Devuelvo a = 0
        pass
    elif e < e_min:                 #Si el error es menor al propuesto
        a = -1                          #Devuelvo a = -1
        pass

    return(a)

#--- Calculo del error computado -------------
def error_computed(v1, v2):         #Determino el error de las soluciones computadas con la norma infinito

    error = beta * ( 1 / ((2**P)-1) ) * np.abs( (v1-v2) / (v1+v2) )
    
    error_max = np.amax(error)         #Determino cual es el error maximo

    return(error_max)

#--------- MAIN ------------------------------

time = [t_inic]                 #Creo vector de tiempos computados
u1 = np.zeros((N,1))            #Matriz de soluciones (para paso k)
u1[:,0] = u0                    #Cargo las condiciones iniciales
u2 = np.zeros((N,1))            #Matriz de soluciones (para paso k/2)
u2[:,0] = u0                    #Cargo las condiciones iniciales
k = [k0]                        #Vector de pasos temporales

i = 0                   #Variable auxiliar

while time[i] < t_fin:          

    v1   = Euler( time[i]          , k[i]  , u1[:,i]   )        #Obtengo la solucion con paso k en ti = t+k
    v2_0 = Euler( time[i]          , k[i]/2, u2[:,i]   )        #Obtengo la solucion con paso k/2 en ti = t+k/2
    v2   = Euler( time[i] + k[i]/2 , k[i]/2, v2_0[:,0] )        #Obtengo la solucion con paso k/2 en ti = t+k

    a = decision_error(v1, v2)              #Le paso los valores a la funcion de decision

    if a == 1 :                             #Si e_min < e < e_max
        u1 = np.append(u1, v1, axis=1)          #Añado el valor a la matriz solucion 1(Nxi)
        u2 = np.append(u2, v2, axis=1)          #Añado el valor a la matriz solucion 2(Nxi)
        k.append (k[i])                         #Sigo utilizando el mismo valor de k
        time.append( time[i] + k[i] )           #Guardo el tiempo computado
        i = i+1                                 #Incremento la variable auxiliar
        pass
    
    elif a == 0 :                           #Si e > e_max
        if k[i] <= k_min:                        
            u1 = np.append(u1, v1, axis=1)      #Añado el valor a la matriz solucion (Nxi)
            u2 = np.append(u2, v2, axis=1)      #Añado el valor a la matriz solucion 2(Nxi)
            k.append(k_min)                     #El proximo K será el minimo
            time.append( time[i] + k[i] )       #Guardo el tiempo computado
            i = i+1                             #Incremento la variable auxiliar
            pass
        else:
            k[i] = 2*k[i]/3                     #El proximo k será mas pequeño
            pass
        pass

    else:                                   #Si e < e_min
        if k[i] >= k_max:
            u1 = np.append(u1, v1, axis=1)      #Añado el valor a la matriz solucion (Nxi)
            u2 = np.append(u2, v2, axis=1)      #Añado el valor a la matriz solucion 2(Nxi) 
            k.append( k_max)                    #El proximo K será el maximo
            time.append( time[i] + k[i] )       #Guardo el tiempo computado
            i = i+1                             #Incremento la variable auxiliar
            pass
        else:               
            u1 = np.append(u1, v1, axis=1)      #Añado el valor a la matriz solucion (4xi)
            u2 = np.append(u2, v2, axis=1)      #Añado el valor a la matriz solucion 2(Nxi)
            k.append( k[i]*3/2 )                #El proximo K será mayor 
            time.append( time[i] + k[i] )       #Guardo el tiempo computado
            i = i+1                             #Incremento la variable auxiliar
            pass
        pass
    
#--- Graficas -----------------------
plt.subplot(2,1,1)
plt.xlabel("Time")
plt.ylabel("Position")
plt.title("Ej3 TP5 - Explicit Euler Method")
plt.plot(time,u1[0], 'ro')       #Grafico solucion exacta

plt.subplot(2,1,2)
plt.xlabel("time")
plt.ylabel("k")
plt.plot([0,t_fin],[k_min, k_min])
plt.plot([0,t_fin],[k_max, k_max])
plt.plot(time, k)

plt.show()