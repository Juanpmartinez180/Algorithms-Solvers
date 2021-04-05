import numpy as np 
import matplotlib.pyplot as plt 

#--- Condiciones iniciales y constantes --------
A = 10**3
Mu = 0.5

t_inic = 0              #Tiempo inicial
t_fin = 1               #Tiempo final
N = 1                   #Numero de ecuaciones a resolver
P = 2                   #Orden del metodo
y0 = 1
u0 = [ 1 ]            #Condiciones iniciales

e_min = 0.00001         #Cota de error minima
e_max = 0.00012         #Cota de error maxima
k0 = 0.0002               #Paso temporal inicial
k_min = 0.0002          #Paso temporal minimo
k_max = 0.01            #Paso temporal maximo
beta = 0.84             #factor de seguridad stepsize

#---- Funcion F(tn,yn) ----------------------
def ecuation(t,y):              #Funcion en forma vectorial

    function = np.array([   [ -(A)*y[0] + A*np.sin(t) ] ])        #Y1                    

#    function = np.array([   [ y[1]] ,                               #Y1
#                            [ 2*y[1] - y[0] + np.cos(t) ] ])        #Y2  
    return(function[:,0]) 

#---- Solucion exacta -----------------------
def exact_solution(t):

    function = np.array([  [ np.exp(-A*t) * (1 + A/(A**2 + 1)) + (A/(A**2 + 1))*(-np.cos(t) + A*np.sin(t))  ] ])
#    function = np.array ([  [ 0.5*(np.exp(t)*(t+2) - np.sin(t) ) ]    ])

    return(function[:,0])

#---- Euler Explicit method --------------
def Euler(time, k, u1):

    k1 = k*ecuation(time             , u1 )
    k2 = k*ecuation(time + (1/2)*k   , u1 + (1/2)*k1 )
    k3 = k*ecuation(time + (3/4)*k   , u1 + (3/4)*k2 )

    a1 = u1 + k2
    a2 = u1 + (2/9)*k1 + (1/3)*k2 + (4/9)*k3

    a1 = a1[:,np.newaxis]                        #Como a me devuelve un vector de N variables, lo convierto en una matriz de Nx1
    a2 = a2[:,np.newaxis]                        #Como a me devuelve un vector de N variables, lo convierto en una matriz de Nx1

    return(a1, a2)            #Devuelvo el valor del tiempo, las soluciones de interes y el maximo error

#--- Error decision function ----------------
def decision_error(v1, v2, k_old):

    e = error_computed(v1, v2)

    s = beta * ((e_max / e)**(1/P))     #Factor de decision

    if s < 1 :     #Si el error es mayor al propuesto
        k_new = s*k_old     #Calculo el k optimo
        a = 0
        if k_new < k_min:
            k_new = k_min
            a = 1
            pass
        pass

    else :         #Si el error igual o menor al propuesto
        k_new = s*k_old     #Calculo el k optimo
        a = 1
        if k_new > k_max:
            k_new = k_max
            pass
        pass

    return(a, k_new)

#--- Calculo del error computado -------------
def error_computed(v1, v2):         #Determino el error de las soluciones computadas con la norma infinito

    error = 2 * np.abs( (v1-v2) / (v1+v2) )

    error_max = np.amax(error)         #Determino cual es el error maximo

    return(error_max)

#--------- MAIN ------------------------------

time = [t_inic]                 #Creo vector de tiempos computados
u_exact = [y0]                  #Vector de solucion exacta
u1 = np.zeros((N,1))            #Matriz de soluciones (para paso k)
u1[:,0] = u0                    #Cargo las condiciones iniciales
k = [k0]                        #Vector de pasos temporales

i = 0                           #Variable auxiliar

while time[i] < t_fin:          

    solution = Euler( time[i], k[i], u1[:,i]  )     #Obtengo las soluciones rk4 y rk5
    v1 = solution[0]                                    #Solucion RK4
    v2 = solution[1]                                    #Solucion RK5

    x = decision_error(v1, v2, k[i])        #Le paso los valores a la funcion de decision
    a = x[0]                                    #Parametro de decision
    k_new = x[1]                                #Valor de k optimo calculado

    if a == 0 :                             #Si e > e_max
        k[i] = k_new                                #Reemplazo el valor de k y vuelvo a iterar
        pass
    
    elif a ==1 :                            #Si e < e_max
        u1 = np.append(u1, v1, axis=1)              #Añado el valor a la matriz solucion (Nxi)
        k.append(k_new)                             #Añado el nuevo valor de k 
        time.append( time[i] + k[i] )               #Guardo el tiempo computado
        u_exact.append( exact_solution(time[i+1]) ) #Calculo la solucion exacta para t[i]+k[i]
        i = i+1                                     #Incremento la variable auxiliar
        pass
    
#--- Graficas -----------------------
plt.subplot(2,1,1)
plt.xlabel("Time")
plt.ylabel("Position")
plt.title("Ej3 TP5 - Explicit Euler Method")
plt.plot(time, u1[0], 'ro')     #Grafica solucion computada    
plt.plot(time, u_exact)         #Grafico solucion exacta

plt.subplot(2,1,2)
plt.xlabel("time")
plt.ylabel("k")
plt.plot([0,t_fin],[k_min, k_min])
plt.plot([0,t_fin],[k_max, k_max])
plt.plot(time, k)

plt.show()