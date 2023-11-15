# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 15:48:47 2022

@author: yasma
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.special as sp
import time
import copy

total_nodos=1000
enlac_x_nodo=3

# Red BA=======================================================================
NetworkBA=nx.barabasi_albert_graph(total_nodos, enlac_x_nodo)
#=============================================================================

print("-------------------------------")
print("Información sobre la red:")
print("-------------------------------")
print("Numero de nodos: N = " + str(NetworkBA.number_of_nodes()))
print("-------------------------------")
print("Numero de enlaces: N = " + str(NetworkBA.number_of_edges()))
print("-------------------------------")

#grados que aparecen en la red asociados a cada nodo
DegreeValues = [degree for nodes, degree in NetworkBA.degree()]
#mayor grado observado en la red
print('promedio grado:', np.mean(DegreeValues))
k_max=np.max(DegreeValues)

#Escalado y espaciado lineal en histograma===================================
# Crea espaciado para el histograma entre [-0.5, n+1.5] donde n es el grado maximo de la red
# con esto se puede calcular la distribución en los bines 0, 1, 2,..., n-1
bins_lineal= np.arange(k_max + 1) - 0.5;
# Realiza el conteo con la función histogram de Numpy
[hist_lin, Bins] = np.histogram(DegreeValues, bins_lineal)
# Calcula la frecuencia de ocurrencia de cada grado
pk = hist_lin / np.sum(hist_lin)

plt.figure()
plt.plot(np.arange(k_max)+1, pk, 'bo');
plt.title("Distribución de grados de BA con N=" + str(total_nodos) + " y m="+ str(enlac_x_nodo));
plt.ylabel("p_k");
plt.xlabel("Grado");
plt.show();

#=============================================================================

#Escalado logaritmico y espaciado lineal en histograma=======================
plt.figure()
plt.loglog(np.arange(k_max)+1, pk, 'bo');
plt.title("Distribución de grados de BA con N=" + str(total_nodos) + " y m="+ str(enlac_x_nodo) + " (log)");
plt.ylabel("p_k");
plt.xlabel("Grado");
#=============================================================================

#Escalado y espaciado logaritmico en histograma=======================
bins = np.logspace(0, np.log10(k_max) + 1, 25) #extremos de los bins
widths = (bins[1:] - bins[:-1]) #ancho de los bins

# Calculate histogram
hist = np.histogram(DegreeValues, bins=bins)
# normalize by bin width
hist_norm = hist[0]/widths
hist_norm = hist_norm/ np.sum(hist_norm)


# Grafica en escala logaritmica espaciado lineal para comparar=================
plt.figure()
plt.loglog(np.arange(k_max)+1, pk, 'bo', label='espac lin');

puntos_hist_log=(bins[1:]+bins[:-1])/2 #centro de los bins logaritmicos
plt.plot(puntos_hist_log, hist_norm,'ro', label='espac log')
plt.title("Distribución de grados de BA con N=" + str(total_nodos) + " y m="+ str(enlac_x_nodo) + " (log)");
plt.ylabel("p_k");
plt.xlabel("Grado");
plt.legend()
plt.xscale('log')
plt.yscale('log')
#=============================================================================

#ajuste de la distribucion====================================================
#calculo del exponente mediante regresion lineal, de acuerdo a ecuacion
# log (p_k) = -gamma * log (k) + C

min_val_permit=3 #generalemnete la ley de potencia se cumple a partir de cierto valor minimo
max_val_permit=16 #los valores de log (p_k) a partir de este indice son -inf

k_val = np.log10(puntos_hist_log[min_val_permit:max_val_permit])

pk_datos_log = -np.log10(hist_norm[min_val_permit:max_val_permit])

A = np.vstack([k_val, np.ones(len(k_val))]).T #A contiene info de k y de C

Gamma, c = np.linalg.lstsq(A, pk_datos_log, rcond=None)[0]

#Del ajuste anterior se utiliza la pendiente 'Gamma' y el otro parametro se ajusta
# de acuerdo a cumplir con la condicion ppal de las PMF (suma 1)

# pk_ajust=np.power(puntos_hist_log,-Gamma)/np.sum(np.power(puntos_hist_log,-Gamma))

puntos_ajuste = np.arange(k_max)+1
pk_ajust=np.power(puntos_ajuste,-Gamma)/np.sum(np.power(puntos_ajuste,-Gamma))
hurtwitz_f=sp.zeta(Gamma,3)
pk_ajust2=np.power(puntos_ajuste,-Gamma)/hurtwitz_f

#=============================================================================

#Comparacion entre histograma y pmf ajustada de ley de potencia================
plt.figure()
plt.plot(puntos_ajuste, pk_ajust,'--', label='ajuste')
plt.plot(puntos_ajuste, pk_ajust2,'*', label='ajusteHurt')
plt.plot(bins[:-1], hist_norm,'ro', label='espac log')
plt.title("ajuste de dist grado Gamma=" + str(Gamma));
plt.ylabel("p_k");
plt.xlabel("Grado");
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
plt.xscale('log')
plt.yscale('log')
#=============================================================================

# #Predicciones del modelo y resultados de los datos============================

# #observaciones
Momento_1orden=np.mean(DegreeValues)
print("Momento de 1er orden calculado: " + str(Momento_1orden))

Momento_2orden=np.var(DegreeValues) + np.power(Momento_1orden,2)
print("Momento de 2do orden calculado: " + str(Momento_2orden))
Kmax=np.max(DegreeValues)
print("Maximo grado observado: " + str(Kmax))
Kmax_pred = np.min(DegreeValues) * np.power(NetworkBA.number_of_nodes(),1/(Gamma-1))
print("Maximo grado predicho: " + str(Kmax_pred))
#Si el grafo esta conectado
conectado = nx.is_connected(NetworkBA)
if conectado == True:
    print("distancia promedio observada: < d > = ", nx.average_shortest_path_length(NetworkBA))
    print("Diámetro = ", nx.diameter(NetworkBA))
else:
    print("Red con al menos un nodo desconectado, distancia promedio y diámetro infinitos")
distancia_prom_pred=np.log(NetworkBA.number_of_nodes())/np.log(np.log(NetworkBA.number_of_nodes()))
print("distancia promedio predicha: < d > = " + str(distancia_prom_pred))
clustering_coef_pred=np.power(np.log(NetworkBA.number_of_nodes()),2)/NetworkBA.number_of_nodes()
print("Coeficiente de agrupamiento promedio predicho: <C> = ", clustering_coef_pred)
print("Coeficiente de agrupamiento promedio calculado: <C> = ", nx.average_clustering(NetworkBA) )

# #=============================================================================
#grados que aparecen en la red asociados a cada nodo
DegreeValues = [degree for nodes, degree in NetworkBA.degree()]
lista_de_nodos=list(NetworkBA.nodes)

#===============================================================================
#===============================================================================
#Robustez a fallas aleatorias para diferentes valores de f = prob de eliminar un nodo
#===============================================================================
#===============================================================================
# Calcula el numero de nodos del mayor componente conectado, el "Giant Component" de la red inicial
Gcc = sorted(nx.connected_components(NetworkBA), key=len, reverse=True)
G0 = NetworkBA.subgraph(Gcc[0]) #subgrafo generado por el Giant Comp
numero_nodos_GC=G0.number_of_nodes()    
Prob_GC_0 = numero_nodos_GC / NetworkBA.number_of_nodes()  #prob de un nodo pertenecer al GC


puntos=100
prob_eliminar_nodo = np.linspace(0, 1, puntos + 1)#valores de probab
compara = 1 - prob_eliminar_nodo
Prob_pert_GC=np.zeros(prob_eliminar_nodo.shape[0]) #almacenar valores de Robustez
numero_nodos_inic=NetworkBA.number_of_nodes()

start_time = time.time()

for ii in range(0,puntos): #para cada valor de prob_eliminar_nodo
    Network_Test=copy.deepcopy(NetworkBA) # es necesario usar esta copia para crear un nuevo objeto y no un enlace al anterior 
        
    for jj in range(0,numero_nodos_inic): #para cada nodo de la red
        
        if np.random.uniform(0,1) < prob_eliminar_nodo[ii]: #se elimina nodo o no
            eliminado = lista_de_nodos[jj]          
            Network_Test.remove_node(eliminado)
            
    # Calcula el numero de nodos del mayor componente conectado el "Giant Component"
    Gcc = sorted(nx.connected_components(Network_Test), key=len, reverse=True)
    
    if len(Gcc) > 0: #si esta lista esta vacia no quedan nodos
        G0 = Network_Test.subgraph(Gcc[0]) #subgrafo generado por el Giant Comp
        numero_nodos_GC=G0.number_of_nodes()    
        Prob_pert_GC[ii] = numero_nodos_GC / numero_nodos_inic #Network_Test.number_of_nodes() #prob de un nodo pertenecer al GC
    else:
        Prob_pert_GC[ii]=0
    print("iteracion: " + str(ii))
    
Prob_pert_GC = Prob_pert_GC/Prob_GC_0    #normalizar
            
plt.figure()
plt.plot(prob_eliminar_nodo, Prob_pert_GC, 'b--', label='robustez')
plt.plot(prob_eliminar_nodo, compara,'r--', label='y=1-x')
plt.title("Robustez de muestra de la red BA")
plt.ylabel("P_inf (f) / P_inf (0)")
plt.xlabel("f")
plt.grid(True)
plt.legend()
plt.show()    



print("---exec time parte 1: %s seconds ---" % (time.time() - start_time))    
start_time = time.time()

#Tolerancia a ataques dirigidos priorizando nodos de mayor grado
Fracc_nodos_eliminar = np.linspace(0, 1, puntos + 1)#valores de probab
compara = 1 - Fracc_nodos_eliminar

Cant_nodos_eliminar = np.floor(Fracc_nodos_eliminar * numero_nodos_inic)
Prob_pert_GC=np.zeros(Fracc_nodos_eliminar.shape[0]) #almacenar valores de Robustez

degree_sequence = sorted((Degree for Node, Degree in NetworkBA.degree()), reverse=True)
degree_sequence=np.array(degree_sequence)

start_time = time.time()

print(puntos)
for ii in range(0,puntos): #para cada valor de prob_eliminar_nodo
    Network_Test=copy.deepcopy(NetworkBA) # es necesario usar esta copia para crear un nuevo objeto y no un enlace al anterior 
        
    if Cant_nodos_eliminar[ii] > 0: #si esto se eliminan X nodos de mayor grado
        
        min_max_grado_a_elimin = degree_sequence[int(Cant_nodos_eliminar[ii])-1] #menor valor de grado de los primeros X nodos de mayor grado
        posic_nodos_a_elim=np.where(DegreeValues > min_max_grado_a_elimin) #me aseguro los mayores
        
        posic_nodos_a_elim2=np.where(DegreeValues == min_max_grado_a_elimin) #de aca se saca una parte
        #posic_nodos_a_elim2 = posic_nodos_a_elim2[0:int(Cant_nodos_eliminar[ii]) - len(posic_nodos_a_elim[0])] #posicion de los X nodos de mayor grado
        posic_nodos_a_elim2 = posic_nodos_a_elim2[0][0:int(Cant_nodos_eliminar[ii]) - len(posic_nodos_a_elim[0])] #posicion de los X nodos de mayor grado
        
        #unir ambos conjuntos
        posic_nodos_a_elim = np.append(posic_nodos_a_elim,posic_nodos_a_elim2)
        
        if len(posic_nodos_a_elim) > 0: #si esta lista esta vacia no hay nodos q eliminar  
        
            for gg in range(0,len(posic_nodos_a_elim)): #se pasan uno a uno para eliminarlos
                Network_Test.remove_node(lista_de_nodos[posic_nodos_a_elim[gg]])           
                
    # Calcula el numero de nodos del mayor componente conectado, el "Giant Component"
    Gcc = sorted(nx.connected_components(Network_Test), key=len, reverse=True)
    
    if len(Gcc) > 0: #si esta lista esta vacia no quedan nodos
        G0 = Network_Test.subgraph(Gcc[0]) #subgrafo generado por el Giant Comp
        numero_nodos_GC=G0.number_of_nodes()    
        Prob_pert_GC[ii] = numero_nodos_GC / numero_nodos_inic #Network_Test.number_of_nodes() #prob de un nodo pertenecer al GC
    else:
        Prob_pert_GC[ii]=0
        
    print("iteracion: " + str(ii))

Prob_pert_GC = Prob_pert_GC/Prob_GC_0  #normalizar

plt.figure()
plt.plot(Fracc_nodos_eliminar, Prob_pert_GC, 'b--', label='tolerancia')
plt.plot(Fracc_nodos_eliminar, compara,'r--', label='y=1-x')
plt.title("Tolerancia de muestra de la red BA")
plt.ylabel("P_inf (f) / P_inf (0)")
plt.xlabel("f")
plt.grid(True)
plt.legend()
plt.show()    

    
print("---exec time parte 2: %s seconds ---" % (time.time() - start_time))  

print(Prob_GC_0)