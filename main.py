# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 12:19:10 2022

@author: yasma
"""

import os as os 
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy.special as sp
import time


#RED que se analizara
NetworkName = 'twitchgamers'
# NetworkName = 'GtsCe' #no libre de escala
# NetworkName = 'AttMpls' #no libre de escala

# Importa el grafo como un grafo no orientado=====================================================================
#File2Read = os.path.join(NetworkName + ".gml")
#Network = nx.read_gml( NetworkName + ".gml")
# La siguiente línea después del comentario es necesaia porque algunos archivos GML tienen múltiples enlaces 
# para algunos pares de nodos. NetworkX lee esta topologías como del tipo multigraph pues el archivo GML tiene el keyword
#  multigraph 1
# en la sección "graph [". Por esto es necesaria, de lo contrario no sería necesario
#Network = nx.Graph(Network)
Data = open('large_twitch_edges.csv', "r")
next(Data, None)  
Graphtype = nx.Graph()

Network = nx.parse_edgelist(Data, delimiter=',', create_using=Graphtype,
                      nodetype=int, data=(('weight', float),))
# Agregamos un nombre a la red
Network.name = NetworkName
print("-------------------------------")
print("Información sobre la red:")
print("-------------------------------")
print("Numero de nodos: N = " + str(Network.number_of_nodes()))
print("-------------------------------")
print("Numero de enlaces: N = " + str(Network.number_of_edges()))
print("-------------------------------")

#grados que aparecen en la red asociados a cada nodo
DegreeValues = [degree for nodes, degree in Network.degree()]

#mayor grado observado en la red
k_max=np.max(DegreeValues)


#Escalado y espaciado lineal en histograma===================================
# Crea espaciado para el histograma entre [-0.5, n+1.5] donde n es el grado maximo de la red
# con esto se puede calcular la distribución en los bines 0, 1, 2,..., n-1
bins_lineal= np.arange(k_max + 1) - 0.5;
# Realiza el conteo con la función histogram de Numpy
[hist_lin, Bins] = np.histogram(DegreeValues, bins_lineal)
# Calcula la frecuencia de ocurrencia de cada grado
pk = hist_lin / np.sum(hist_lin)

# plt.figure()
# plt.plot(np.arange(k_max)+1, pk, 'bo');
# plt.title("Distribución de grados de " + NetworkName + " (lineal)");
# plt.ylabel("p_k");
# plt.xlabel("Grado");
# plt.show();

#=============================================================================


#Escalado logaritmico y espaciado lineal en histograma=======================
# plt.figure()
# plt.loglog(np.arange(k_max)+1, pk, 'bo');
# plt.title("Distribución de grados de " + NetworkName  + " (log)");
# plt.ylabel("p_k");
# plt.xlabel("Grado");
# plt.show();
#=============================================================================


#Escalado y espaciado logaritmico en histograma=======================
espacios=25
bins = np.logspace(0, np.log10(k_max) + 1, espacios)#extremos de los bins
widths = (bins[1:] - bins[:-1]) #ancho de los bins

# Calculate histogram
hist = np.histogram(DegreeValues, bins=bins)
# normalize by bin width
hist_norm = hist[0]/widths
hist_norm = hist_norm/ np.sum(hist_norm)


# Grafica en escala logaritmica espaciado lineal para comparar=================
# plt.figure()
# plt.loglog(np.arange(k_max)+1, pk, 'bo', label='espac lin');

puntos_hist_log=(bins[1:]+bins[:-1])/2 #centro de los bins logaritmicos
# plt.plot(puntos_hist_log, hist_norm,'ro', label='espac log')
# plt.title("Distribución de grados de " + NetworkName  + " (log)");
# plt.ylabel("p_k");
# plt.xlabel("Grado");
# plt.legend()
# plt.xscale('log')
# plt.yscale('log')
# plt.show();
# #=============================================================================

# #ajuste de la distribucion====================================================
# #calculo del exponente mediante regresion lineal, de acuerdo a ecuacion
# # log (p_k) = -gamma * log (k) + C

# min_val_permit=0 #generalemnete la ley de potencia se cumple a partir de cierto valor minimo
# max_val_permit=6 #los valores de log (p_k) a partir de este indice son -inf

# k_val = np.log10(puntos_hist_log[min_val_permit:max_val_permit])

# pk_datos_log = -np.log10(hist_norm[min_val_permit:max_val_permit])

# A = np.vstack([k_val, np.ones(len(k_val))]).T #A contiene info de k y de C

# Gamma, c = np.linalg.lstsq(A, pk_datos_log, rcond=None)[0]


# #Del ajuste anterior se utiliza la pendiente 'Gamma' y el otro parametro se ajusta
# # de acuerdo a cumplir con la condicion ppal de las PMF (suma 1)

# # pk_ajust=np.power(puntos_hist_log,-Gamma)/np.sum(np.power(puntos_hist_log,-Gamma))

# min_k=np.floor(bins[min_val_permit]) #grado minimo que corresponde a primer valor 
# #de log10 (hist_norm) diferente de -inf
# puntos_ajuste = np.arange(min_k,k_max)
# # pk_ajust=np.power(puntos_ajuste,-Gamma)/np.sum(np.power(puntos_ajuste,-Gamma))
# # hurtwitz_f=sp.zeta(Gamma,1)
# hurtwitz_f=sp.zeta(Gamma,min_k)
# pk_ajust2=np.power(puntos_ajuste,-Gamma)/hurtwitz_f
# #=============================================================================


#=============================================================================

# #ajuste con Clauset===================================================

# # k_min=4

posib_kmin=np.arange(200,300)
D=np.zeros(posib_kmin.shape[0])
Gamma_est=np.zeros(posib_kmin.shape[0])
DegreeValues = np.array(DegreeValues)#convertir de lista a array para aplicar lo de abajo

for ii in range(0,posib_kmin.shape[0]):

    k_min=posib_kmin[ii]

    DegreeValues_min = DegreeValues[np.where(DegreeValues >= k_min)]
        
    N=DegreeValues_min.shape[0]
    
    Gamma = 1 + N / np.sum(np.log(DegreeValues_min/(k_min-1/2)))
    
    #Del ajuste anterior se utiliza la pendiente 'Gamma' y el otro parametro se ajusta   
    #calculo de estadístico de Kolmogorov-Smirnov
    #CDFs
    #CDF Modelo (solo para valores mayores de k_min) 
    puntos_ajuste = np.arange(k_min, k_max)
    hurtwitz_f=sp.zeta(Gamma,k_min)
    pk_ajust2=np.power(puntos_ajuste,-Gamma)/hurtwitz_f
    CDF_ajuste=np.cumsum(pk_ajust2, axis=0)
    
    #CDF datos (solo para observaciones mayores de k_min)
    hist_lin2=hist_lin[k_min:]
    pk_datos=hist_lin2/np.sum(hist_lin2)
    CDF_datos=np.cumsum(pk_datos, axis=0)
        
    D[ii]=np.max(np.absolute(CDF_ajuste-CDF_datos))
    Gamma_est[ii]=Gamma
    
#Comportamiento de D y Gamma en funcion de kmin================
# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.plot(posib_kmin, D, color="red", marker="o")
# set x-axis label
ax.set_xlabel("KMIN", fontsize = 14)
# set y-axis label
ax.set_ylabel("D", color="red", fontsize=14)

# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(posib_kmin, Gamma_est, color="blue", marker="o")
ax2.set_ylabel("Gamma",color="blue",fontsize=14)
plt.xscale('log')
plt.show()


#valores ajustados=======
KMIN=posib_kmin[np.where(D == np.min(D))] 

DegreeValues_min = DegreeValues[np.where(DegreeValues >= KMIN)]

N=DegreeValues_min.shape[0]

Gamma = 1 + N / np.sum(np.log(DegreeValues_min/(KMIN-1/2)))

print("valor de Gamma=" + str(round(Gamma,2)) + " y KMIN=" + str(KMIN[0]));

#PMF ajustada
puntos_ajuste = np.arange(KMIN, k_max)
hurtwitz_f=sp.zeta(Gamma,KMIN[0])
pk_ajust2=np.power(puntos_ajuste,-Gamma)/hurtwitz_f

#=============================================================================

#Comparacion entre histograma log, hist lineal y pmf ajustada de ley de potencia================
plt.figure()
plt.plot(np.arange(k_max)+1, pk,'bo', label='histo lin')
plt.plot(puntos_ajuste, pk_ajust2,'y--', label='ajusteMLE')
plt.plot(puntos_hist_log, hist_norm,'ro', label='espac log')
#para MSE
# plt.title("ajuste de dist grado Gamma=" + str(round(Gamma,2)) + ' espacios=' + str(espacios) + ' kmin=' + str(min_k));
#para MLE
plt.title("ajuste de dist grado Gamma=" + str(round(Gamma,2)) + ' kmin=' + str(KMIN[0]));
plt.ylabel("p_k");
plt.xlabel("Grado");
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.show();

#Comparacion entre CDFs hist lineal y pmf ajustada de ley de potencia================
CDF_ajuste=np.cumsum(pk_ajust2, axis=0)

# hist_lin2=hist_lin[KMIN[0]:] #para ajuste MLE
#hist_lin2=hist_lin # para ajuste MSE
hist_lin2=hist_lin[KMIN[0]:] # para ajuste MLE
pk_datos=hist_lin2/np.sum(hist_lin2)
CDF_datos=np.cumsum(pk_datos, axis=0)

plt.figure()
plt.plot(np.arange(KMIN,k_max), CDF_datos,'bo', label='CDF lin')#para ajuste MLE
# plt.plot(np.arange(k_max)+1, CDF_datos,'bo', label='CDF lin')# para ajuste MSE
plt.plot(puntos_ajuste, CDF_ajuste,'r--', label='ajusteMLE')
plt.title("Comparacion de CDFs Gamma=" + str(round(Gamma,2)));
plt.ylabel("p_k");
plt.xlabel("Grado");
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.show();

# CDF_s complemenetarias=====================================================
CDF_ajuste_comp=1-CDF_ajuste
CDF_datos_comp=1-CDF_datos

plt.figure()
plt.plot(np.arange(KMIN,k_max), CDF_datos_comp,'bo', label='CDF lin')#para ajuste MLE
# plt.plot(np.arange(k_max)+1, CDF_datos,'bo', label='CDF lin')# para ajuste MSE
plt.plot(puntos_ajuste, CDF_ajuste_comp,'r--', label='ajusteMLE')
plt.title("CDFs complementarias P(K>k) Gamma=" + str(round(Gamma,2)));
plt.ylabel("p_k");
plt.xlabel("Grado");
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.show();

# #=============================================================================
# #=============================================================================
# # Test de Bondad de ajuste===============================================================
# #=============================================================================
# #=============================================================================
# #para comprobar tiempo de ejecucion del programa
# start_time = time.time()

# #PAso 1==================================
# #Ajuste de los datos a power-law y obtener Gamma, KMIN y D
# Gamma_sample=Gamma
# KMIN_sample=KMIN
# D_sample=np.min(D)
# hurtwitz_f_sample=sp.zeta(Gamma_sample,KMIN_sample[0])


# #PAso 2====================================
# # Generar Lo=2500 conjuntos de datos provenientes de una ley de potencia con los mismos 
# # parámetros del PAso 1. Cada uno de estos conjuntos tienen que ajustarse a una ley de 
# # potencia y respecto a este ajuste calcular D

# Lo=2500
# D_opt=np.zeros(Lo) 
# Gamma_est2=np.zeros(Lo)

# # Tamaño de la muestra de cada conjunto tam_muestra
# tam_muestra=DegreeValues.shape[0]
# #cantidad de valores mayores que KMIN
# n_tail=DegreeValues[np.where(DegreeValues >= KMIN_sample)].shape[0]
# #probabilidad de generar power-law
# p_power_law = n_tail/tam_muestra
# #distribucion bajo KMIN contiene al cero en la posicion 0
# PMF_bajo_KMIN = hist_lin[0:KMIN_sample[0]]/np.sum(hist_lin[0:KMIN_sample[0]])
# dist_bajo_KMIN = np.cumsum(PMF_bajo_KMIN, axis=0)

# for jj in range(0,Lo):
#     #Generar un conjunto
#     # los datos se generan de acuerdo a Clausset 2009

#     #inicializar muestra generada
#     muestra_i=np.zeros(tam_muestra) 
    
#     for ii in range(0,tam_muestra):
#         if np.random.uniform(0,1) > p_power_law:
#             #dist no power-law
#             p_grado=np.random.uniform(0,1)
#             aux=0
#             flag=0
#             while flag == 0: # recorro la CDF
#                 if p_grado < dist_bajo_KMIN [aux]:
#                     muestra_i[ii]=aux
#                     flag=1
#                 else:
#                     aux+=1
#         else:
#             #dist power-law
#             p_grado=np.random.uniform(0,1)
#             aux=KMIN_sample[0]
#             flag=0
#             p=np.power(KMIN_sample,-Gamma_sample)/hurtwitz_f_sample
#             F=p
#             while flag == 0: # recorro la CDF
#                 if p_grado < F:
#                     muestra_i[ii]=aux
#                     flag=1
#                 else:                
#                     p = p * np.power(1 + 1/aux,-Gamma_sample)
#                     F+=p
#                     aux+=1
    
#     # ajustar el conjunto muestra_i a una ley de potencia
#     k_max=np.max(muestra_i)
#     posib_kmin=np.arange(np.min(muestra_i),100)
#     D=np.zeros(posib_kmin.shape[0])
#     muestra_i = np.array(muestra_i)#convertir de lista a array para aplicar lo de abajo
    
#     #histograma lineal
#     bins_lineal= np.arange(k_max + 1) - 0.5;
#     # Realiza el conteo con la función histogram de Numpy
#     [hist_lin, Bins] = np.histogram(muestra_i, bins_lineal)
#     # Calcula la frecuencia de ocurrencia de cada grado
#     pk = hist_lin / np.sum(hist_lin)
    
    
#     for ii in range(0,posib_kmin.shape[0]):
    
#         k_min=posib_kmin[ii]
    
#         muestra_i_min = muestra_i[np.where(muestra_i >= k_min)]
            
#         N=muestra_i_min.shape[0]
        
#         Gamma = 1 + N / np.sum(np.log(muestra_i_min/(k_min-1/2)))
        
#         #Del ajuste anterior se utiliza la pendiente 'Gamma' y el otro parametro se ajusta   
#         #calculo de estadístico de Kolmogorov-Smirnov
#         #CDFs
#         #CDF Modelo (solo para valores mayores de k_min) 
#         puntos_ajuste = np.arange(k_min, k_max)
#         hurtwitz_f=sp.zeta(Gamma,k_min)
#         pk_ajust2=np.power(puntos_ajuste,-Gamma)/hurtwitz_f
#         CDF_ajuste=np.cumsum(pk_ajust2, axis=0)
        
#         #CDF datos (solo para observaciones mayores de k_min)
#         hist_lin2=hist_lin[int(k_min):]
#         pk_datos=hist_lin2/np.sum(hist_lin2)
#         CDF_datos=np.cumsum(pk_datos, axis=0)
            
#         D[ii]=np.max(np.absolute(CDF_ajuste-CDF_datos))
        
#     #almacenar valor de D_min=======
#     D_opt[jj]=np.min(D)
    
#     #valores ajustados=======
#     KMIN=posib_kmin[np.where(D == np.min(D))] 

#     muestra_i_min = muestra_i[np.where(muestra_i >= KMIN)]

#     N=muestra_i_min.shape[0]

#     Gamma = 1 + N / np.sum(np.log(muestra_i_min/(KMIN-1/2)))
    
#     Gamma_est2[jj]=Gamma
#     print("iteracion=" + str(jj))# + " y D=" + str(np.min(D)));

# total_mayores= D_opt[np.where(D_opt >= D_sample)].shape[0]        
# p_value=total_mayores / Lo

# print("---exec time: %s seconds ---" % (time.time() - start_time))
# print("---p_value=%s ---" % p_value)

# #histograma de D
# plt.figure()
# plt.hist(D_opt)
# plt.title("Distribución estadistico KS");
# plt.ylabel("p(D)");
# plt.xlabel("D");
# plt.show();


# #=============================================================================


# #Predicciones del modelo y resultados de los datos============================

# #observaciones
# # Momento_1orden=np.mean(DegreeValues)
# # print("Momento de 1er orden calculado: " + str(Momento_1orden))

# # Momento_2orden=np.var(DegreeValues) + np.power(Momento_1orden,2)
# # print("Momento de 2do orden calculado: " + str(Momento_2orden))

# # Kmax=np.max(DegreeValues)
# # print("Maximo grado observado: " + str(Kmax))

# # Kmax_pred = np.min(DegreeValues) * np.power(Network.number_of_nodes(),1/(Gamma-1))
# # print("Maximo grado predicho: " + str(Kmax_pred))

# # #Si el grafo esta conectado
# # conectado = nx.is_connected(Network);

# # if conectado == True:
# #     print("distancia promedio observada: < d > = ", nx.average_shortest_path_length(Network))    
# #     print("Diámetro = ", nx.diameter(Network))
# # else:
# #     print("Red con al menos un nodo desconectado, distancia promedio y diámetro infinitos")


# # distancia_prom_pred=np.log(np.log(Network.number_of_nodes()))
# # print("distancia promedio predicha: < d > = " + str(distancia_prom_pred))
# # #=============================================================================

