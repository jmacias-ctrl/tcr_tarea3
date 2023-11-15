# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 14:45:40 2022

@author: yasma
"""

import numpy as np
from scipy import stats
import networkx as nx
import matplotlib.pyplot as plt
import time
import copy
#Generar redes aleatorias=====================================================

# #numero de nodos
# N=1000
# #probabilidad de generar enlace
# p=0.03

#numero de nodos
N=1000
#probabilidad de generar enlace
p=5.982/(N-1)

print('probabilidad:',p)
#generar matriz de adyacencia de red inicializada con ceros
MA=np.zeros((N,N))


#generar los enlaces en la diagonal superior
for i in range(0,MA.shape[0]-1):
    for j in range(i+1,MA.shape[0]):
        MA[i,j]=stats.bernoulli.rvs(p)


#para construir la matriz de adyacencia completa es necesario tener los enlaces
#bidireccionales o sea m[i,j]=m[j,i]. para ello se usa la suma de la matriz triangular
#superior obtenida con su transpuesta.
MA=MA+MA.transpose()

#print(np.sum(MA,axis=0))

#==============================================================================
#obtener el grafo a partir de la matriz de adyacencia de la red post falla
Network = nx.from_numpy_array(MA)
Network.name = 'Red Aleatoria'
#==============================================================================
print("-------------------------------")
print("Información sobre la red:")
print("-------------------------------")
print("Numero de nodos: N = " + str(Network.number_of_nodes()))
print("-------------------------------")
print("Numero de enlaces: N = " + str(Network.number_of_edges()))
print("-------------------------------")

#Si la red es pequeña se puede observar su topología en una FIG================
if N < 11:
    nx.draw_networkx(Network, font_size=8)
    print("-------------------------------")
    plt.title(Network.name + ' con N=' + str(N))
    plt.show()
#==============================================================================

#distribucon del grado==================================================================================================
print("-------------------------------")
# # Obtiene el grado de cada nodo y lo ordena de manera descendente para mostrarlo en la gráfica
# degree_sequence = sorted((Degree for Node, Degree in Network.degree()), reverse=True)
# plt.figure()
# plt.bar(np.arange(Network.number_of_nodes()), degree_sequence, width=0.80, color='b' );
# plt.title("Grados de los nodos de " + Network.name + " (orden descendente)");
# plt.ylabel("Grado");
# plt.xlabel("ID");
# plt.show();
# Crea intervalos de bines para el histograma entre [-0.5, n+1.5] donde n es el número de nodos, i.e, n=number_of_nodes(), 
# con esto se puede calcular la distribución en los bines 0, 1, 2,..., n-1
BinEdges = np.arange(Network.number_of_nodes() + 1) - 0.5;
# Obtiene el grado de cada nodo y lo almacena para su uso
DegreeValues = [degree for nodes, degree in Network.degree()]
# Realiza el conteo con la función histogram de Numpy
[Counts, Bins] = np.histogram(DegreeValues, BinEdges)
# Calcula la frecuencia de ocurrencia de cada grado
pk = Counts / Network.number_of_nodes();

#Ajuste de la distribucion del grado=======================================
grado_promedio=np.mean(DegreeValues)
print("Grado promedio: " + str(grado_promedio))
Poisson_fit=stats.poisson.pmf(np.arange(Network.number_of_nodes()), grado_promedio)


# Grafica en escala lineal de la distribucion=====================================
plt.plot(np.arange(Network.number_of_nodes()), pk, 'bo', label='dist');
# Grafica del ajuste
plt.plot(np.arange(Network.number_of_nodes()), Poisson_fit, 'r', label='ajuste')
plt.title("Distribución y ajuste de grados de " + Network.name + " grado promedio " + str(grado_promedio));
plt.ylabel("p_k");
plt.xlabel("Grado");
plt.legend()
plt.show();


#concentrar alrededor del peak de Poisson pmf===================================
min_inp=max(np.ceil(grado_promedio/2),0)
max_inp=min(np.ceil(3*grado_promedio/2),Network.number_of_nodes())
escala=np.arange(min_inp,max_inp,1, dtype=int)
# Grafica en escala lineal de la distribucion
plt.plot(escala, pk[escala], 'bo', label='dist');
# Grafica del ajuste
plt.plot(escala, Poisson_fit[escala], 'r', label='ajuste')
plt.title("Distribución y ajuste de grados de " + Network.name + " grado promedio " + str(grado_promedio));
plt.ylabel("p_k");
plt.xlabel("Grado");
plt.legend()
plt.show();


# # Grafica en escala logaritmica 
# plt.loglog(escala, pk[escala], 'bo', label='dist');
# plt.loglog(escala, Poisson_fit[escala], 'r', label='ajuste')
# plt.title("Distribución y ajuste de grados de " + Network.name + " grado promedio " + str(grado_promedio));
# plt.ylabel("p_k");
# plt.xlabel("Grado");
# plt.legend()
# plt.show();

#Distribucion de distancias====================================================
# Genera un arreglo para almacenar las distancias =============================
Distances = np.array([]);
# Obtiene un "iterator" del tipo (source, dictionary) con un diccionario cada nodo con 
# todos los nodos destino de la red y sus distancias, todo ordenado de manera ascendente
# El diccionario está indexado por los nodos destino a cada nodo origen
DistanceIterator = nx.shortest_path_length(Network);
# Se itera por cada nodo fuente y en cada key del diccionario se sacan los valores de las distancias con .values()
for source, targets in DistanceIterator:
    # Elimina el nodo origen de la lista pues siempre es de distancia 0
    del targets[source]
    # Almacena las distancias en un vector
    Distances = np.append(Distances, list(targets.values()));
Bins = np.arange(np.max(Distances)) + 1; 
# Calcula y grafica el histograma usando a funcion hist de numpy 
plt.figure()
plt.hist(Distances, Bins, density=True) 
plt.title("Distribución de distancias");
plt.xlabel("Distancias");
plt.ylabel("frec relat");
plt.show();

#Si el grafo esta conectado
conectado = nx.is_connected(Network);

if conectado == True:
    print("distancia promedio: < d > = ", nx.average_shortest_path_length(Network))    
    print("Diámetro = ", nx.diameter(Network))
else:
    print("Red con al menos un nodo desconectado, distancia promedio y diámetro infinitos")
    Gcc = sorted(nx.connected_components(Network), key=len, reverse=True)
    G0 = Network.subgraph(Gcc[0]) #subgrafo generado por el Giant Comp
    print("distancia promedio: < d > = ", nx.average_shortest_path_length(G0))    
    print("Diámetro = ", nx.diameter(G0))


#numero de componentes del grafo===============================================
print("Número de componentes: ", nx.number_connected_components(Network))


#Indexa solo los nombres de los nodos ===============================================================================
#print(Reuna.degree())
NodeNames = nx.clustering(Network).keys()
# Indexa solo los grados de los nodos
ClusteringCoef = nx.clustering(Network).values()
plt.bar(NodeNames, ClusteringCoef, width=0.80, color='b' )
plt.title("Coeficiente de agrupamiento de cada nodo")
plt.ylabel("C(i)")
plt.xlabel("Nodo"); plt.xticks(rotation=90);
plt.show();
print("Coeficiente de agrupamiento promedio: <C> = ", nx.average_clustering(Network) )

#grados que aparecen en la red asociados a cada nodo
DegreeValues = [degree for nodes, degree in Network.degree()]
lista_de_nodos=list(Network.nodes)

#===============================================================================
#===============================================================================
#Robustez a fallas aleatorias para diferentes valores de f = prob de eliminar un nodo
#===============================================================================
#===============================================================================
# Calcula el numero de nodos del mayor componente conectado, el "Giant Component" de la red inicial
Gcc = sorted(nx.connected_components(Network), key=len, reverse=True)
G0 = Network.subgraph(Gcc[0]) #subgrafo generado por el Giant Comp
numero_nodos_GC=G0.number_of_nodes()    
Prob_GC_0 = numero_nodos_GC / Network.number_of_nodes()  #prob de un nodo pertenecer al GC


puntos=100
prob_eliminar_nodo = np.linspace(0, 1, puntos + 1)#valores de probab
compara = 1 - prob_eliminar_nodo
Prob_pert_GC=np.zeros(prob_eliminar_nodo.shape[0]) #almacenar valores de Robustez
numero_nodos_inic=Network.number_of_nodes()

start_time = time.time()

for ii in range(0,puntos): #para cada valor de prob_eliminar_nodo
    Network_Test=copy.deepcopy(Network) # es necesario usar esta copia para crear un nuevo objeto y no un enlace al anterior 
        
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

degree_sequence = sorted((Degree for Node, Degree in Network.degree()), reverse=True)
degree_sequence=np.array(degree_sequence)

start_time = time.time()

print(puntos)
for ii in range(0,puntos): #para cada valor de prob_eliminar_nodo
    Network_Test=copy.deepcopy(Network) # es necesario usar esta copia para crear un nuevo objeto y no un enlace al anterior 
        
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