import os.path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import community


#Definitions
def distance(i,j):
    return np.linalg.norm(i.values - j.values).round(0)

v_seuil=np.vectorize(lambda x,y:1 if x<y else 0)


def create_matrix(seuil):
    if os.path.isfile("adjacence_artefact.csv"):
        matrice=np.asmatrix(np.loadtxt("adjacence_artefact.csv",delimiter=" "),dtype=np.int16)
    else:
        data = pd.read_excel("cnx013_supp_table_s1.xlsx")
        composition = data.loc[:, "As (ppm)":"Se (ppm)"]
        matrice = np.asmatrix(np.zeros((len(data),len(data))),dtype=np.int16)

        for i in composition.index:
            for j in composition.index:
                if(matrice[i,j]==0 and i!=j):
                    d=distance(composition.ix[i],composition.ix[j])
                    matrice[i,j]=d
                    matrice[j,i]=d

        np.savetxt("adjacence_artefact.csv",matrice)

    if(seuil==0):
        matrice_ww=matrice
    else:
        matrice_ww=v_seuil(matrice,seuil)

    return matrice_ww


G=nx.from_numpy_matrix(create_matrix(0))
partition = community.kernighan_lin_bisection(G,None,10,weight='weight')

#drawing
size = float(len(set(partition.values())))
pos = nx.spring_layout(G)
count = 0.
for com in set(partition.values()) :
    count = count + 1.
    list_nodes = [nodes for nodes in partition.keys()
                                if partition[nodes] == com]
    nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
                                node_color = str(count / size))


nx.draw_networkx_edges(G,pos, alpha=0.5)
plt.show()


# pos=nx.spring_layout(G)
#
# nx.draw(G,node_size=5,node_color="blue")
#
# plt.axis('off')
# plt.savefig('filename.png', dpi = 300)

#
# partition=best_partition(G)
#
# #drawing
# size = float(len(set(partition.values())))
# pos = nx.spring_layout(G)
# count = 0.
# for com in set(partition.values()) :
#     count = count + 1.
#     list_nodes = [nodes for nodes in partition.keys()
#                                 if partition[nodes] == com]
#     nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 5,
#                                 node_color = str(count / size))
#
#
# nx.draw_networkx_edges(G,pos, alpha=0.5)
# plt.show()
