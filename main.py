import os.path
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from community.community_louvain import best_partition


def distance(i,j):
    return np.linalg.norm(i.values - j.values).round()

v_seuil=np.vectorize(lambda x,y:1 if x<y else 0)

data=pd.read_excel("cnx013_supp_table_s1.xlsx")
composition=data.loc[:,"As (ppm)":"Se (ppm)"]

if os.path.isfile("adjacence_artefact.csv"):
    matrice=np.loadtxt("adjacence_artefact.csv",delimiter=" ")
else:
    matrice = np.zeros((len(data),len(data)))

    for i in composition.index:
        for j in composition.index:
            if(matrice[i,j]==0 and i!=j):
                d=distance(composition.ix[i],composition.ix[j])
                matrice[i,j]=d
                matrice[j,i]=d

    np.savetxt("adjacence_artefact.csv",matrice)

matrice_ww=v_seuil(matrice,700)
G = nx.from_numpy_matrix(matrice_ww)
nx.draw(G)

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
plt.show()
