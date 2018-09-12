import os.path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import community


class cluster:
    name=""
    labels=[]
    color="red"

    def __init__(self, name, index,nodes, color):
        self.nodes = []
        self.name = name
        for i in index:
            self.nodes.append(nodes[i])
            self.labels.append(i)
        self.color= color

    def contain(self,i):
        for n in nodes:
            if n.ix == i:return True



#Definitions
def distance(i,j):
    return np.linalg.norm(i.values - j.values).round(0)

v_seuil=np.vectorize(lambda x,y:1 if x<y else 0)


def create_matrix(seuil,limit):
    if os.path.isfile("adjacence_artefact.csv") and limit>200:
        matrice=np.asmatrix(np.loadtxt("adjacence_artefact.csv",delimiter=" "),dtype=np.int16)
    else:
        data = pd.read_excel("cnx013_supp_table_s1.xlsx").head(limit)
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


def trace_artefact(G,partitions):
    colors = ["red", "green", "blue", "yellow", "grey", "purple", "orange"]
    pos = nx.spring_layout(G)
    for i in range(0, len(partitions)):
        c = cluster("", partitions[i], list(G.nodes), colors[i])
        nx.draw_networkx_nodes(G, pos, c.nodes, label=c.labels, node_size=20, node_color=c.color)

    nx.draw_networkx_edges(G, pos, alpha=0.5)

    plt.axis('off')
    plt.savefig('graph.pdf', format="pdf")
    plt.show()


def create_site_matrix(clusters,limit):
    data = pd.read_excel("cnx013_supp_table_s1.xlsx").head(limit)
    sites=data.loc[:"Site":"Type of site/context"]
    sites.drop_duplicates()
    M = np.asmatrix(np.zeros((len(sites), len(sites))), dtype=np.int16)
    for i in sites.index:
            for j in sites.index:
                if(M[i,j]==0 and i!=j):
                    for c in clusters:
                        if c.contain(i) and c.contain(j):
                            M[i,j]=1
                            M[j,i]=1



limit=50
n_community=4

G=nx.from_numpy_matrix(create_matrix(700,limit))
#partition = community.kernighan_lin_bisection(G,None,20,weight='weight')

#comp=community.girvan_newman(G)
#partition = tuple(sorted(c) for c in next(comp))

coms=community.asyn_fluidc(G,n_community,100)
partition=[]
for c in coms:
    partition.append(c)

create_site_matrix(partition,limit)




