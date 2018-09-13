import os.path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import folium
from networkx.algorithms import community

colors = ["red", "green", "blue", "yellow", "grey", "purple", "orange"]

class cluster:
    name=""
    color="red"

    def __init__(self, name,index=[]):
        self.index = index
        self.name = name

    def contain(self,i):
        for n in self.index:
            if n == i:
                return True
        return False

    def add_index(self,index):
        self.index.append(index)

    def print(self,data,label_col=""):
        print("\nCluster:"+self.name)
        print(" + ".join(data[label_col][self.index]))



#Definitions
def distance(i,j):
    return np.linalg.norm(i.values - j.values).round(0)

v_seuil=np.vectorize(lambda x,y:1 if x<y else 0)


def create_matrix(data,seuil):
    if os.path.isfile("adjacence_artefact.csv") and len(data)>149:
        matrice=np.asmatrix(np.loadtxt("adjacence_artefact.csv",delimiter=" "),dtype=np.int16)
    else:
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


def trace_artefact(G,clusters,data,label_col=""):
    pos = nx.spring_layout(G)

    if len(clusters)>0:
        i=0
        for c in clusters:
            labels=data[label_col][c.index]
            nx.draw_networkx_nodes(G, pos, c.index, label=labels, node_size=20, node_color=colors[i])
            i=i+1

        nx.draw_networkx_edges(G, pos, alpha=0.5)
    else:
        nx.draw(G)

    plt.axis('off')
    plt.savefig('graph.pdf', format="pdf")
    plt.show()


def create_sites_df(data):
    sites = data.loc[:, "Site":"Type of site/context"]
    sites=sites.drop_duplicates("Site")
    return sites

def create_site_matrix(data,artefact_clusters):
    sites=create_sites_df(data)
    M = np.asmatrix(np.zeros((len(sites), len(sites))), dtype=np.int16)
    liste_sites = list(sites.Site)
    for c in artefact_clusters:
        for s_i in data.Site[c.index]:
            for s_j in data.Site[c.index]:
                if s_i!=s_j:
                    M[liste_sites.index(s_i),liste_sites.index(s_j)]=1
                    M[liste_sites.index(s_j), liste_sites.index(s_i)] = 1

    return M


def create_clusters_from_girvannewman(G):
    comp=community.girvan_newman(G)
    clusters=[]
    for partition in list(sorted(c) for c in next(comp)):
        clusters.append(cluster("girvan_newman",partition))

    return clusters


def create_two_clusters(G:nx.Graph):
    clusters = []
    partition = community.kernighan_lin_bisection(G, None, 20, weight='weight')

    i=0
    for p in partition:
        clusters.append(cluster("kernighan_lin - "+str(i),p))
        i=i+1
    return clusters


def create_ncluster(G:nx.Graph,target=4):
    clusters =[cluster("premier",G.nodes)]

    backup_G=G.copy()
    while len(clusters) < target:
       #on cherche le plus grand cluster
       maxlen=0
       k=-1
       for i in range(0,len(clusters)):
           if len(clusters[i].index)>maxlen:
               maxlen=len(clusters[i].index)
               k=i
        #On divise en deux le plus grand cluster et on le supprime
       G = backup_G.subgraph(clusters[k].index)
       clusters.remove(clusters[k])
       for c in create_two_clusters(G):clusters.append(c)


    return clusters


def create_clusters_from_asyncfluid(G,n_community):
    coms = community.asyn_fluidc(G, n_community, 500)
    partition = []
    for c in coms:
        partition.append(cluster("asyncfluid",c))
    return partition


def draw_site_onmap(mymap:folium.Map, G, sites_clusters, sites:pd.DataFrame ,file):
    for site in sites.iterrows():
        pos=[site[1]["Geo Latitude"],site[1]["Geo Longitude"]]
        folium.Marker(pos,site["Site"]).add_to(mymap)




data = pd.read_excel("cnx013_supp_table_s1.xlsx")

G0=nx.from_numpy_matrix(create_matrix(data,0))
artefact_clusters=create_ncluster(G0,40);

#artefact_clusters=create_clusters_from_girvannewman(G0);

for c in artefact_clusters:c.print(data,"Sample label")
#trace_artefact(G0,artefact_clusters,data,"Sample label")

M=create_site_matrix(data,artefact_clusters)
G=nx.from_numpy_matrix(M)
sites_clusters=create_clusters_from_asyncfluid(G,4)

#Affichage
for c in sites_clusters:c.print(data,"Site")
trace_artefact(G,sites_clusters,data,"Site")


#https://python-visualization.github.io/folium/docs-v0.6.0/
#mymap=folium.Map(location=[48,2],zoom_start=13)
#draw_site_onmap(mymap,G,sites_clusters,create_sites_df(data),"map.html")







