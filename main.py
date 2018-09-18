import os.path
import metrics
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import folium
import sklearn.cluster as sk
from sklearn.mixture import GaussianMixture, gaussian_mixture
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
    return np.linalg.norm(i.values - j.values)

v_seuil=np.vectorize(lambda x,y:1 if x<y else 0)


def create_matrix(data,seuil,start_col,end_col):
    if os.path.isfile("adjacence_artefact.csv") and len(data)>149:
        matrice=np.asmatrix(np.loadtxt("adjacence_artefact.csv",delimiter=" "))
    else:
        composition = data.loc[:, start_col:end_col]
        matrice = np.asmatrix(np.zeros((len(data),len(data))))

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
    M = np.asmatrix(np.zeros((len(sites), len(sites))))
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

#http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture.fit
def create_gaussian_mixture_model(data,params):
    models = [GaussianMixture(n_components=param).fit(data)
              for param in params]
    plt.plot(params, [m.bic(data) for m in models], label="BIC")
    plt.plot(params, [m.aic(data) for m in models], label="AIC")
    plt.show()


#http://scikit-learn.org/stable/modules/generated/sklearn.cluster.dbscan.html#sklearn.cluster.dbscan
def create_clusters_from_dbscan(M,l_eps,min_elements):
    clusters = []
    models=[sk.DBSCAN(metric="precomputed",eps=eps,min_samples=min_elements,n_jobs=4)
                .fit(M) for eps in l_eps]

    model=models[0]
    core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
    core_samples_mask[model.core_sample_indices_] = True

    n_clusters_ = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
    for i in range(n_clusters_):
        clusters.append(cluster("cluster"+str(i),[]))

    print('Estimated number of clusters: %d' % n_clusters_)

    i=0
    for l in model.labels_:
        if l>=0:
            clusters[l].index.append(i)
        i=i+1

    return clusters


def create_two_clusters(G:nx.Graph):
    clusters = []
    partition = community.kernighan_lin_bisection(G, None, 500, weight='weight')

    i=0
    for p in partition:
        clusters.append(cluster("kernighan_lin - "+str(i),p))
        i=i+1
    return clusters


def create_ncluster(G:nx.Graph,target=4):
    clusters =[cluster("premier",G.nodes)]
    print("Recherche des clusters")
    backup_G=G.copy()
    while len(clusters) < target:
       #on cherche le plus grand cluster
       print(target-len(clusters))
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




#data = pd.read_excel("cnx013_supp_table_s1.xlsx").head(200)
data = pd.read_excel("donnéesCIPIA.xlsx")
data["Ref"]=data.index
data.index=range(len(data))

mes=data.loc[:,"Dim.1":"Dim.10"]
create_gaussian_mixture_model(mes,params=range(1,40))
exit(0)

#M0=create_matrix(data,0,"As (ppm)","Se (ppm)")
M0=create_matrix(data,0,"Dim.1","Dim.10")
plt.matshow(M0, cmap=plt.cm.Blues)
plt.show()

G0=nx.from_numpy_matrix(M0)
#artefact_clusters=create_ncluster(G0,8);
#artefact_clusters=create_clusters_from_dbscan(M0,np.arange(0.5,4,0.25),2);
#artefact_clusters=create_clusters_from_girvannewman(G0);

for c in artefact_clusters:c.print(data,"Ref")
#for c in artefact_clusters:c.print(data,"Sample label")
#trace_artefact(G0,artefact_clusters,data,"Sample label")


exit(0)

M=create_site_matrix(data,artefact_clusters)
G=nx.from_numpy_matrix(M)
sites_clusters=create_clusters_from_asyncfluid(G,4)

#Affichage
for c in sites_clusters:c.print(data,"Site")
trace_artefact(G,sites_clusters,data,"Site")


#https://python-visualization.github.io/folium/docs-v0.6.0/
#mymap=folium.Map(location=[48,2],zoom_start=13)
#draw_site_onmap(mymap,G,sites_clusters,create_sites_df(data),"map.html")

