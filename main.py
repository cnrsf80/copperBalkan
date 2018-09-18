import os.path
import metrics
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import folium
import plotly.plotly as py
import sklearn.cluster as sk
import sklearn.decomposition as decomp
from networkx.algorithms import community
from mpl_toolkits.mplot3d import Axes3D


colors = ["red", "green", "blue", "yellow", "grey", "purple", "orange","grey","black"]

class cluster:

    def __init__(self, name="",index=[],color="red"):
        self.index = index
        self.name = name
        self.color=color

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



def trace_artefact_3d(data,clusters):
    pca = decomp.pca.PCA(n_components=3)
    pca.fit(data)
    newdata = pca.transform(data)

    lp=[]
    for c in clusters:
        for p in c.index:
            lp.append([newdata[p,0],newdata[p,1],newdata[p,2],dict(size=2, color=c.color)])




    fig = plt.figure(1, figsize=(4, 3),dpi=300)
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)


    plt.show()


def trace_artefact(G,clusters,data,label_col=""):

    pos=nx.spectral_layout(G,scale=5,dim=len(clusters))
    #pos=nx.spring_layout(G,iterations=500,dim=len(clusters))
    # nx.set_node_attributes(G,"name")

    if len(clusters)>0:
        i=0
        labels={}
        for c in clusters:
            # for k in c.index:
            #     G.nodes[k].name = data[label_col][k]
            #     labels[G.nodes[k]]=G.nodes[k]

            nx.draw_networkx_nodes(G,
                                   pos=pos,
                                   nodelist=c.index,
                                   alpha=0.6,node_size=200, node_color=c.color)

            i=i+1

        # nx.draw_networkx_labels(G, pos, labels=labels,font_size=10, font_family='sans-serif')

        #nx.draw_networkx_edges(G, pos, alpha=0.5)
    else:
        nx.draw(G)



    plt.axis('off')
    #plt.savefig('graph.pdf', format="pdf")
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

def create_clusters_from_spectralclustering(distance_matrix,n_clusters:np.int,method="precomputed"):
    comp = sk.SpectralClustering(n_clusters=n_clusters,affinity=method).fit(distance_matrix)

    clusters = []
    for i in range(n_clusters): clusters.append(cluster("cluster" + str(i), [],colors[i]))

    for i in range(len(comp.labels_)):
        clusters[comp.labels_[i]].index.append(i)

    return clusters


def create_clusters_from_girvannewman(G):
    comp=community.girvan_newman(G)
    clusters=[]
    i=0
    for partition in list(sorted(c) for c in next(comp)):
        clusters.append(cluster("girvan_newman",partition,colors[i]))
        i=i+1

    return clusters

#http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture.fit
# def create_gaussian_mixture_model(data,params):
#     models = [GaussianMixture(n_components=param).fit(data)
#               for param in params]
#     plt.plot(params, [m.bic(data) for m in models], label="BIC")
#     plt.plot(params, [m.aic(data) for m in models], label="AIC")
#     plt.show()
#     return models


#http://scikit-learn.org/stable/modules/generated/sklearn.cluster.dbscan.html#sklearn.cluster.dbscan
def create_clusters_from_dbscan(M,l_eps,min_elements):
    clusters = []
    models=[sk.DBSCAN(metric="precomputed",eps=eps,min_samples=min_elements,n_jobs=4)
                .fit(M) for eps in l_eps]

    model=models[0]
    core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
    core_samples_mask[model.core_sample_indices_] = True

    n_clusters_ = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
    for i in range(n_clusters_):clusters.append(cluster("cluster"+str(i),[],colors[i]))

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



for i in range(5):plt.clf()

#data = pd.read_excel("cnx013_supp_table_s1.xlsx").head(200)
data = pd.read_excel("donnéesCIPIA.xlsx")
data["Ref"]=data.index
data.index=range(len(data))

mes:np.matrix=data.loc[:,"Dim.1":"Dim.10"]
# models=create_gaussian_mixture_model(mes,params=np.arange(1,30))
# X=models[25].predict(data)
# exit(0)

#M0=create_matrix(data,0,"As (ppm)","Se (ppm)")
M0:np.matrix=create_matrix(data,0,"Dim.1","Dim.10")
plt.matshow(M0, cmap=plt.cm.Blues)
plt.show()

G0=nx.from_numpy_matrix(M0)

artefact_clusters=create_clusters_from_spectralclustering(mes,8,"nearest_neighbors")
#artefact_clusters=create_ncluster(G0,8);
#artefact_clusters=create_clusters_from_dbscan(M0,np.arange(0.5,4,0.25),2);
#artefact_clusters=create_clusters_from_girvannewman(G0);

for c in artefact_clusters:c.print(data,"Ref")
#for c in artefact_clusters:c.print(data,"Sample label")

#trace_artefact(G0,artefact_clusters,data,"Ref")
trace_artefact_3d(mes,artefact_clusters)


# M=create_site_matrix(data,artefact_clusters)
# G=nx.from_numpy_matrix(M)
# sites_clusters=create_clusters_from_asyncfluid(G,4)

#Affichage
# for c in sites_clusters:c.print(data,"Site")
# trace_artefact(G,sites_clusters,data,"Site")


#https://python-visualization.github.io/folium/docs-v0.6.0/
#mymap=folium.Map(location=[48,2],zoom_start=13)
#draw_site_onmap(mymap,G,sites_clusters,create_sites_df(data),"map.html")

