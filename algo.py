import draw
import metrics
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import community
import sklearn.cluster as sk

colors=[]
for i in range(200):colors.append(i)



#Represente un model après entrainement
class model:

    name=""

    def __init__(self, data,name_col=0,mesures_col=range(1,5)):
        self.name_col=name_col
        self.mesures_col=mesures_col
        self.clusters=[]
        self.data=data


    def print_cluster(self):
        for c in self.clusters:
            c.print(self.data, self.name_col)

    def trace(self,filename):
        draw.trace_artefact_3d(self.mesures(), self.clusters, filename)
        draw.trace_artefact_2d(self.mesures(), self.clusters, filename)
        self.print_cluster()

    def mesures(self):
        return self.data.iloc[:,self.mesures_col]

    def init_metrics(self,test):
        self.homogeneity_score=metrics.homogeneity_score(test,self.clusters)
        self.silhouette_score=metrics.silhouette_score(self.mesures(),self.clusters)



#definie un cluster
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





def create_clusters_from_spectralclustering(model:model,n_clusters:np.int,method="precomputed"):
    model.name="spectralclustering"

    mes=model.mesures()
    comp = sk.SpectralClustering(n_clusters=n_clusters,affinity=method).fit(mes)

    for i in range(n_clusters): model.clusters.append(cluster("cluster" + str(i), [],colors[i]))

    for i in range(len(comp.labels_)):
        model.clusters[comp.labels_[i]].index.append(i)

    return model



def create_clusters_from_girvannewman(G):
    comp=community.girvan_newman(G)
    clusters=[]
    i=0
    for partition in list(sorted(c) for c in next(comp)):
        clusters.append(cluster("girvan_newman",partition,colors[i]))
        i=i+1

    return model("girvannewman",clusters)


#http://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture.fit
# def create_gaussian_mixture_model(data,params):
#     models = [GaussianMixture(n_components=param).fit(data)
#               for param in params]
#     plt.plot(params, [m.bic(data) for m in models], label="BIC")
#     plt.plot(params, [m.aic(data) for m in models], label="AIC")
#     plt.show()
#     return models


#http://scikit-learn.org/stable/modules/generated/sklearn.cluster.dbscan.html#sklearn.cluster.dbscan
def create_clusters_from_dbscan(mod:model,eps,min_elements=1):
    mod.name="dbscan"

    model:sk.DBSCAN=sk.DBSCAN(eps=eps,min_samples=min_elements,n_jobs=4).fit(mod.mesures())

    core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
    core_samples_mask[model.core_sample_indices_] = True

    n_clusters_ = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
    for i in range(n_clusters_):
        mod.clusters.append(cluster("cluster"+str(i),[],colors[i]))


    i=0
    for l in model.labels_:
        if l>=0:
            mod.clusters[l].index.append(i)
        i=i+1




    return mod


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
