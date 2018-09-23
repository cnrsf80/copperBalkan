from sklearn.cluster import estimate_bandwidth, MeanShift

import os
import draw
import time
import sklearn.metrics as metrics
import networkx as nx
import numpy as np

from networkx.algorithms import community
import sklearn.cluster as sk

from tools import tirage

colors=[]
for i in range(200):colors.append(i)





#Represente un model après entrainement
class model:
    name=""
    delay:int=0 #delay en secondes
    silhouette_score:int=0
    score:int=0

    def __init__(self, data,name_col=0,mesures_col=range(1,5)):
        self.name_col=name_col
        self.mesures_col=mesures_col
        self.clusters=[]
        self.data=data

    def init_distances(self,func_distance,force=False):
        print("Calcul de la matrice de distance")
        size=len(self.mesures())
        composition=self.mesures()

        namefile="./saved/matrix_distance_"+self.name
        if os.path.isfile(namefile) and force==False:
            self.distances=np.load(namefile)
        else:
            self.distances = np.asmatrix(np.zeros((len(composition.index), len(composition.index))))
            for i in range(size):
                print(size-i)
                for j in range(size):
                    if(self.distances[i,j]==0 and i!=j):
                        name_i=self.data[self.name_col][i]
                        name_j=self.data[self.name_col][j]
                        vecteur_i=composition.iloc[i].values
                        vecteur_j=composition.iloc[j].values
                        d=func_distance(vecteur_i,vecteur_j,name_i,name_j)
                        self.distances[i,j]=d
                        self.distances[j,i]=d

            np.save(namefile,self.distances)


    def print_cluster(self):
        s=""
        for c in self.clusters:
            s=s+c.print(self.data, self.name_col)+"\n"
        return s

    def start_treatment(self):
        self.delay=time.time()

    def end_treatment(self):
        self.delay=round((time.time()-self.delay)*10)/10

    def trace(self,filename,label_col_name="",url_base=""):
        title=self.print_perfs()+"\n"+self.print_cluster()
        s=("<a href='"+url_base+"/"+draw.trace_artefact_3d(self.mesures(), self.clusters, filename,label_col_name,title))+"'>représentation 3D</a>\n"
        #s=s+("<a href='"+url_base + "/" + draw.trace_artefact_2d(self.mesures(), self.clusters, filename,label_col_name))+"'>représentation 2D</a>\n"
        return self.print_perfs()+"\n"+s+"\n"

    def cluster_toarray(self):
        rc:np.ndarray=[0]*len(self.data)
        for k in range(len(self.clusters)):
            for i in self.clusters[k].index:
                rc[i]=k
        return rc

    def mesures(self):
        return self.data.iloc[:,self.mesures_col]


    def init_metrics(self,test=None):
        if len(self.clusters)>1:
            self.silhouette_score= metrics.silhouette_score(self.mesures(), self.cluster_toarray())
            self.score=round(self.silhouette_score*20*10)/10
        else:
            self.score=0

    def clusters_from_labels(self,labels):
        n_clusters_=max(labels)+1
        for i in range(n_clusters_):
            self.clusters.append(cluster("cl_" + str(i), [], colors[i]))

        i = 0
        for l in labels:
            if l >= 0:
                self.clusters[l].add_index(i,self.data,self.name_col)
            i = i + 1


    def print_perfs(self):
        s=("Algorithme : %s" % self.name)+"\n"
        s=s+("Nombre de clusters : %s" % len(self.clusters))+"\n"
        s = s +("Delay de traitement : %s sec" % self.delay)+"\n"
        if self.silhouette_score>0:
            s=s+("Silhouette score %s" % self.silhouette_score)+"\n"

        s = s +("Score %s" % self.score)+"\n"
        return s


#definie un cluster
class cluster:
    def __init__(self, name="",index=[],color="red"):
        self.index = index
        self.name = name
        self.color=color
        self.labels=[]
        self.marker=tirage(['^','o','v','<','>','x','D','*'])

    def contain(self,i):
        for n in self.index:
            if n == i:
                return True
        return False

    def add_index(self,index,data=None,label_col=""):
        self.index.append(index)
        if data is not None:
            col=data[label_col]
            self.labels.append(col[index])

    def print(self,data,label_col=""):
        s=("Cluster:"+self.name+"\n")
        s=s+(" + ".join(data[label_col][self.index]))
        return s+"\n"



def create_clusters_from_spectralclustering(model:model,n_clusters:np.int,n_neighbors=10,method="precomputed"):
    model.name="spectralclustering avec n_cluster="+str(n_clusters)+" et n_neighbors="+str(n_neighbors)

    model.start_treatment()
    if method=="precomputed":
        comp = sk.SpectralClustering(n_clusters=n_clusters,affinity=method,n_neighbors=n_neighbors).fit(model.distances)
    else:
        comp = sk.SpectralClustering(n_clusters=n_clusters, affinity=method, n_neighbors=n_neighbors).fit(model.mesures())
    model.end_treatment()

    model.clusters_from_labels(comp.labels_)

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
def create_clusters_from_dbscan(mod:model,eps,min_elements,iter=100,metric="euclidean"):
    mod.name="dbscan avec eps="+str(eps)+" et min_elements="+str(min_elements)

    mod.start_treatment()
    if metric=="precomputed":
        model:sk.DBSCAN=sk.DBSCAN(eps=eps,min_samples=min_elements,n_jobs=4,metric=metric).fit(mod.distances)
    else:
        model: sk.DBSCAN = sk.DBSCAN(eps=eps, min_samples=min_elements, n_jobs=4, metric=metric).fit(mod.mesures())

    mod.end_treatment()

    mod.clusters_from_labels(model.labels_)
    return mod



def create_clusters_from_agglomerative(mod, n_cluster=10,affinity="euclidean"):
    mod.name = "Classification Hierarchique avec ncluster=" + str(n_cluster)

    mod.start_treatment()
    if affinity=="precomputed":
        model = sk.AgglomerativeClustering(n_cluster, affinity,linkage="complete").fit(mod.distances)
    else:
        model= sk.AgglomerativeClustering(n_cluster,affinity).fit(mod.mesures())

    mod.end_treatment()

    mod.clusters_from_labels(model.labels_)
    return mod


def create_model_from_meanshift(mod:model,quantile=0.2,min_bin_freq=1,method="euclidean"):
    mod.name = "meanshift avec quantile a "+str(quantile)+" & min_bin_freq="+str(min_bin_freq)

    mod.start_treatment()
    bandwidth = estimate_bandwidth(mod.mesures(), quantile=quantile)
    ms = MeanShift(bandwidth=bandwidth,min_bin_freq=min_bin_freq).fit(mod.mesures())
    mod.end_treatment()
    mod.clusters_from_labels(ms.labels_)

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

#https://people.eecs.berkeley.edu/~jordan/sail/readings/luxburg_ftml.pdf
