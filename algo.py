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
        if os.path.isfile(namefile+".npy") and force==False:
            self.distances=np.load(namefile+".npy")
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

    def trace(self,path,filename,label_col_name="",url_base=""):
        title=self.print_perfs()+"\n"+self.print_cluster()
        s=("<a href='"+url_base+"/"+draw.trace_artefact_3d(self.mesures(), self.clusters, path,filename,label_col_name,title))+"'>représentation 3D</a>\n"
        #s=s+("<a href='"+url_base + "/" + draw.trace_artefact_2d(self.mesures(), self.clusters, path,filename,label_col_name))+"'>représentation 2D</a>\n"
        return self.print_perfs()+"\n"+s+"\n"

    def cluster_toarray(self):
        rc:np.ndarray=[0]*len(self.data)
        for k in range(len(self.clusters)):
            for i in self.clusters[k].index:
                rc[i]=k
        return rc

    def mesures(self):
        return self.data.iloc[:,self.mesures_col]


    def init_metrics(self,labels_true):
        if len(self.clusters)>1:
            labels=self.cluster_toarray()
            self.silhouette_score= metrics.silhouette_score(self.mesures(), labels)
            self.rand_index=metrics.adjusted_rand_score(labels_true, labels)

            #self.self.adjusted_mutual_info_score=metrics.self.adjusted_mutual_info_score(labels_true,labels)

            self.homogeneity_score=metrics.homogeneity_score(labels_true,labels)
            self.completeness_score=metrics.completeness_score(labels_true,labels)
            self.v_measure_score=metrics.v_measure_score(labels_true,labels)

            self.score=(self.silhouette_score+(self.rand_index+1)/2+self.v_measure_score+self.homogeneity_score/2+self.completeness_score/2)/4
            self.score=round(self.score*20*100)/100
        else:
            self.silhouette_score=0
            self.score=0
            self.rand_index=0
            self.homogeneity_score=0
            self.completeness_score=0
            self.v_measure_score=0

        return self.print_perfs()

    def print_perfs(self):
        s=("<h2>Algorithme : %s</h2>" % self.name)+"\n"
        s = s + ("Delay de traitement : %s sec" % self.delay) + "\n"
        s=s+("Nombre de clusters : %s" % len(self.clusters))+"\n\n"

        if len(self.clusters)>1:
            s=s+"<a href='http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.cluster'>Indicateurs de performance du clustering</a>\n"
            s=s+("Silhouette score %s" % self.silhouette_score)+"\n"
            s=s+"Rand_index %s" % self.rand_index+"\n"
            #s=s+"Information mutuelle (https://fr.wikipedia.org/wiki/Information_mutuelle) : %s" % self.adjusted_mutual_info_score+"\n"
            s=s+"homogeneity_score %s" % self.homogeneity_score+"\n"
            s=s+"v_measure_score %s" % self.homogeneity_score+"\n"
            s=s+"completeness_score  %s" % self.completeness_score+"\n"

            s = s +("\n<h2>Score (silhouette sur 10 + rand,homogeneité, v_mesure et completness sur 2,5) <strong>%s / 20</strong></h2>" % self.score)
        return s


    def clusters_from_labels(self, labels,name="cl_"):
        n_clusters_ = max(labels) + 1
        for i in range(n_clusters_):
            self.clusters.append(cluster(name + str(i), [], i,i))

        i = 0
        for l in labels:
            if l >= 0:
                self.clusters[l].add_index(i, self.data, self.name_col)
            i = i + 1



    def ideal_matrix(self):
        print("Fabrication de la matrice ideal")
        clusters=np.asarray(np.zeros(len(self.data)),np.int8)
        next_cluster=0
        for k in range(len(self.data)):
            print(len(self.data)-k)
            item=self.data[self.name_col][k]
            find=False
            for i in range(k):
                if item==self.data[self.name_col][i]:
                    clusters[k]=clusters[i]
                    find=True
                    break
            if not find:
                next_cluster=next_cluster+1
                clusters[k]=next_cluster

        return clusters

    def setname(self, name):
        self.name=name
        self.type=name.split(" ")[0]
        print(name)


#definie un cluster
class cluster:
    def __init__(self, name="",index=[],color="red",pos=0):
        self.index = index
        self.name = name
        self.color=draw.get_color(color)
        self.labels=[]
        self.position=pos
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

    def __eq__(self, other):
        if set(other.index).issubset(self.index) and set(self.index).issubset(other.index):
            return True
        else:
            return False


def create_clusters_from_spectralclustering(model:model,n_clusters:np.int,n_neighbors=10,method="precomputed"):
    model.setname("SPECTRAL avec n_cluster="+str(n_clusters)+" et n_neighbors="+str(n_neighbors))

    model.start_treatment()
    if method=="precomputed":
        comp = sk.SpectralClustering(n_clusters=n_clusters,affinity=method,n_neighbors=n_neighbors).fit(model.distances)
    else:
        comp = sk.SpectralClustering(n_clusters=n_clusters, affinity=method, n_neighbors=n_neighbors).fit(model.mesures())
    model.end_treatment()

    model.clusters_from_labels(comp.labels_,"spectralclustering")

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
    mod.setname("DBSCAN avec eps="+str(eps)+" et min_elements="+str(min_elements))

    mod.start_treatment()
    if metric=="precomputed":
        model:sk.DBSCAN=sk.DBSCAN(eps=eps,min_samples=min_elements,n_jobs=4,metric=metric).fit(mod.distances)
    else:
        model: sk.DBSCAN = sk.DBSCAN(eps=eps, min_samples=min_elements, n_jobs=4, metric=metric).fit(mod.mesures())

    mod.end_treatment()

    mod.clusters_from_labels(model.labels_,"dbscan_")
    return mod


def create_clusters_from_optics(mod:model,rejection_ratio=0.5,maxima_ratio =0.5,min_elements=5,iter=100,metric="euclidean",max_bound=np.inf):
    mod.setname("OPTICS rejection_ratio="+str(round(rejection_ratio*1000)/1000)+" maxima_ratio="+str(maxima_ratio )+" min_elements="+str(min_elements))

    try:
        X=mod.mesures()
        mod.start_treatment()
        model: sk.OPTICS= sk.OPTICS(max_bound=max_bound,
                                    maxima_ratio =maxima_ratio ,
                                    rejection_ratio=rejection_ratio ,
                                    min_samples=min_elements,
                                    n_jobs=-1,
                                    metric=metric)\

        model.fit(X)

        #Production du spectre
        #draw.trace_spectre(model,X)

        mod.clusters_from_labels(model.labels_)

    except:
        print("Exception: on continue")
        return None

    mod.end_treatment()

    return mod



def create_clusters_from_agglomerative(mod, n_cluster=10,affinity="euclidean"):
    mod.setname("HAC " + str(n_cluster)+" clusters")

    mod.start_treatment()
    if affinity=="precomputed":
        model = sk.AgglomerativeClustering(n_cluster, affinity,linkage="complete").fit(mod.distances)
    else:
        model= sk.AgglomerativeClustering(n_cluster,affinity).fit(mod.mesures())

    mod.end_treatment()

    mod.clusters_from_labels(model.labels_,"Hierarchique")
    return mod


def create_model_from_meanshift(mod:model,quantile=0.2,min_bin_freq=1,method="euclidean"):
    mod.setname("meanshift q="+str(quantile)+" & min_freq="+str(min_bin_freq))

    mod.start_treatment()
    bandwidth = estimate_bandwidth(mod.mesures(), quantile=quantile)
    ms = MeanShift(bandwidth=bandwidth,min_bin_freq=min_bin_freq).fit(mod.mesures())
    mod.end_treatment()
    mod.clusters_from_labels(ms.labels_,"Meanshift")

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
