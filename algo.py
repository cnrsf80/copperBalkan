from sklearn.cluster import estimate_bandwidth, MeanShift
from gng import GrowingNeuralGas
import os
import draw
import time
import sklearn.metrics as metrics
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms import community
import sklearn.cluster as sk

from tools import tirage

colors=[]
for i in range(200):colors.append(i)

#Représente un model
#un model est une liste de cluster après application d'un algorithme de clustering
class model:
    name=""
    delay:int=0 #delay en secondes
    silhouette_score:int=0
    score:int=0
    url=""


    def __init__(self, data,name_col=0,mesures_col=range(1,5)):
        self.name_col=name_col
        self.mesures_col=mesures_col
        self.clusters=[]
        self.data=data

    #Calcul de la matrice de distance
    #func_distance est la fonction de calcul de la distance entre 2 mesures
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


    #Retourne la liste des composants par cluster
    def print_cluster(self,end_line=" - "):
        s=""
        for c in self.clusters:
            s=s+c.print(self.data, self.name_col)+end_line
        return s

    #Mesure le temps de traitement de l'algorithme
    def start_treatment(self):
        self.delay=time.time()

    #Mesure le temps de traitement
    def end_treatment(self):
        self.delay=round((time.time()-self.delay)*10)/10

    #Produit une réprésentation 3D et une représentation 2D des mesures
    #après PCA et coloration en fonction du cluster d'appartenance
    def trace(self,path,filename,label_col_name="",url_base=""):
        title=self.print_perfs()+"\n"+self.print_cluster("<br>")
        self.url=url_base+"/"+draw.trace_artefact_3d(
            self.mesures(), self.clusters, path,filename,label_col_name,title
        )
        s="<a href='"+self.url+"'>représentation 3D</a>\n"
        s=s+("<a href='"+url_base + "/" + draw.trace_artefact_2d(self.mesures(), self.clusters, path,filename,label_col_name))+"'>représentation 2D</a>\n"
        return self.print_perfs()+"\n"+s+"\n"

    #Convertis les clusters en un vecteur simple
    #la position désigne la mesure
    #le contenu désigne le numéro du cluster
    #format utilisé notamment pour les métriques
    def cluster_toarray(self):
        rc=np.zeros((len(self.data),),np.int)
        for k in range(len(self.clusters)):
            for i in self.clusters[k].index:
                rc[i]=k
        return rc

    #Enregistre le clustering dans un fichier au format binaire
    def save_cluster(self):
        if len(self.name)==0:return False
        res:np.ndarray=self.cluster_toarray()
        res.tofile("./clustering/"+self.name+".array")
        return True

    #Charge le clustering depuis un fichier si celui-ci existe
    def load_cluster(self):
        try:
            res=np.fromfile("./clustering/"+self.name+".array",np.int,-1)
            self.clusters_from_labels(res)
            return True
        except:
            return False


    def mesures(self):
        return self.data.iloc[:,self.mesures_col]

    def toDataframe(self,labels_true=None):
        if self.score==0:
            self.init_metrics(labels_true=labels_true)

        obj={
            "Name":self.name,
            "Algo":self.type,
            "nClusters":len(self.clusters),
            "delay (secondes)":self.delay,
            "URL":self.url,
            "Clusters":self.print_cluster(),
            "Score":[self.score],
            "Rand_index":[self.rand_index],
            "Silhouette":[self.silhouette_score],
            "V-mesure":[self.v_measure_score]
        }
        rc=pd.DataFrame(data=obj)

        return rc


    def init_metrics(self,labels_true):
        if len(self.clusters)>2:
            labels=self.cluster_toarray()
            self.silhouette_score= metrics.silhouette_score(self.mesures(), labels)
            self.rand_index=metrics.adjusted_rand_score(labels_true, labels)

            #self.self.adjusted_mutual_info_score=metrics.self.adjusted_mutual_info_score(labels_true,labels)

            self.homogeneity_score=metrics.homogeneity_score(labels_true,labels)
            self.completeness_score=metrics.completeness_score(labels_true,labels)
            self.v_measure_score=metrics.v_measure_score(labels_true,labels)

            self.score=(self.silhouette_score
                        +(self.rand_index+1)
                        +self.v_measure_score
                        )/3
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
        n_clusters_ = round(max(labels) + 1)
        for i in range(n_clusters_):
            self.clusters.append(cluster(name + str(i), [], i,i))

        i = 0
        for l in labels:
            if l >= 0:
                self.clusters[l].add_index(i, self.data, self.name_col)
            i = i + 1

        self.save_cluster()

    def execute(self,algo_name,algo,params:dict):
        name=algo_name+" "
        for key in params.keys():
            name=name+key+"="+str(params.get(key))
        self.setname(name)
        self.start_treatment()
        comp = algo(params).fit(self.mesures())
        self.end_treatment()
        self.clusters_from_labels(comp.labels_, algo_name)
        return self



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

    def clusters_from_real(self, data,name):
        if len(self.clusters)==0:
            pts=self.mesures().values
            labels=[0]*len(pts)
            for p in data:
                for i in range(len(pts)):
                    a=p[0]
                    b=pts[i]
                    if np.array_equal(a,b):
                        labels[i]=int(p[1])

            self.clusters_from_labels(labels,name)


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
        s=("Cluster:"+self.name+"=")
        s=s+(" / ".join(data[label_col][self.index]))
        return s

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

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
def create_cluster_from_neuralgasnetwork(model:model,a=0.5,passes=80,distance_toremove_edge=8):
    data=model.mesures().values
    model.setname("NEURALGAS avec distance_toremove="+str(distance_toremove_edge)+" passes="+str(passes))

    if not model.load_cluster():
        model.start_treatment()
        gng = GrowingNeuralGas(data)
        gng.fit_network(e_b=0.05, e_n=0.006,
                        distance_toremove_edge=distance_toremove_edge,
                        l=100, a=0.5, d=0.995,
                        passes=passes, plot_evolution=False)
        model.end_treatment()
        print('Found %d clusters.' % gng.number_of_clusters())
        model.clusters_from_real(gng.cluster_data(), "NEURALGAS_")

    #gng.plot_clusters(gng.cluster_data())
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




def create_clusters_from_optics(mod:model,rejection_ratio=0.5,maxima_ratio =0.5,min_elements=5,iter=100,metric="euclidean",max_bound=np.inf):
    mod.setname("OPTICS rejection_ratio="
                +str(round(rejection_ratio*1000)/1000)+" maxima_ratio="
                +str(maxima_ratio )+" min_elements="
                +str(min_elements))

    X=mod.mesures()
    mod.start_treatment()
    model: sk.OPTICS= sk.OPTICS(max_bound=max_bound,maxima_ratio =maxima_ratio ,
                                rejection_ratio=rejection_ratio ,min_samples=min_elements,
                                n_jobs=-1,metric=metric)


    model.fit(X)
    mod.clusters_from_labels(model.labels_,"cl_optics")

    mod.end_treatment()

    return mod



def create_clusters_from_agglomerative(mod:model, n_cluster=10,affinity="euclidean"):
    mod.setname("HAC " + str(n_cluster)+" clusters")

    if not mod.load_cluster():
        mod.start_treatment()
        if affinity=="precomputed":
            model = sk.AgglomerativeClustering(n_cluster, affinity,linkage="complete").fit(mod.distances)
        else:
            model= sk.AgglomerativeClustering(n_cluster,affinity).fit(mod.mesures())

        mod.end_treatment()
        mod.clusters_from_labels(model.labels_,"Hierarchique")

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
