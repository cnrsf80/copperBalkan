import copy

import folium
import scipy
import datetime
import numpy as np
import pandas as pd
import sklearn as sk

import algo

#Definitions
from tools import create_html
import tools
import os

v_seuil=np.vectorize(lambda x,y:1 if x<y else 0)


def create_matrix(data,seuil,start_col,end_col):
    if os.path.isfile("adjacence_artefact.csv") and len(data)>149:
        matrice=np.asmatrix(np.loadtxt("adjacence_artefact.csv",delimiter=" "))
    else:
        composition = data.loc[:, start_col:end_col]
        matrice = np.asmatrix(np.zeros((len(data),len(data))))

        np.savetxt("adjacence_artefact.csv",matrice)

    if(seuil==0):
        matrice_ww=matrice
    else:
        matrice_ww=v_seuil(matrice,seuil)

    return matrice_ww




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



def draw_site_onmap(mymap:folium.Map, G, sites_clusters, sites:pd.DataFrame ,file):
    for site in sites.iterrows():
        pos=[site[1]["Geo Latitude"],site[1]["Geo Longitude"]]
        folium.Marker(pos,site["Site"]).add_to(mymap)


def getOccurenceCluster(models,filter=""):
    occurence = []
    list_clusters=[]
    list_model=[]
    list_algo=[]
    for m in models:
        if(len(filter)==0 or m.type==filter):
            for c in m.clusters:
                if list_clusters.__contains__(c):
                    k = list_clusters.index(c)
                    occurence[k] = occurence[k] + 1
                    list_model[k].append(m.name)
                    if not list_algo[k].__contains__(m.type):list_algo[k].append(m.type)
                else:
                    print("Ajout de "+c.name)
                    list_clusters.append(c)
                    occurence.append(1)
                    list_algo.append([m.type])
                    list_model.append([m.name])

    rc=pd.DataFrame(columns=["Occurence","Cluster","Model"])
    rc["Occurence"]=occurence
    rc["Cluster"] = list_clusters
    rc["Model"]=list_model
    rc["Algos"]=list_algo

    rc=rc.sort_values("Occurence")

    return rc


# Création des occurences
def create_occurence_file(modeles, data, col_name, name,size,filter=""):
    code = ""
    rc = getOccurenceCluster(modeles[0:size],filter)
    for r in range(len(rc)):
        code = code + "\n<h1>Cluster présent dans " + str(round(100 * rc["Occurence"][r] / size)) + "% des algos</h1>"
        c = rc["Cluster"][r]
        code = code + c.print(data, col_name) + "\n"
        code = code + "\n présent dans " + ",".join(rc["Model"][r]) + "\n"

    print(create_html("occurences", code, "http://f80.fr/cnrs"))

    dfOccurences = pd.DataFrame(
        data={"Cluster": rc["Cluster"], "Model": rc["Model"], "Algos": rc["Algos"], "Occurence": rc["Occurence"]})
    l_items = list(set(data[col_name].get_values()))

    for item in l_items:
        print(item)
        dfOccurences[item] = [0] * len(rc)
        for i in range(len(rc)):
            c = dfOccurences["Cluster"][i]
            dfOccurences[item][i] = c.labels.count(item)

    writer = pd.ExcelWriter("./saved/" + name + ".xlsx")
    dfOccurences.to_excel(writer, "Sheet1")
    writer.save()
    return(name)


def create_trace(modeles,col_name, url="http://f80.fr/cnrs",name="best_"):
    name=name.replace(" ","_")
    code = "Calcul du " + str(datetime.datetime.now()) + "\n\n"
    for i in range(0, len(modeles)):
        print("Trace du modele " + str(i))
        code = code + "\nPosition " + str(i + 1) + "<br>"
        code = code + modeles[i].trace("./saved",name + str(i), col_name, url)
        code = code + modeles[i].print_perfs()

    print(create_html("index_"+name, code, url))


def create_synthes_file(modeles,labels_true=None,filename="synthese.xlsx"):
    rc:pd.DataFrame=pd.DataFrame()
    for m in modeles:
        rc=rc.append(m.toDataframe(labels_true))

    writer=pd.ExcelWriter("./saved/" + filename)
    rc.to_excel(writer)
    writer.save()

    return rc


#data = pd.read_excel("cnx013_supp_table_s1.xlsx").head(200)

#data = pd.read_excel("ACP_article.xlsx")
#col_name="Nom article"
#n_mesures=19

#tools.clear_dir()
data = pd.read_excel("Pour clustering.xlsx")
col_name="id"
n_mesures=11

data["Ref"]=data.index
data.index=range(len(data))

# models=create_gaussian_mixture_model(mes,params=np.arange(1,30))
# X=models[25].predict(data)
# exit(0)

#M0=create_matrix(data,0,"As (ppm)","Se (ppm)")
#M0:np.matrix=create_matrix(data,0,"Dim.1",lastColumn)
#plt.matshow(M0, cmap=plt.cm.Blues)
#plt.show()

#G0=nx.from_numpy_matrix(M0)
modeles=[]

def distance(i,j,name_i,name_j):
    rc=scipy.spatial.distance.cityblock(i,j)
    return rc

mod=algo.model(data,col_name,range(1,n_mesures))
mod.init_distances(distance)

true_labels=mod.ideal_matrix()
reference=algo.model(data,col_name)
reference.clusters_from_labels(true_labels)
print(reference.print_cluster("\n"))


for n_cluster in range(10,25):
    modeles.append(copy.deepcopy(mod).execute("HAC_euc",
                                lambda x:sk.cluster.AgglomerativeClustering(x["n_cluster"],x["method"]),
                                {"n_cluster":n_cluster,"method":"eucledean"}))

    modeles.append(copy.deepcopy(mod).execute("HAC_pre",
                                lambda x:sk.cluster.AgglomerativeClustering(x["n_cluster"],x["method"]),
                                {"n_cluster":n_cluster,"method":"precomputed"}))



    modeles.append(copy.deepcopy(mod).execute("HAC_pre",
                                lambda x:sk.cluster.AgglomerativeClustering(x["n_cluster"],x["method"]),
                                {"n_cluster":n_cluster,"method":"precomputed"}))


for min_elements in range(2, 6):
    for eps in np.arange(0.1, 0.9, 0.1):
        modeles.append(copy.deepcopy(mod).execute(
            "DBSCAN",
            lambda x: sk.DBSCAN(eps=x["eps"], min_samples=x["min_elements"], n_jobs=4),
            {"n_cluster": n_cluster, "method": "precomputed"})
        )

for min_bin_freq in range(1,4):
        for i in np.arange(0.1,0.9,0.1):
            modeles.append(copy.deepcopy(mod).execute(
                "MEANSHIFT",
                lambda x: sk.cluster.MeanShift(bandwidth=x["bandwidth"],min_bin_freq=x["min_bin_freq"]),
                {"bandwidth": i, "min_bin_freq": min_bin_freq})
            )


for n_neighbors in range(5,10):
    for n_cluster in range(6,20):
        modeles.append(copy.deepcopy(mod).execute(
            "SPECTRALCLUSTERING",
            lambda x: sk.cluster.SpectralClustering(n_cluster=x["n_cluster"],n_neighbors=x["n_neighbors"]),
            {"n_cluster": n_cluster, "n_neighbors": n_neighbors})
        )



if False:
    for method in ["euclidean"]:
        for min_elements in range(10,2,-1):
            for maxima_ratio in np.arange(0.3,0.9,0.1):
                for rejection_ratio in np.arange(0.3,0.8,0.1):
                    mod2= algo.create_clusters_from_optics(copy.deepcopy(mod),rejection_ratio ,maxima_ratio , min_elements,1,method)
                    if not mod2 is None:
                        print(mod2.init_metrics(true_labels))
                        modeles.append(mod2)


print("Neural gas")
if True:
    for passes in range(10,90,20):
        for distance_toremove_edge in range(6,38,4):
            mod2= algo.create_cluster_from_neuralgasnetwork(
                copy.deepcopy(mod),
                passes=passes,
                distance_toremove_edge=distance_toremove_edge)
            print(mod2.init_metrics(true_labels))
            print(mod2.print_cluster())
            modeles.append(mod2)




print(str(round(len(modeles)))+" models calculés")
modeles.sort(key=lambda x:x.score,reverse=True)
size = round(len(modeles))

url_base="http://f80.fr/cnrs"
name=str(datetime.datetime.now()).split(".")[0].replace(":","").replace("2018-","")
create_trace(modeles,col_name,url_base,"best"+name)
create_synthes_file(modeles,true_labels)
print("Matrice d'occurence : "+url_base+"/"+create_occurence_file(modeles,data,col_name,"occurencesOPTICS",size,filter="OPTICS")+".html")
print("Matrice d'occurence : "+url_base+"/"+create_occurence_file(modeles,data,col_name,"occurencesNEURALGAS",size,filter="NEURALGAS")+".html")


exit(0)

#artefact_clusters=create_ncluster(G0,8);
#artefact_clusters=create_clusters_from_girvannewman(G0);

#for c in artefact_clusters:c.print(data,"Sample label")

#trace_artefact(G0,artefact_clusters,data,"Ref")



# M=create_site_matrix(data,artefact_clusters)
# G=nx.from_numpy_matrix(M)
# sites_clusters=create_clusters_from_asyncfluid(G,4)

#Affichage
# for c in sites_clusters:c.print(data,"Site")
# trace_artefact(G,sites_clusters,data,"Site")


#https://python-visualization.github.io/folium/docs-v0.6.0/
#mymap=folium.Map(location=[48,2],zoom_start=13)
#draw_site_onmap(mymap,G,sites_clusters,create_sites_df(data),"map.html")

