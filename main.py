import vispy
import os.path
import metrics
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import folium
import algo

#Definitions
from tools import create_html


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


#data = pd.read_excel("cnx013_supp_table_s1.xlsx").head(200)
data = pd.read_excel("donnéesCIPIA.xlsx")
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

print("Arbre")
for n_cluster in range(2,40):
    mod=algo.model(data,"Ref",range(0,14))
    mod= algo.create_clusters_from_agglomerative(mod, n_cluster)
    mod.init_metrics()
    modeles.append(mod)

print("dbscan")
for min_elements in range(5):
    print(min_elements)
    for i in np.arange(3,5,0.25):
        mod=algo.model(data,"Ref",range(0,14))
        mod= algo.create_clusters_from_dbscan(mod, i, min_elements,1)
        mod.init_metrics()
        modeles.append(mod)


print("menshift")
for min_bin_freq in range(1,1):
    print(min_bin_freq)
    for i in np.arange(0.1,1,0.1):
        mod=algo.model(data,"Ref",range(0,14))
        mod= algo.create_model_from_meanshift(mod, i,min_bin_freq)
        mod.init_metrics()
        modeles.append(mod)

print("spectral")
for n_neighbors in range(5,15):
    print(n_neighbors)
    for i in range(6,20):
        mod = algo.model(data, "Ref", range(0, 14))
        mod=algo.create_clusters_from_spectralclustering(mod,i,n_neighbors,"nearest_neighbors")
        mod.init_metrics()
        modeles.append(mod)

print(str(round(len(modeles)))+" models calculés")
modeles.sort(key=lambda x:x.score,reverse=True)

code=""
for i in range(0,len(modeles)):
    print("Trace du modele "+str(i))
    code=code+"\nPosition "+str(i+1)+"<br>"
    if i<50:
        code=code+modeles[i].trace("best"+str(i),"Ref","http://shifumix.com/test")
    else:
        code=code+modeles[i].print_perfs()

print(create_html("index",code,"http://shifumix.com/test"))


#mod2.trace("dbscan_v3")

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

