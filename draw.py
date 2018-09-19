import matplotlib.colors as colors_lib
import sklearn.decomposition as decomp
from mpl_toolkits.mplot3d import Axes3D
import ezvis3d as v3d
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def trace_artefact_2d(data, clusters, name):
    pca = decomp.pca.PCA(n_components=2)
    pca.fit(data)
    newdata = pca.transform(data)
    x=[]
    y=[]
    colors=[]
    for c in clusters:
        x.extend(newdata[c.index,0])
        y.extend(newdata[c.index,1])
        colors.extend([c.color]*len(c.index))

    plt.scatter(x, y, c=colors)
    plt.savefig("./saved/"+name+".png")


def trace_artefact_3d(data, clusters, name):
    pca = decomp.pca.PCA(n_components=3)
    pca.fit(data)
    newdata = pca.transform(data)

    li_data = []
    for c in clusters:
        for p in c.index:
            li_data.append({
                'x': newdata[p, 0],
                'y': newdata[p, 1],
                'z': newdata[p, 2],
                'style': c.color
            })

    df_data = pd.DataFrame(li_data)

    g = v3d.Vis3d()
    g.width = '1200px'
    g.height = '800px'
    g.style = 'dot-color'
    g.showPerspective = True
    g.showGrid = True
    g.keepAspectRatio = True
    g.verticalRatio = 1.0

    g.cameraPosition = {'horizontal': -0.54,
                        'vertical': 0.5,
                        'distance': 2
                        }

    g.plot(df_data)
    g.html(df_data, save=True, save_name=name, dated=False)


def trace_artefact(G, clusters, data, label_col=""):
    pos = nx.spectral_layout(G, scale=5, dim=len(clusters))
    # pos=nx.spring_layout(G,iterations=500,dim=len(clusters))
    # nx.set_node_attributes(G,"name")

    if len(clusters) > 0:
        i = 0
        labels = {}
        for c in clusters:
            # for k in c.index:
            #     G.nodes[k].name = data[label_col][k]
            #     labels[G.nodes[k]]=G.nodes[k]

            nx.draw_networkx_nodes(G,
                                   pos=pos,
                                   nodelist=c.index,
                                   alpha=0.6, node_size=200, node_color=c.color)

            i = i + 1

        # nx.draw_networkx_labels(G, pos, labels=labels,font_size=10, font_family='sans-serif')

        # nx.draw_networkx_edges(G, pos, alpha=0.5)
    else:
        nx.draw(G)

    plt.axis('off')
    # plt.savefig('graph.pdf', format="pdf")
    plt.show()




