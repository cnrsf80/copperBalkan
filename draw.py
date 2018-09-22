import sklearn.decomposition as decomp
import ezvis3d as v3d
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

def trace_artefact_2d(data, clusters, name,col_name=""):
    pca = decomp.pca.PCA(n_components=2)
    pca.fit(data)
    newdata = pca.transform(data)
    x=[]
    y=[]


    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    for c in clusters:
        cv=list(colors.values())
        plt.scatter(x=newdata[c.index,0],y=newdata[c.index,1],c=cv[c.color],marker=c.marker,label=c.name,alpha=0.3)

    plt.legend(title_fontsize="xx-small", bbox_to_anchor=(0.95, 1), loc=2, borderaxespad=0.)
    plt.savefig("./saved/"+name+".png",dpi=400)
    plt.clf()

    return name+".png"



def trace_artefact_3d(data, clusters, name,label_col="",footer=""):
    pca = decomp.pca.PCA(n_components=3)
    pca.fit(data)
    newdata = pca.transform(data)

    li_data = []
    for c in clusters:
        for k in range(len(c.index)):
            if label_col=="":
                label=""
                filter=""
            else:
                label=c.name+"<br>"+c.labels[k]
                filter=label[0]

            li_data.append({
                'x': newdata[c.index[k], 0],
                'y': newdata[c.index[k], 1],
                'z': newdata[c.index[k], 2],
                'style': c.color,
                'label':label,
                'cluster':c.name
            })

    df_data = pd.DataFrame(li_data)

    g = v3d.Vis3d()
    g.width = '1200px'
    g.height = '800px'
    g.style = 'dot-color'
    g.tooltip="""function (point) { return '<b>' + point.data.label + '</b>'; }"""
    # g.onclick="""function (point) {
    #                             this.dataPoints.forEach((p)=>{
    #                                 debugger;
    #                                 if(p.point.data.cluster!=point.cluster);
    #                                     this.dataPoints.remove(p);
    #                             });
    #             }"""
    g.showPerspective = True
    g.showXAxis=False
    g.showYAxis=False
    g.showZAxis = False
    g.showGrid = True
    g.keepAspectRatio = True
    g.verticalRatio = 1.0

    g.cameraPosition = {
        'horizontal': -0.54,
        'vertical': 0.5,
        'distance': 2
    }

    g.plot(df_data)
    g.html(df_data, save=True, save_name=name, dated=False)

    footer=footer.replace("\n","<br>")

    file=open("./saved/"+name+".html","a")
    file.write("<p>" + footer + "</p>")
    file.close()

    return name+".html"


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




