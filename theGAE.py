"""
Env: chatHqu
Este codigo SI incluye las aristas del grafo, CONVOLUCIONES

pip install chardet
pip install stellargraph
pip install -U scikit-learn
pip install networkx
pip install --upgrade pandas
pip install numpy
pip install torch
pip install keras==2.9

"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy.random import normal as gaus
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch_geometric.nn import GCNConv
import torch_geometric.utils as utils
import matplotlib.pyplot as plt
import sys
import networkx as nx
import pandas as pd
from collections import Counter
import ast
import random

np.random.seed(37)

def contar_elementos_no_repetidos(array):
    contador = Counter(array)
    elementos_no_repetidos = [elemento for elemento, frecuencia in contador.items() if frecuencia == 1]
    cantidad_elementos_no_repetidos = len(elementos_no_repetidos)
    return cantidad_elementos_no_repetidos

def add_edges( G, edge_tuples):
    for edge in edge_tuples:
        #print(edge[0], edge[1])
        G.add_edge(edge[0], edge[1])


def read_graph_from_csv(file1, file2, G, edges, labels):
    df_nodos = pd.read_csv(file1)
    df_aristas = pd.read_csv(file2)
    
    numeric_features = df_nodos.columns.tolist()[1:]
    #labels = []
    for i in numeric_features:
        labels.append(df_nodos[i].sum())
    print('++++ frecuencias', labels, 'enbd', type(labels))
    # For DEBUGGING
    #print('ooo',numeric_features)

    mapping = {}
    

    # Add nodes
    for i in range(len(df_nodos)):
        #print(df_nodos.iloc[i,0], df_nodos.iloc[i,1])
        nodo_id = G.number_of_nodes() 
        mapping[df_nodos['Nodo'][i]] = nodo_id
        features_values = [df_nodos[x][i] for x in numeric_features]
        #print('feat_nodes', [df_nodos[x][i] for x in numeric_features])
        G.add_node(nodo_id, features={"crime_type": features_values })
        #labels.append(features_values)
    #print('labeeeeeeeeeeeeeeeeeeeeeeeelss', labels)
    
    # Add edges
    for i in range(len(df_aristas)):
        nodo1_id = mapping[df_aristas['Nodo1'][i]]
        nodo2_id = mapping[df_aristas['Nodo2'][i]]
        G.add_edge(nodo1_id, nodo2_id)
        edges.append((nodo1_id, nodo2_id))

    return numeric_features
    
if __name__ == '__main__':
    # Crear el grafo de Networkx con nodos geolocalizados y 8 características
    G = nx.Graph()
    edges = []
    file1 = 'nodos_subgrafo' #month
    file2 = 'aristas_subgrafo'
    hidden_dim = 0
    labels = []
    numeric_features = []
    weekdays = []

    def add_nodes_edges(G, edges):
        sig = 0.2 # Desviación estándar 
        # Add nodos con sus características
        G.add_node(0, features={"crime_type": [ 1+gaus(0, sig), 1+gaus(0, sig), 1+gaus(0, sig), 1+gaus(0, sig) ] })
        G.add_node(1, features={"crime_type": [ 1+gaus(0, sig), 1+gaus(0, sig), 1+gaus(0, sig), 1+gaus(0, sig) ] })
        G.add_node(2, features={"crime_type": [ 1+gaus(0, sig), 1+gaus(0, sig), 1+gaus(0, sig), 1+gaus(0, sig) ] })
        G.add_node(3, features={"crime_type": [ 1+gaus(0, sig), 1+gaus(0, sig), 1+gaus(0, sig), 1+gaus(0, sig) ] })
        
        G.add_node(4, features={"crime_type": [ 1+gaus(0, sig), 1+gaus(0, sig), 1+gaus(0, sig), -1+gaus(0, sig) ] })
        G.add_node(5, features={"crime_type": [ 1+gaus(0, sig), 1+gaus(0, sig), 1+gaus(0, sig), -1+gaus(0, sig) ] })
        G.add_node(6, features={"crime_type": [ 1+gaus(0, sig), 1+gaus(0, sig), 1+gaus(0, sig), -1+gaus(0, sig) ] })
        G.add_node(7, features={"crime_type": [ 1+gaus(0, sig), 1+gaus(0, sig), 1+gaus(0, sig), -1+gaus(0, sig) ] })
        G.add_node(8, features={"crime_type": [ 1+gaus(0, sig), 1+gaus(0, sig), 1+gaus(0, sig), -1+gaus(0, sig) ] })
        
        G.add_node(9 , features={"crime_type": [ 1+gaus(0, sig), 1+gaus(0, sig), -1+gaus(0, sig), -1+gaus(0, sig) ] })
        G.add_node(10, features={"crime_type": [ 1+gaus(0, sig), 1+gaus(0, sig), -1+gaus(0, sig), -1+gaus(0, sig) ] })
        G.add_node(11, features={"crime_type": [ 1+gaus(0, sig), 1+gaus(0, sig), -1+gaus(0, sig), -1+gaus(0, sig) ] })
        
        #G.add_node(0, features={"crime_type": [1+gaus(0, sig), 1+gaus(0, sig), 1+gaus(0, sig), 1+gaus(0, sig) ] })
       
        # Crear edges en el grafo
        edges = [(0, 1), (0, 2), (0, 3), (1,2), (2, 3), (3, 1),
             (4,5),(4,6),(4,7),(4,8),(5,8),(5,6),(5,7),(6,7),(6,8),(7,8),
             (9,10),(9,11),(10,11), 
             (3,8), (8,10), (3,10)]
        add_edges(G, edges)

    # Menu
    if (len(sys.argv)==1):
        print("Small graph")
        add_nodes_edges(G, edges)
        hidden_dim = 4 # 16
        labels = [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2 ]  # this is my y
        n_components = 3
        numeric_features = ['crime_type', 'x2', 'x3', 'x4']
        weekdays = numeric_features
        # criar histograma por quantidade de crime
        plt.subplot(232)
        plt.hist(labels, bins=max(labels)+1)
        plt.title("Histogram of crimes")

    elif (sys.argv[1] =="-h"):
        print("Run with a small graph: \n python theGAE.py")
        print("Run with a graph using csv: \n python theGAE.py -csv")
        print("Run with a graph using csv with just one feature: \n python theGAE.py -csv -f=1")

    elif(sys.argv[1] =="-csv"):
        print("Graph from csv")
        n_components = 3 # contar_elementos_no_repetidos(labels)
        # criar histograma por quantidade de crime
        plt.subplot(232)
        
        if ( len(sys.argv) == 3 ):
        
            if (sys.argv[2] =="-f=1" or sys.argv[2] == "-1"):
                file_str = '3all.csv'
                bins_customized = 'auto' #max(max(arr) for arr in labels)
                hidden_dim = 1
                weekdays = np.array(['Frequency'])

            elif(sys.argv[2] =="-f=month" or sys.argv[2] == "-m"):
                file_str = '3month.csv'
                hidden_dim = 12
                bins_customized = 12
                weekdays = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct','Nov','Dec'])

            elif(sys.argv[2] =="-f=period" or sys.argv[2] == "-p" ):
                file_str = '3period.csv'
                hidden_dim = 5
                bins_customized = 5
                weekdays = np.array(['Unknow', 'morning', 'afternoon', 'night', 'early_morning'])

            elif(sys.argv[2] == "-f=week" or sys.argv[2] == "-w"):
                file_str = '3week.csv'
                hidden_dim = 7
                bins_customized=7
                weekdays = np.array(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        
        file1 += file_str
        file2 += file_str

        
        numeric_features = read_graph_from_csv(file1, file2, G, edges, labels)
        plt.bar(weekdays, labels, color ='maroon',width = 0.4)
        #plt.hist(labels, bins=bins_customized)
        plt.title("Histogram of crimes")

    plt.figure(1)
    plt.subplot(231)
    random.seed(37)
    nx.draw(G, with_labels = True)
    
    data = {}
    #workeaba
    #data['crime_type']=[x[1]['features']['crime_type'] for x in G.nodes(data=True)]
    """
    data['x2']        =[x[1]['features']['crime_type'][1] for x in G.nodes(data=True)]
    data['x3']        =[x[1]['features']['crime_type'][2] for x in G.nodes(data=True)]
    """
    #[ print('mmm',x[1]['features']['crime_type']) for x in G.nodes(data=True)]

    data = {}
    k = 0
    for i in numeric_features:
        data[i] = [x[1]['features']['crime_type'][k] for x in G.nodes(data=True)]
        k = k + 1  

    node_features = pd.DataFrame.from_dict(data)
    print(' ================ node_features', node_features)
    #####################################################################
    """datos_graph = []
    for node, data in G.nodes(data=True):
        print(node, type(data['features']['crime_type']))  # old: 0 {'crime_type': 0.8874868743225615, 'x2': 0.6662334548036992, 'x3': 0.7445410198753913, 'x4': 0.9550832071294588}
        datos_graph.append(data['features']['crime_type'])
        break
    print('datos_graph',datos_graph)
    node_features = pd.DataFrame(datos_graph, columns=numeric_features) #, orient='index'"""

    ###########################

    # Crear un DataFrame con las características de los nodos
    """node_features = pd.DataFrame.from_dict(
        {node: data['features'] for node, data in G.nodes(data=True)}, orient='index'
    )"""
    # For DEBBUGING
    #print('+++++++++++++++++++++++++++++ typ +++++++++++++++++ \n',(node_features))
     
    """
         crime_type        x2        x3        x4
    0     1.340978  1.272639  0.684242  1.036812
    1     0.831190  1.055397  1.030833  0.842458
    2     1.302808  0.891135  0.617001  1.074186
    """

    # Preprocesar las características no binarias
    """non_binary_features = ['crime_type', 'income_level', 'education_level']
    label_encoders = {}
    for feature in non_binary_features:
        label_encoders[feature] = LabelEncoder()
        node_features[feature] = label_encoders[feature].fit_transform(node_features[feature])"""

    # Normalizar as caraterísticas numéricas da lista: numeric_features
    scaler = StandardScaler()
    node_features[numeric_features] = scaler.fit_transform(node_features[numeric_features])
    #node_features = pd.DataFrame(node_features, columns=numeric_features[numeric_features] )
    # For DEBBUGING
    #print('+++++++++++++++++++++++++++++ w2 +++++++++++++++++ \n',(node_features[numeric_features] ))
    
    """numeric_features = ['population_density', 'age_median', 'employment_rate'] # 'latitude', 'longitude', 
    scaler = StandardScaler()
    node_features[numeric_features] = scaler.fit_transform(node_features[numeric_features])
    """
    # Dividir los nodos en entrenamiento y prueba
    train_nodes, test_nodes = train_test_split(node_features.index, train_size=0.8, test_size=0.2)
    
    # Convertir las características en tensores de PyTorch
    features = torch.FloatTensor(node_features.values)
    
    
    # Crear los índices de los edges
    edge_index = torch.LongTensor(list(edges)).t().contiguous()
    
    # Obtener el número de nodos
    num_nodes = len(node_features)
    
    # Asegurarse de que los índices de los bordes sean únicos y en el rango correcto
    edge_index, _ = utils.add_self_loops(edge_index, num_nodes=num_nodes)
    
    # Definir el modelo del grafo autoencoder
    class GraphAutoencoder(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            torch.random.manual_seed(37)
            super(GraphAutoencoder, self).__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, input_dim),
                nn.ReLU()
            )
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = torch.relu(x)
            x = self.conv2(x, edge_index)
            x = self.decoder(x)
            return x
    
    # Definir los hiperparámetros del modelo
    input_dim = features.size(1)
    #hidden_dim = 40 # Already defined above 
    
    # Crear una instancia del modelo
    model = GraphAutoencoder(input_dim, hidden_dim)
    
    # Definir la función de pérdida y el optimizador
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    
    # Entrenar el modelo
    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(features, edge_index)
        #print(train_nodes, 'len = ', (outputs))
        loss = criterion(outputs[train_nodes], features[train_nodes])
        loss.backward()
        optimizer.step()
    
    # Obtener las representaciones embebidas de los nodos
    embeddings = model(features, edge_index).detach().numpy()
    
    
    # Encoding
    x = model.conv1(features, edge_index)
    x = torch.relu(x)
    embeddings = model.conv2(x, edge_index)
    embeddings = embeddings.detach().numpy()
    
    ########### TSNE ############
    from sklearn.manifold import TSNE
    
    #labels = [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2 ]
    #print('labels = ======================================= ',len(labels))
    #print((labels))
    tsne = TSNE(n_components=2, random_state=37) #42 #init='pca', 
    
    # Perform t-SNE dimensionality reduction
    X_tsne = tsne.fit_transform(embeddings)
    #print('+++ \n', X_tsne)
    
    var_3d = False
    mid = int(len(embeddings)/2)
    #print('mid ======================================= ', X_tsne[:,0][0:mid])
    # Display the plot
    if (var_3d): plt.subplot(233,  projection='3d')
    else: plt.subplot(233)
    
    # Create a scatter plot with different colors for each cluster
    if (var_3d): 
        plt.scatter(X_tsne[:,0][0:mid], X_tsne[:,1][0:mid], X_tsne[:,2][0:mid], c= 'r', marker='+')
        plt.scatter(X_tsne[:,0][mid:], X_tsne[:,1][mid:], X_tsne[:,2][mid:], c= 'b', marker='.')
    else: 
        k = 0
        for i in X_tsne:
            plt.text(i[0], i[1],k)
            k +=1
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], cmap='viridis')  # c=labels
    
    # Add a colorbar legend
    plt.colorbar()
    
    # Add labels and title
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    #if (var_3d): plt.set_zlabel('t-SNE Dimension 3')
    plt.title('t-SNE Visualization with Clusters')
    
    
    ########### TSNE ############
    
    

    ########### kmeans ############
    
    # Aplicar algoritmo de clustering en las representaciones embebidas
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=4, random_state=0)
    clusters = kmeans.fit_predict(embeddings)
    
    # Imprimir los resultados del clustering
    color_map = []
    for node, cluster in zip(node_features.index, clusters):
        # For DEBUGING
        #print(f"Nodo {node} -> Cluster {cluster}")
        if (cluster ==0):
            color_map.append('blue')
        elif (cluster ==1):
            color_map.append('red')
        elif (cluster ==2):
            color_map.append('green')
        elif (cluster ==3):
            color_map.append('yellow')
    
    plt.subplot(234)
    nx.draw(G, node_color=color_map, with_labels=True)

    plt.show()
    
    ########### kmeans ############

    ########### TIME-SERIES ############

    import datetime
    node_features2 = pd.DataFrame.from_dict(data)

    def get_fetures_of_Node_from_Graph(number_node, verbose = False):
        if verbose: print(node_features2.iloc[number_node].values)
        return node_features2.iloc[number_node].values

    num_nodes = input('Enter the AMOUNT of nodes: ')
    nodes = []
    for i in range(int(num_nodes)):
        variable = input('# node ')
        nodes.append(int(variable))
    print('ok')
    plt.subplot(121)

    # Defined Before  
    # weekdays = np.array([...])
    print(len(weekdays))

    for i in range(int(num_nodes)):
        frequency = get_fetures_of_Node_from_Graph(nodes[i], True)
        dataframe = pd.DataFrame({'date_of_week': weekdays,
                              'frequency': frequency})
        plt.plot(weekdays, dataframe.frequency, marker='o', label="node "+str(nodes[i]))

    # Giving title to the chart using plt.title and a legend
    plt.title('Crimes by Day')
    plt.legend(loc="upper left")
 
    # rotating the x-axis tick labels at 30degree  towards right
    plt.xticks(rotation=30, ha='right')
 
    # Providing x and y label to the chart
    plt.xlabel('Day')
    plt.ylabel('Amount of Crimes')
    
    ########### TIME-SERIES ############


    plt.subplot(122)
    color_map = ['blue'] * len(node_features.index)
    for node in nodes:
        color_map[node] = 'red'

    nx.draw(G, node_color=color_map, with_labels=True)

    plt.show()
    #umap 
    #t-sne
    
    