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
    
    print(type(df_nodos))

    mapping = {}

    # Add nodes
    for i in range(len(df_nodos)):
        #print(df_nodos.iloc[i,0], df_nodos.iloc[i,1])
        nodo_id = G.number_of_nodes() 
        mapping[df_nodos['Nodo'][i]] = nodo_id
        G.add_node(nodo_id, features={"crime_type": df_nodos.iloc[i,1] })
        labels.append(df_nodos.iloc[i,1])
    
    # Add edges
    for i in range(len(df_aristas)):
        nodo1_id = mapping[df_aristas['Nodo1'][i]]
        nodo2_id = mapping[df_aristas['Nodo2'][i]]
        G.add_edge(nodo1_id, nodo2_id)
        edges.append((nodo1_id, nodo2_id))

    return True
    
if __name__ == '__main__':
    # Crear el grafo de Networkx con nodos geolocalizados y 8 características
    G = nx.Graph()
    edges = []
    file1 = 'nodos_subgrafo.csv'
    file2 = 'aristas_subgrafo.csv'
    hidden_dim = 0
    labels = []

    def add_nodes_edges(G, edges):
        sig = 0.2 # Desviación estándar 
        # Add nodos con sus características
        G.add_node(0, features={"crime_type": 1+gaus(0, sig), "x2": 1+gaus(0, sig), "x3":1+gaus(0, sig), "x4":1+gaus(0, sig) })
        G.add_node(1, features={"crime_type": 1+gaus(0, sig), "x2": 1+gaus(0, sig), "x3":1+gaus(0, sig), "x4":1+gaus(0, sig) })
        G.add_node(2, features={"crime_type": 1+gaus(0, sig), "x2": 1+gaus(0, sig), "x3":1+gaus(0, sig), "x4":1+gaus(0, sig) })
        G.add_node(3, features={"crime_type": 1+gaus(0, sig), "x2": 1+gaus(0, sig), "x3":1+gaus(0, sig), "x4":1+gaus(0, sig) })
    
        G.add_node(4, features={"crime_type": 1+gaus(0, sig), "x2": 1+gaus(0, sig), "x3":1+gaus(0, sig), "x4":-1+gaus(0, sig)})
        G.add_node(5, features={"crime_type": 1+gaus(0, sig), "x2": 1+gaus(0, sig), "x3":1+gaus(0, sig), "x4":-1+gaus(0, sig)})
        G.add_node(6, features={"crime_type": 1+gaus(0, sig), "x2": 1+gaus(0, sig), "x3":1+gaus(0, sig), "x4":-1+gaus(0, sig)})
        G.add_node(7, features={"crime_type": 1+gaus(0, sig), "x2": 1+gaus(0, sig), "x3":1+gaus(0, sig), "x4":-1+gaus(0, sig)})
        G.add_node(8, features={"crime_type": 1+gaus(0, sig), "x2": 1+gaus(0, sig), "x3":1+gaus(0, sig), "x4":-1+gaus(0, sig)})
    
        G.add_node(9 , features={"crime_type": 1+gaus(0, sig), "x2": 1+gaus(0, sig), "x3":-1+gaus(0, sig), "x4":1+gaus(0, sig)})
        G.add_node(10, features={"crime_type": 1+gaus(0, sig), "x2": 1+gaus(0, sig), "x3":-1+gaus(0, sig), "x4":1+gaus(0, sig)})
        G.add_node(11, features={"crime_type": 1+gaus(0, sig), "x2": 1+gaus(0, sig), "x3":-1+gaus(0, sig), "x4":1+gaus(0, sig)})
    
        # Crear edges en el grafo
        edges = [(0, 1), (0, 2), (0, 3), (1,2), (2, 3), (3, 1),
             (4,5),(4,6),(4,7),(4,8),(5,8),(5,6),(5,7),(6,7),(6,8),(7,8),
             (9,10),(9,11),(10,11), 
             (3,8), (8,10), (3,10)]   # Ejemplo de edges, modificar según la estructura de tu grafo
        add_edges(G, edges)


    if (len(sys.argv)==1):
        print("Small graph")
        add_nodes_edges(G, edges)
        hidden_dim = 2 # 16
        labels = [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2 ]
        n_components = 3
        numeric_features = ['crime_type', 'x2', 'x3', 'x4']
        # criar histograma por quantidade de crime
        plt.subplot(232)
        plt.hist(labels, bins=max(labels)+1)
        plt.title("Histogram of crimes")

    elif (sys.argv[1] =="-h"):
        print("Run with a small graph: \n python theGAE.py")
        print("Run with a graph using csv: \n python theGAE.py -csv")

    elif(sys.argv[1] =="-csv"):
        print("Graph from csv")
        argument = str(sys.argv[1])
        read_graph_from_csv(file1, file2, G, edges, labels)
        hidden_dim = 42
        n_components = contar_elementos_no_repetidos(labels)
        numeric_features = ['crime_type']
        # criar histograma por quantidade de crime
        plt.subplot(232)
        plt.hist(labels, bins=max(labels)+1)
        plt.title("Histogram of crimes")        
        
    #print('----------------------------')
    #print(G.nodes(data=True))
    #print(G.edges)
    #print('----------------------------')

    plt.figure(1)
    plt.subplot(231)
    nx.draw(G, with_labels = True)
    #plt.show()

    # Crear un DataFrame con las características de los nodos
    node_features = pd.DataFrame.from_dict(
        {node: data['features'] for node, data in G.nodes(data=True)}, orient='index'
    )

    # Preprocesar las características no binarias
    """non_binary_features = ['crime_type', 'income_level', 'education_level']
    label_encoders = {}
    for feature in non_binary_features:
        label_encoders[feature] = LabelEncoder()
        node_features[feature] = label_encoders[feature].fit_transform(node_features[feature])"""

    # Normalizar as caraterísticas numéricas da lista: numeric_features
    scaler = StandardScaler()
    node_features[numeric_features] = scaler.fit_transform(node_features[numeric_features])
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
            super(GraphAutoencoder, self).__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim +1)
            self.conv2 = GCNConv(hidden_dim +1, hidden_dim)
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
    print('############################################')
    
    # Entrenar el modelo
    num_epochs = 1
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(features, edge_index)
        #print(train_nodes, 'len = ', (outputs))
        loss = criterion(outputs[train_nodes], features[train_nodes])
        loss.backward()
        optimizer.step()
    print('############################################')
    # Obtener las representaciones embebidas de los nodos
    embeddings = model(features, edge_index).detach().numpy()
    
    
    # Encoding
    x = model.conv1(features, edge_index)
    x = torch.relu(x)
    embeddings = model.conv2(x, edge_index)
    embeddings = embeddings.detach().numpy()
    
    #print(embeddings.shape)
    
    ########### TSNE ############
    from sklearn.manifold import TSNE
    
    #labels = [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2 ]
    print('labels = ======================================= ',len(labels))
    print((labels))
    tsne = TSNE(n_components=3, random_state=37) #42 #init='pca', 
    
    # Perform t-SNE dimensionality reduction
    X_tsne = tsne.fit_transform(embeddings)
    #print('+++', X_tsne)
    
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
    else: plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis')
    
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
    
    kmeans = KMeans(n_clusters=3, random_state=0)
    clusters = kmeans.fit_predict(embeddings)
    
    # Imprimir los resultados del clustering
    color_map = []
    for node, cluster in zip(node_features.index, clusters):
        print(f"Nodo {node} -> Cluster {cluster}")
        if (cluster ==0):
            color_map.append('blue')
        elif (cluster ==1):
            color_map.append('red')
        else:
            color_map.append('green')
    
    plt.subplot(234)
    nx.draw(G, node_color=color_map, with_labels=True)

    plt.show()
    
    ########### kmeans ############
    
    #umap 
    #t-sne
    
    