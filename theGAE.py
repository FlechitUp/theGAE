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
import matplotlib as mp
import matplotlib.pyplot as plt
import sys
import networkx as nx
import pandas as pd
from collections import Counter
import ast
import random
import folium

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
    global lat_long_nodes
    global df_aristas
    
    df_nodos = pd.read_csv(file1)
    df_aristas = pd.read_csv(file2)
    
    numeric_features = df_nodos.columns.tolist()[1:-2]  # get all columns except lat and long
    lat_long_nodes = df_nodos.columns.tolist()[-2:] # get just lat and long
    print('lat_long_nodes', lat_long_nodes)

    #labels = []
    for i in numeric_features:
        labels.append(df_nodos[i].sum())
    print('++++ frecuencias', labels, 'enbd', type(labels))
    # For DEBUGGING
    #print('ooo',numeric_features)
    global mapping_lat_long, mapping
    mapping = {}
    mapping_lat_long = {}

    # Add nodes
    for i in range(len(df_nodos)):
        #print(df_nodos.iloc[i,0], df_nodos.iloc[i,1])
        nodo_id = G.number_of_nodes() 
        mapping[df_nodos['Nodo'][i]] = nodo_id
        features_values = [df_nodos[x][i] for x in numeric_features]# lat_long_nodes
        #print('feat_nodes', [df_nodos[x][i] for x in numeric_features])
        G.add_node(nodo_id, features={"crime_type": features_values })
        mapping_lat_long[nodo_id] = [df_nodos['lat'][i] ,df_nodos['long'][i] ]
        #labels.append(features_values)

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
    file3 = 'nodos_lat_long'
    hidden_dim = 0
    labels = []
    numeric_features = []
    weekdays = []
    choice = ''

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
            subgraph = '52'
            '''
            '52' // Case Study
            '6'  // Av. Paulista Bairro perigoso
            '7'  // Bairro Vila Madalena (tranquilo)
            '8'  // Perto da Rodoviaria Tiete
            '''
        
            if (sys.argv[2] =="-f=1" or sys.argv[2] == "-1"):
                choice = 'Frequency'
                file_str = subgraph+'all.csv'
                bins_customized = 'auto' #max(max(arr) for arr in labels)
                hidden_dim = 1
                weekdays = np.array(['Frequency'])

            elif(sys.argv[2] =="-f=month" or sys.argv[2] == "-m"):
                choice = 'month'
                file_str = subgraph+'month.csv'
                hidden_dim = 8 #12
                bins_customized = 12
                weekdays = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct','Nov','Dec'])

            elif(sys.argv[2] =="-f=period" or sys.argv[2] == "-p" ):
                choice = 'period'
                file_str = subgraph+'period.csv'
                hidden_dim = 5
                bins_customized = 5
                weekdays = np.array(['Unknow', 'morning', 'afternoon', 'night', 'early_morning'])
            elif(sys.argv[2] == "-f=week" or sys.argv[2] == "-w"):
                choice = 'weekday'
                file_str = subgraph+'week.csv'
                hidden_dim = 6 #7
                bins_customized=7
                weekdays = np.array(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        out_dim = 4

        file1 += file_str
        file2 += subgraph+'.csv'
        file3 += subgraph+'.csv'
        
        numeric_features = read_graph_from_csv(file1, file2, G, edges, labels)
        plt.bar(weekdays, labels, color ='#F47C2F',width = 0.4)
        #plt.hist(labels, bins=bins_customized)
        plt.title("Subgraph crime histogram")

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
    # print(' ================ node_features', node_features)
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


    from torch_geometric.nn import GAE
    # Definir el modelo 2 del grafo autoencoder
    class GCNEncoder(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim):
            torch.random.manual_seed(37)
            super(GCNEncoder, self).__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim, cached=True) # cached only for transductive learning
            self.conv2 = GCNConv(hidden_dim, out_dim, cached=True) # cached only for transductive learning

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            return self.conv2(x, edge_index)

    
    # Definir el modelo 1 del grafo autoencoder
    """class GraphAutoencoder(nn.Module):
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
            return x"""
            
    
    # Definir los hiperparámetros del modelo 1
    input_dim = features.size(1)
    #print('######################3', features)
    
    #hidden_dim = 40 # Already defined above 
    
    # Crear una instancia del modelo 2
    model = GAE(GCNEncoder(input_dim, hidden_dim))

    # Crear una instancia del modelo 1
    # model = GraphAutoencoder(input_dim, hidden_dim)

    # Pro model 2 
    # move to GPU (if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Definir la función de pérdida y el optimizador
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Entrenar el modelo 2
    def train():
        model.train()
        optimizer.zero_grad()
        z = model.encode(features, edge_index)
        loss = model.recon_loss(z, edge_index)
        #if args.variational:
        #   loss = loss + (1 / data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()
        return float(loss)
    
    # Test para el modelo 2
    """def test(pos_edge_index, neg_edge_index):
        model.eval()
        with torch.no_grad():
            z = model.encode(features, edge_index)
        return model.test(z, pos_edge_index, neg_edge_index)"""

    num_epochs = 100

    # Entrenar el modelo 2
    
    for epoch in range(1, num_epochs+1):
        loss = train()
        #auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
        #print('Epoch: {:03d}'.format(epoch))
    
    
    # Entrenar el modelo 1
    """num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(features, edge_index)
        #print(train_nodes, 'len = ', (outputs))
        loss = criterion(outputs[train_nodes], features[train_nodes])
        loss.backward()
        optimizer.step()"""
    
    # Obtener las representaciones embebidas de los nodos para el modelo 1
    #embeddings = model(features, edge_index).detach().numpy()

    
    # Encoding for model 1 
    """x = model.conv1(features, edge_index)
    x = torch.relu(x)
    embeddings = model.conv2(x, edge_index)
    embeddings = embeddings.detach().numpy()"""

    # Encoding for model 2
    embeddings = model.encode(features, edge_index).detach().numpy()
    
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
    
    plt.figure(2)

    ########### TIME-SERIES ############

    import datetime
    node_features2 = pd.DataFrame.from_dict(data)

    def get_fetures_of_Node_from_Graph(number_node, verbose = False):
        if verbose: print(number_node, '=', node_features2.iloc[number_node].values)
        return node_features2.iloc[number_node].values

    num_nodes = input('Enter the AMOUNT of nodes: ')
    nodes = []
    for i in range(int(num_nodes)):
        variable = input('# node ')
        nodes.append(int(variable))
    print('ok')
    plt.subplot(221)

    # Defined Before  
    # weekdays = np.array([...])
    print(len(weekdays))
    frequencies_nodes_selected = np.array([0] * len(weekdays))
    for i in range(int(num_nodes)):
        frequency = get_fetures_of_Node_from_Graph(nodes[i], True)
        dataframe = pd.DataFrame({'date_of_week': weekdays,
                              'frequency': frequency})
        frequencies_nodes_selected += np.array(frequency)
        plt.plot(weekdays, dataframe.frequency, marker='o', label="node "+str(nodes[i]))

    # Giving title to the chart using plt.title and a legend
    plt.title('Crimes by '+ choice)
    plt.legend(loc="upper left")
 
    # rotating the x-axis tick labels at 30degree  towards right
    plt.xticks(rotation=30, ha='right')
 
    # Providing x and y label to the chart
    plt.xlabel(choice.capitalize()+'s')
    plt.ylabel('Amount of Crimes')
    
    ########### TIME-SERIES ############


    plt.subplot(122)
    color_map = ['blue'] * len(node_features.index)
    for node in nodes:
        color_map[node] = 'red'
        mapping_lat_long[node].append('red')
        print(node,'+-----------------------------------------+', mapping_lat_long[node])
        

    nx.draw(G, node_color=color_map, with_labels=True)
    color_map2 = color_map
    
    plt.subplot(223)
    likeability_scores = np.array(frequencies_nodes_selected)
    data_normalizer = mp.colors.Normalize()
    color_map = mp.colors.LinearSegmentedColormap(
        "my_map",
        {
            "red": [(0, 1.0, 1.0),
                    (1.0, .5, .5)],
            "green": [(0, 0.5, 0.5),
                    (1.0, 0, 0)],
            "blue": [(0, 0.50, 0.5),
                    (1.0, 0, 0)]
        }
        )
    plt.bar(weekdays, frequencies_nodes_selected, 
            color=color_map(data_normalizer(likeability_scores))
            ,width = 0.4)

    plt.show()
    #umap 
    #t-sne

    print("@@@@@@@@@@@@@@@",mapping_lat_long)
   

    #### show a map with folium ####
    lat_long = [-23.545043, -46.703592]
    
    import webbrowser
    
    m = folium.Map(location=[lat_long[0], lat_long[1]], zoom_start=15, tiles='CartoDB positron')

    """
    nearest=ch.query_point_in_city_mesh(lat_long[0], lat_long[1], True)
    print(nearest)
    folium.CircleMarker(rectangle_center, radius=5, color='blue').add_to(m)
    folium.CircleMarker(ch.city_vert_list[nearest], radius=8, color='pink').add_to(m)
    """
    # edges
    for i in range(len(df_aristas)):
        folium.PolyLine([(df_aristas['lat1'][i], df_aristas['long1'][i]), 
                         (df_aristas['lat2'][i], df_aristas['long2'][i]) ],
                           color="yellow", weight=2.5, opacity=1).add_to(m)

    # nodes
    
    for vert in mapping_lat_long:
        color_node = 'blue'
        
        lat = mapping_lat_long[vert][0]
        long = mapping_lat_long[vert][1]
        color_node = mapping_lat_long[vert][2] if len(mapping_lat_long[vert]) == 3 else 'blue'
        
        folium.CircleMarker( [lat, long], 
                            radius=5, color=color_node,
                            popup=str(vert)+' '+str(lat)+','+str(long)).add_to(m)
    
    print('**********************',mapping)
    
    m.save("map.html")
    webbrowser.open("map.html")
    #m.showMap()