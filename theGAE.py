"""
Env: chatHqu
Est3e codigo SI incluye las aristas del grafo, CONVOLUCIONES

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
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch_geometric.nn import GCNConv
import torch_geometric.utils as utils

# Crear el grafo de Networkx con nodos geolocalizados y 8 características
G = nx.Graph()

# Agregar nodos con sus características geolocalizadas
G.add_node(0, features={"latitude": 42.3542, "longitude": -71.0661, "crime_type": "1",
                         "population_density": 500, "income_level": "High", "education_level": "Bachelor", "age_median": 30, "employment_rate": 0.85})
G.add_node(1, features={"latitude": 42.3601, "longitude": -71.0589, "crime_type": "2", 
                        "population_density": 800, "income_level": "Medium", "education_level": "High School", "age_median": 35, "employment_rate": 0.75})
G.add_node(2, features={"latitude": 42.3662, "longitude": -71.0621, "crime_type": "3",
                         "population_density": 600, "income_level": "Low", "education_level": "None", "age_median": 40, "employment_rate": 0.60})
G.add_node(3, features={"latitude": 42.3652, "longitude": -71.0621, "crime_type": "2",
                         "population_density": 600, "income_level": "Low", "education_level": "Bachelor", "age_median": 40, "employment_rate": 0.50})
G.add_node(4, features={"latitude": 42.3542, "longitude": -71.0661, "crime_type": "1",
                         "population_density": 500, "income_level": "High", "education_level": "Bachelor", "age_median": 30, "employment_rate": 0.85})
G.add_node(5, features={"latitude": 42.3601, "longitude": -71.0589, "crime_type": "2", 
                        "population_density": 800, "income_level": "Medium", "education_level": "High School", "age_median": 35, "employment_rate": 0.75})
G.add_node(6, features={"latitude": 42.3662, "longitude": -71.0621, "crime_type": "3",
                         "population_density": 600, "income_level": "Low", "education_level": "None", "age_median": 40, "employment_rate": 0.60})
G.add_node(7, features={"latitude": 42.3652, "longitude": -71.0621, "crime_type": "2",
                         "population_density": 600, "income_level": "Low", "education_level": "Bachelor", "age_median": 40, "employment_rate": 0.50})
G.add_node(8, features={"latitude": 42.3652, "longitude": -71.0621, "crime_type": "0",
                         "population_density": 600, "income_level": "Low", "education_level": "v", "age_median": 40, "employment_rate": 0.50})
G.add_node(9, features={"latitude": 42.3652, "longitude": -71.0621, "crime_type": "0",
                         "population_density": 600, "income_level": "Medium", "education_level": "Bachelor", "age_median": 0, "employment_rate": 0.50})
# Agregar más nodos con características geolocalizadas según sea necesario

# Crear un DataFrame con las características de los nodos
node_features = pd.DataFrame.from_dict(
    {node: data['features'] for node, data in G.nodes(data=True)}, orient='index'
)

# Preprocesar las características no binarias
non_binary_features = ['crime_type', 'income_level', 'education_level']
label_encoders = {}
for feature in non_binary_features:
    label_encoders[feature] = LabelEncoder()
    node_features[feature] = label_encoders[feature].fit_transform(node_features[feature])

# Normalizar las características numéricas
numeric_features = ['population_density', 'age_median', 'employment_rate'] # 'latitude', 'longitude', 
scaler = StandardScaler()
node_features[numeric_features] = scaler.fit_transform(node_features[numeric_features])

# Dividir los nodos en entrenamiento y prueba
train_nodes, test_nodes = train_test_split(node_features.index, train_size=0.8, test_size=0.2)

# Convertir las características en tensores de PyTorch
features = torch.FloatTensor(node_features.values)

# Crear los edges en el grafo
edges = [(0, 1), (0, 2),(1, 3), (4, 1), (5, 2), (2, 6), (2, 7)]  # Ejemplo de edges, modificar según la estructura de tu grafo

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
hidden_dim = 64

# Crear una instancia del modelo
model = GraphAutoencoder(input_dim, hidden_dim)

# Definir la función de pérdida y el optimizador
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
print('############################################')

# Entrenar el modelo
num_epochs = 50
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(features, edge_index)
    print(len(outputs[train_nodes]), len(features[train_nodes]))
    loss = criterion(outputs[train_nodes], features[train_nodes])
    loss.backward()
    optimizer.step()
print('############################################')
# Obtener las representaciones embebidas de los nodos
embeddings = model(features, edge_index).detach().numpy()

# Aplicar algoritmo de clustering en las representaciones embebidas
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0)
clusters = kmeans.fit_predict(embeddings)

# Imprimir los resultados del clustering
for node, cluster in zip(node_features.index, clusters):
    print(f"Nodo {node} -> Cluster {cluster}")
