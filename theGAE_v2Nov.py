import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import datetime
from torch_geometric.utils import subgraph
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_networkx, subgraph
from networkx.algorithms.community import greedy_modularity_communities
from torch_geometric.data import Data
import networkx as nx

"""

Run the project:

> python3 theGAE_v2Nov.py

Files: 
    file_nodes = 'dataset_germina.csv'
    file_edges = 'aristas_subgrafoSPdaily.csv'

Change the Hyperparameters:
    input_dim = node_features_tensor
    hidden_dim =  int(input_dim/2)
    hidden_dim2 = int(hidden_dim/2)
    out_dim = int(hidden_dim2/2)
    n_epochs = 100

Change the parameters: 
    typeData : 'A' = all, 'C' = crimes , 'S' = Census, 'I' = Infraestructure

"""

# Auxiliary function
def print_shape_or_size(variable):
    """Prints the shape or size of the variable if it has the 'shape' or 'size' attributes."""
    if hasattr(variable, 'shape'):
        print(f"Shape: {variable.shape}")
    elif hasattr(variable, 'size'):
        print(f"Size: {variable.size}")
    else:
        print("La variable no tiene atributos 'shape' ni 'size'.")

# Data processing and generation of tensors with node and edge features
def read_graph_from_csv(file_nodes, file_edges, feature_type='A', normalize=True):
    """
    Reads and processes nodes and edges from CSV files to generate feature and edge tensors.

    Args:
        file_nodes (str): nodes CSV file
        file_edges (str): edges CSV file
        feature_type (str): Type of features to select for nodes ('A', 'IC', 'IS', 'SC', 'CS', 'C').        normalize (bool): Indica si se debe normalizar las características de los nodos.

    Returns:
        tuple: Node feature tensor, edge index tensor, and selected columns
    """
    # Loading node and edge data
    df_nodes = pd.read_csv(file_nodes)
    df_edges = pd.read_csv(file_edges)

    # Create a mapping of unique indexes to positions
    unique_indices = df_nodes['Nodo'].unique()
    index_mapping = {index: pos for pos, index in enumerate(unique_indices)}

    # Apply mapping to edges
    df_edges['Nodo1'] = df_edges['Nodo1'].map(index_mapping)
    df_edges['Nodo2'] = df_edges['Nodo2'].map(index_mapping)

    # Filter edges to remove those with NaN values ​​after mapping
    df_edges = df_edges.dropna(subset=['Nodo1', 'Nodo2'])

    # Convert edge data to a tensor
    edge_index_tensor = torch.tensor(df_edges[['Nodo1', 'Nodo2']].values.T, dtype=torch.long)
    print('edge_index_tensor 1', edge_index_tensor)

    # Node feature normalization if enabled
    if normalize:
        scaler = StandardScaler()
        columns_to_normalize = df_nodes.columns[1:-2]  # Excluding first (Nodo) and last 2 columns (lat, long) 
        df_nodes[columns_to_normalize] = scaler.fit_transform(df_nodes[columns_to_normalize])

    # Determining column ranges based on feature type
    range_start, range_end, range_start_2, range_end_2 = get_feature_column_ranges(feature_type, df_nodes)

    # Feature column selection
    selected_columns = select_columns(df_nodes, range_start, range_end, range_start_2, range_end_2)

    # Node feature tensor generation
    node_features = df_nodes[selected_columns].values
    node_features_tensor = torch.tensor(node_features, dtype=torch.float)
    #print('selected_columns', len(selected_columns) )
    print('node_features', node_features.shape)
    print(" !!!!! creo qh este es Xime", node_features_tensor.shape)
    # Filtrado y generación de tensor de índices de aristas
    #edge_index_tensor = filter_edges(df_nodes, df_edges)
    #print('edges 2', edge_index_tensor)

    return node_features_tensor, edge_index_tensor, selected_columns

def get_feature_column_ranges(feature_type, df_nodes):
    """Determine the column ranges to select based on the feature type"""
    if feature_type == 'A':  # All features
        return 1, -2, None, None
    elif feature_type == 'IC':  # Infrastructure + Crimes
        return 1, 145, 152, -2
    elif feature_type == 'IS':  # Infrastructure + Census
        return 1, 1 , 145, -2
    elif feature_type in ('SC', 'CS'):  # Census + Crimes
        return 1, 152, None, None
    elif feature_type == 'C':  # Only Crimes
        return 1, 145, None, None
    else:
        raise ValueError("Invalid feature type")

def select_columns(df_nodes, range_start, range_end, range_start_2=None, range_end_2=None):
    """Select feature columns according to the specified ranges"""
    if range_start_2 is not None and range_end_2 is not None:
        selected_columns = df_nodes.columns.tolist()[range_start:range_end] + \
                           df_nodes.columns.tolist()[range_start_2:range_end_2]
    else:
        selected_columns = df_nodes.columns.tolist()[range_start:range_end]
    return selected_columns

def filter_edges(df_nodes, df_edges):
    """Filter edges to include only those whose nodes are present in the node set and generate a tensor"""
    node_set = set(df_nodes['Nodo'])
    filtered_edges = df_edges[df_edges['Nodo1'].isin(node_set) & df_edges['Nodo2'].isin(node_set)]
    return torch.tensor(filtered_edges[['Nodo1', 'Nodo2']].values.T, dtype=torch.long)

def community_based_split(node_features_tensor, edge_index_tensor, train_ratio=0.8):
    """
    Split the data into training and test graphs based on communities, using modularity heuristics.
    
    Args:
        node_features_tensor (torch.Tensor): Node feature tensor
        edge_index_tensor (torch.Tensor): Edge index tensor
        train_ratio (float): Proportion of nodes for the training subset

    Returns:
        tuple: Training and test tensors (train_node_features, train_edge_index, test_node_features, test_edge_index).
    """
    
    # Create a Data object for the graph
    data = Data(x=node_features_tensor, edge_index=edge_index_tensor)

    # Convert edge_index to a NetworkX graph
    G = nx.Graph()
    edge_index_np = edge_index_tensor.numpy()  # Convert the tensor to NumPy

    # Add edges to the graph
    G.add_edges_from(edge_index_np.T)  # Transpose to be (Node1, Node2)
    
    # Detect communities using the modularity heuristic
    communities = list(greedy_modularity_communities(G))
    
    # Sort the communities by size in descending order
    communities = sorted(communities, key=len, reverse=True)
    
    # Define training and test nodes
    train_nodes = set()
    test_nodes = set()
    
    total_nodes = G.number_of_nodes()
    target_train_size = int(total_nodes * train_ratio)
    print('total_nodes', total_nodes)
    print('target_train_size', target_train_size)
    
    # Combinar comunidades hasta obtener el 80% de nodos en el subconjunto de entrenamiento
    for community in communities:
        if len(train_nodes) + len(community) <= target_train_size:
            train_nodes.update(community)
        else:
            test_nodes.update(community)
    
    # Asegurar que los nodos restantes se asignen al subconjunto de prueba
    if len(train_nodes) < target_train_size and communities:
        remaining_community = communities[len(train_nodes) % len(communities)]
        train_nodes.update(remaining_community)
        communities.remove(remaining_community)
        for community in communities:
            test_nodes.update(community)

    train_graph = G.subgraph(train_nodes).copy()
    test_graph = G.subgraph(test_nodes).copy()
    
    # Create training and test subgraphs from the selected nodes
    train_nodes = torch.tensor(list(train_nodes), dtype=torch.long)
    test_nodes = torch.tensor(list(test_nodes), dtype=torch.long)
   
    # Filter the feature tensor and edge_index_tensor using the indices of filtered_train_nodes    
    train_node_features = node_features_tensor[train_nodes]
    test_node_features = node_features_tensor[test_nodes]

    train_edge_index, _ = subgraph(train_nodes, edge_index_tensor, relabel_nodes=True)
    test_edge_index, _ = subgraph(test_nodes, edge_index_tensor, relabel_nodes=True)
    
    ## test of nan ###
    """
    print(' ++ ', torch.isnan(train_node_features).any())
    nan_indices = torch.isnan(train_node_features).nonzero(as_tuple=True)
    print("Indices de NaN en train_node_features:", len(nan_indices))
    print(train_node_features.min(), train_node_features.max())
    """
    #### end test
   
    #print(' test nodes', train_node_features)
    #print(' test indices' , train_edge_index)
    #print(' test nodes', test_node_features )
    #print(' test indexs', test_edge_index)

    return train_node_features, train_edge_index, test_node_features, test_edge_index

# GraphAutoencoder model
class GCNEncoder(torch.nn.Module): 
    def __init__(self, input_dim, hidden_dim, hidden_dim2, out_dim): 
        torch.random.manual_seed(37) 
        super(GCNEncoder, self).__init__() 
        self.conv1 = GCNConv(input_dim, hidden_dim, cached=True) 
        self.conv2 = GCNConv(hidden_dim, hidden_dim2, cached=True) 
        self.conv3 = GCNConv(hidden_dim2, out_dim, cached=True) 
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu() 
        x = self.conv2(x, edge_index).relu() 
        return self.conv3(x, edge_index) 

class Decoder(torch.nn.Module):
    def __init__(self, in_features, hidden_dim, hidden_dim2, out_features):
        super(Decoder, self).__init__() 
        self.deconv1 = GCNConv(in_features, hidden_dim2, cached=True)
        self.deconv2 = GCNConv(hidden_dim2, hidden_dim, cached=True) 
        self.deconv3 = GCNConv(hidden_dim, out_features, cached=True) 
        
    def forward(self, x, edge_index): 
        x = self.deconv1(x, edge_index).relu() 
        x = self.deconv2(x, edge_index).relu() 
        x = self.deconv3(x, edge_index)
        return x

class GraphAutoencoder(torch.nn.Module): 
    def __init__(self, input_dim, hidden_dim, hidden_dim2, out_dim):
        super(GraphAutoencoder, self).__init__() 
        self.encoder = GCNEncoder(input_dim, hidden_dim, hidden_dim2, out_dim) 
        self.decoder = Decoder(out_dim, hidden_dim2, hidden_dim, input_dim) 
        
    def forward(self, x, edge_index):
        encoded = self.encoder(x, edge_index) 
        decoded = self.decoder(encoded, edge_index) 
        return decoded

if __name__ == '__main__':
    file_nodes = 'dataset_germina.csv'
    file_edges = 'aristas_subgrafoSPdaily.csv'
    typeData = 'A'
    node_features_tensor, edge_index_tensor, selected_columns = read_graph_from_csv(file_nodes, file_edges, typeData)

    print_shape_or_size(node_features_tensor)
    print_shape_or_size(edge_index_tensor)
    #print("Selected columns:", selected_columns)

    # Community Split 
    train_node_features, train_edge_index, test_node_features, test_edge_index = community_based_split(
        node_features_tensor, edge_index_tensor, train_ratio=0.8
    )

    # Transpose the edge tensor
    #train_edge_index = train_edge_index.t()

    # Results
    print("Training node features:", train_node_features.shape)
    print("Training edge indices:", train_edge_index.shape)
    print("Test node features:", test_node_features.shape)
    print("Test edge indices:", test_edge_index.shape)

    # Define the model parameters 
    input_dim = node_features_tensor.shape[1]
    hidden_dim =  int(input_dim/2)
    hidden_dim2 = int(hidden_dim/2)
    out_dim = int(hidden_dim2/2)
    n_epochs = 100

    print('Hyperparameters: ', input_dim, ' --> ', hidden_dim, ' --> ', hidden_dim2, ' --> ',  out_dim )
    
    model = GraphAutoencoder(input_dim, hidden_dim, hidden_dim2, out_dim)
    
    # Set up the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 
    criterion = torch.nn.MSELoss() 
    
    # Define the training function 
    def train():
        model.train()
        optimizer.zero_grad()
        output = model(train_node_features, train_edge_index) 
        loss = criterion(output, train_node_features) 
        loss.backward()
        optimizer.step() 
        print('loss:', loss) 
        return float(loss.item()) 
    
    # Call the training function
    for epoch in range(n_epochs): 
        train_loss = train() 
        print(f"Epoch {epoch+1}, Loss: {train_loss}")
