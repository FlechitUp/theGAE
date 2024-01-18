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
from collections import Counter
import ast
import random
import folium
import webbrowser   #para ver el mapa
import time
import datetime

# Path to save the weights and parameters of the model
model_path = 'gae_model'
model_path0 = 'gae_model100.pth'
model_path2 = 'gae_model1000.pth'
model_path3 = 'gae_model10000.pth'
model_loaded_bool = False  # False for training

np.random.seed(37)
Fig1 = False

def generate_sufix_for_file():
    # Get the current date and time
    fecha_hora_actual = datetime.datetime.now()

    # Convert the date and time into a string with the desired format
    fecha_hora_formateada = fecha_hora_actual.strftime("%d%m%Y%H%M%S")
    return '_'+fecha_hora_formateada

def add_edges( G, edge_tuples):
    for edge in edge_tuples:
        G.add_edge(edge[0], edge[1])

def read_graph_from_csv(file1, file2, G, edges, labels):
    global lat_long_nodes
    global df_aristas
    
    df_nodos = pd.read_csv(file1)
    df_aristas = pd.read_csv(file2)

    numeric_features = df_nodos.columns.tolist()[1:-2]  # get all columns except lat and long
    lat_long_nodes = df_nodos.columns.tolist()[-2:] # get just lat and long
    
    for i in numeric_features:
        labels.append(df_nodos[i].sum())
    
    # For DEBUGGING
    global mapping_lat_long, mapping
    mapping = {}
    mapping_lat_long = {}

    print('adding nodes')
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

    """nodos_lat_long_sem_crimes = get_nodes_lat_long_without_crimes(df_nodos, df_aristas)
    for i in range(len(nodos_lat_long_sem_crimes)):
        nodo_id = G.number_of_nodes() 
    """
    print('adding edges')
    # Add edges
    for i in range(len(df_aristas)):

        try:
            nodo1_id = mapping[df_aristas['Nodo1'][i]]
        except KeyError:
            #print("La clave no existe en el diccionario.")
            nodo_id = G.number_of_nodes() 
            mapping[df_aristas['Nodo1'][i]] = nodo_id
            nodo1_id = mapping[df_aristas['Nodo1'][i]]
            #print(i, df_aristas['long1'][1]) #df_aristas['lat1'][i], df_aristas['long1'][i] )
            mapping_lat_long[nodo_id] = [df_aristas['lat1'][i], df_aristas['long1'][i] ]
        
        try:
            nodo2_id = mapping[df_aristas['Nodo2'][i]]
        except KeyError:
            #print("La clave no existe en el diccionario.")
            nodo_id = G.number_of_nodes() 
            mapping[df_aristas['Nodo2'][i]] = nodo_id
            nodo2_id = mapping[df_aristas['Nodo2'][i]]
            mapping_lat_long[nodo_id] = [df_aristas['lat2'][i] ,df_aristas['long2'][i] ]
        
        G.add_edge(nodo1_id, nodo2_id)
        edges.append((nodo1_id, nodo2_id))
        
    return numeric_features

def help_function():
    print("Run with a graph using csv: \n python theGAE.py -csv -dd hidden_dim output_dim num_epochs")
    #print("Run with a graph using csv with just one feature: \n python theGAE.py -csv -f=1")

# python mainGae.py -csv -dd -h=8
    
if __name__ == '__main__':
    # Crear el grafo de Networkx con nodos geolocalizados y 8 caracteristicas
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

    # Save the start time
    start_time = time.time()
    sufix_for_file = generate_sufix_for_file()

    # Menu
    if (len(sys.argv)==1 or sys.argv[1] =="-help"):
        help_function()
        sys.exit()

    elif(sys.argv[1] =="-csv"):
        print("Graph from csv")
        n_components = 3 # contar_elementos_no_repetidos(labels)
        
        # criar histograma naranja por quantidade de crime
        if (Fig1): plt.subplot(232)
        
        if ( len(sys.argv) == 6 ):
            print("Graph from csv with hidden_dim, output_dim and num_epochs")
            subgraph = 'SPdaily' #'Tt8' 'VM7' #82 'SP'
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
                #hidden_dim = 1
                weekdays = np.array(['Frequency'])

            elif(sys.argv[2] =="-f=month" or sys.argv[2] == "-m"):
                choice = 'month'
                file_str = subgraph+'month.csv'
                #hidden_dim = 8 #12
                bins_customized = 12
                weekdays = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct','Nov','Dec'])

            elif(sys.argv[2] == "-md"): # month demographic
                choice = 'month with demographic socio-economic'
                file_str = subgraph+'_DEMOG.csv'
                #hidden_dim = 18 
                bins_customized = 32
                # Nodo,january,february,march,april,may,june,july,august,september,october,november,december,lat,long,Household_income_avg,Householder_income_avg,Householder_unemployment_rate,Literate_7_15_yrs_children_rate,residents_under_18_years_rate,residents_aged_18_to_65_years_rate,residents_over_65_years_rate,bus_stops,subway_stations,train_stations,bus_terminals,subnormal_agglomerates_around,crime_mobile,crime_vehicle,crime_all,precipitation_total,temperature_max,temperature_min
                weekdays = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct','Nov','Dec','Renda_media_por_domicilio', 'Renda_media_responsaveis', 'Responsaveis_sem_renda_taxa', 'Alfabetizados_de_7_a_15_anos', 'menores_de_18_anos_taxa', '18_a_65_anos_taxa', 'maiores_de_65_anos_taxa', 'Pontos_de_onibus', 'Estacao_de_metro', 'Estacao_de_trem', 'Terminal_de_onibus', 'Favela_proxima'])

            elif(sys.argv[2] == "-dd"): # daily demographic
                choice = 'daily with demographic socio-economic'
                file_str = subgraph+'_DEMOG.csv'
                #hidden_dim = 30 
                bins_customized = 32
                # Nodo,january,february,march,april,may,june,july,august,september,october,november,december,lat,long,Household_income_avg,Householder_income_avg,Householder_unemployment_rate,Literate_7_15_yrs_children_rate,residents_under_18_years_rate,residents_aged_18_to_65_years_rate,residents_over_65_years_rate,bus_stops,subway_stations,train_stations,bus_terminals,subnormal_agglomerates_around,crime_mobile,crime_vehicle,crime_all,precipitation_total,temperature_max,temperature_min
                weekdays = np.array(['d0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9', 'd10', 'd11', 'd12', 'd13', 'd14', 'd15', 'd16', 'd17', 'd18', 'd19', 'd20', 'd21', 'd22', 'd23', 'd24', 'd25', 'd26', 'd27', 'd28', 'd29', 'd30', 'd31', 'd32', 'd33', 'd34', 'd35', 'd36', 'd37', 'd38', 'd39', 'd40', 'd41', 'd42', 'd43', 'd44', 'd45', 'd46', 'd47', 'd48', 'd49', 'd50', 'd51', 'd52', 'd53', 'd54', 'd55', 'd56', 'd57', 'd58', 'd59', 'd60', 'd61', 'd62', 'd63', 'd64', 'd65', 'd66', 'd67', 'd68', 'd69', 'd70', 'd71', 'd72', 'd73', 'd74', 'd75', 'd76', 'd77', 'd78', 'd79', 'd80', 'd81', 'd82', 'd83', 'd84', 'd85', 'd86', 'd87', 'd88', 'd89', 'd90', 'd91', 'd92', 'd93', 'd94', 'd95', 'd96', 'd97', 'd98', 'd99', 'd100', 'd101', 'd102', 'd103', 'd104', 'd105', 'd106', 'd107', 'd108', 'd109', 'd110', 'd111', 'd112', 'd113', 'd114', 'd115', 'd116', 'd117', 'd118', 'd119', 'd120', 'd121', 'd122', 'd123', 'd124', 'd125', 'd126', 'd127', 'd128', 'd129', 'd130', 'd131', 'd132', 'd133', 'd134', 'd135', 'd136', 'd137', 'd138', 'd139', 'd140', 'd141', 'd142', 'd143', 'd144', 'd145', 'd146', 'd147', 'd148', 'd149', 'd150', 'd151', 'd152', 'd153', 'd154', 'd155', 'd156', 'd157', 'd158', 'd159', 'd160', 'd161', 'd162', 'd163', 'd164', 'd165', 'd166', 'd167', 'd168', 'd169', 'd170', 'd171', 'd172', 'd173', 'd174', 'd175', 'd176', 'd177', 'd178', 'd179', 'd180', 'd181', 'd182', 'd183', 'd184', 'd185', 'd186', 'd187', 'd188', 'd189', 'd190', 'd191', 'd192', 'd193', 'd194', 'd195', 'd196', 'd197', 'd198', 'd199', 'd200', 'd201', 'd202', 'd203', 'd204', 'd205', 'd206', 'd207', 'd208', 'd209', 'd210', 'd211', 'd212', 'd213', 'd214', 'd215', 'd216', 'd217', 'd218', 'd219', 'd220', 'd221', 'd222', 'd223', 'd224', 'd225', 'd226', 'd227', 'd228', 'd229', 'd230', 'd231', 'd232', 'd233', 'd234', 'd235', 'd236', 'd237', 'd238', 'd239', 'd240', 'd241', 'd242', 'd243', 'd244', 'd245', 'd246', 'd247', 'd248', 'd249', 'd250', 'd251', 'd252', 'd253', 'd254', 'd255', 'd256', 'd257', 'd258', 'd259', 'd260', 'd261', 'd262', 'd263', 'd264', 'd265', 'd266', 'd267', 'd268', 'd269', 'd270', 'd271', 'd272', 'd273', 'd274', 'd275', 'd276', 'd277', 'd278', 'd279', 'd280', 'd281', 'd282', 'd283', 'd284', 'd285', 'd286', 'd287', 'd288', 'd289', 'd290', 'd291', 'd292', 'd293', 'd294', 'd295', 'd296', 'd297', 'd298', 'd299', 'd300', 'd301', 'd302', 'd303', 'd304', 'd305', 'd306', 'd307', 'd308', 'd309', 'd310', 'd311', 'd312', 'd313', 'd314', 'd315', 'd316', 'd317', 'd318', 'd319', 'd320', 'd321', 'd322', 'd323', 'd324', 'd325', 'd326', 'd327', 'd328', 'd329', 'd330', 'd331', 'd332', 'd333', 'd334', 'd335', 'd336', 'd337', 'd338', 'd339', 'd340', 'd341', 'd342', 'd343', 'd344', 'd345', 'd346', 'd347', 'd348', 'd349', 'd350', 'd351', 'd352', 'd353', 'd354', 'd355', 'd356', 'd357', 'd358', 'd359', 'd360', 'd361', 'd362', 'd363', 'd364', 'd365','Renda_media_por_domicilio', 'Renda_media_responsaveis', 'Responsaveis_sem_renda_taxa', 'Alfabetizados_de_7_a_15_anos', 'menores_de_18_anos_taxa', '18_a_65_anos_taxa', 'maiores_de_65_anos_taxa', 'Pontos_de_onibus', 'Estacao_de_metro', 'Estacao_de_trem', 'Terminal_de_onibus', 'Favela_proxima'])

            elif(sys.argv[2] == "-mto"):
                choice = 'month with Tipe of crime'
                file_str = subgraph+'monthTO.csv'
                #hidden_dim = 8 #12
                bins_customized = 14
                weekdays = np.array(['TIPO_OCORRENCIA_FURTO', 'TIPO_OCORRENCIA_ROUBO', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct','Nov','Dec'])

            elif(sys.argv[2] =="-f=period" or sys.argv[2] == "-p" ):
                choice = 'period'
                file_str = subgraph+'period.csv'
                #hidden_dim = 5
                bins_customized = 5
                weekdays = np.array(['Unknow', 'morning', 'afternoon', 'night', 'early_morning'])
            
            elif(sys.argv[2] == "-f=week" or sys.argv[2] == "-w"):
                choice = 'weekday'
                file_str = subgraph+'week.csv'
                #hidden_dim = 6 #7
                bins_customized=7
                weekdays = np.array(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
            if (sys.argv[3]):
                hidden_dim = int(sys.argv[3])
            if (sys.argv[4]):
                out_dim = int(sys.argv[4])
            if (sys.argv[5]):
                num_epochs = int(sys.argv[5])
        else:
            print('else', len(sys.argv))
            help_function()
            sys.exit()

        file1 += file_str
        file2 += subgraph+'.csv'
        file3 += subgraph+'.csv'
        
        numeric_features = read_graph_from_csv(file1, file2, G, edges, labels)
        
        if( Fig1):
            plt.bar(weekdays, labels, color ='#F47C2F',width = 0.4)
            #plt.hist(labels, bins=bins_customized)
            plt.title("Subgraph crime histogram")

    # Graph of all nodes in my dataset
    random.seed(37)
    if( Fig1):
        plt.figure(1)
        plt.subplot(231)
        #random.seed(37)
        nx.draw(G, with_labels = True)
    
    #random.seed(37)
    data = {}
    k = 0
    
    for i in numeric_features:
        temp_arr = []
        for node in G.nodes(data=True):
            if 'features' in node[1] and 'crime_type' in node[1]['features']:
                temp_arr.append(node[1]['features']['crime_type'][k])
            else:
                # Agregar features con 'crime_type' en [0,0,0,...,0]
                G.nodes[node[0]]['features'] = {'crime_type': [0] * len(weekdays)}
                temp_arr.append(node[1]['features']['crime_type'][k])
                nodo_id = G.number_of_nodes() 
                
        data[i] = temp_arr
        k = k + 1
    print('fim')
    
    node_features = pd.DataFrame.from_dict(data)

    # Normalize the numeric features of the list: numeric_features
    scaler = StandardScaler()
    node_features[numeric_features] = scaler.fit_transform(node_features[numeric_features])
    
    # Split the nodes into training and testing
    train_nodes, test_nodes = train_test_split(node_features.index, train_size=0.8, test_size=0.2)
    
    # Convert the features into PyTorch tensors
    features = torch.FloatTensor(node_features.values)
    
    # Create the indices of the edges
    edge_index = torch.LongTensor(list(edges)).t().contiguous()
    
    # Get the number of nodes
    num_nodes = len(node_features)
    
    # Ensure that the edge indices are unique and in the correct range
    edge_index, _ = utils.add_self_loops(edge_index, num_nodes=num_nodes)

    from torch_geometric.nn import GAE
    # Define the model of the graph autoencoder

    class GCNEncoder(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, out_dim):
            torch.random.manual_seed(37)
            super(GCNEncoder, self).__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim, cached=True)  # cached only for transductive learning
            self.conv2 = GCNConv(hidden_dim, out_dim, cached=True)    # cached only for transductive learning

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            return self.conv2(x, edge_index)


    print('Torch esta disponivel?', torch.cuda.is_available())
    print('weekdays-->',len(weekdays),'hidden_dim -->', hidden_dim, 'out_dim-->', out_dim)
    # move to GPU (if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # numeric_features.to(device)
    # G.to(device)
    
    # Define the hyperparameters of model 
    input_dim = features.size(1)
    features = features.to(device)
    edge_index = edge_index.to(device)

    # Creating an instance of model 
    model = GAE(GCNEncoder(input_dim, hidden_dim, out_dim))

    # move to GPU (if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Define the loss function and the optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training the model
    def train():
        model.train()
        optimizer.zero_grad()
        z = model.encode(features, edge_index)
        loss = model.recon_loss(z, edge_index)
        loss.backward()
        optimizer.step()
        return float(loss)

    # If I do not have the saved weights of my model
    if (not model_loaded_bool):

        # Training the model
        for epoch in range(1, num_epochs+1):
            loss = train()
            #auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
            #print('Epoch: {:03d}'.format(epoch))
        model_path = model_path + subgraph + sufix_for_file+'.pth'	
        torch.save(model.state_dict(), model_path) 
        print('Model parameters saved ... ',model_path)
    else: 
        # Load the weights and parameters of the saved model
        print('Loading model ...')
        model.load_state_dict(torch.load(model_path+'.pth'))
        print('extra training')

        # extra training
        num_epochs = 10000
        for epoch in range(100, num_epochs+1):
            loss = train()
        model_path = model_path + subgraph + str(num_epochs) +sufix_for_file+'.pth'	
        torch.save(model.state_dict(), model_path) 
        print('Model parameters saved ... step 2 ',model_path)

    # Move the model to the GPU if available
    # loaded_model = loaded_model.to(device)

    # Encoding for model
    embeddings = model.encode(features, edge_index).detach().cpu().numpy()
    print('-- embeddings ')
    df_nodos = pd.read_csv(file1)
    columna_nodo = df_nodos['Nodo']

    # save embeddings with out_dim
    nuevo_dataframe = pd.concat([columna_nodo, pd.DataFrame(embeddings)], axis=1)
    np.savetxt("embeddings" + sufix_for_file + ".txt", nuevo_dataframe, delimiter=",")
    
    # Calculate the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("Tiempo de ejecucion (seg.):", elapsed_time)

    # Creating a metadata file using the sufix_for_file 
    name_metadataFile = "metadata_"+sufix_for_file+".txt"
    with open(name_metadataFile, 'w') as archivo:
        archivo.write("== NAME FILE: "+ sufix_for_file+" == \n")
        archivo.write("Execution time of training (seg.):"+str(elapsed_time)+"\n" )
        archivo.write("+ Hiperparameters: \n"+ "nro of epochs: "+str(num_epochs)+"\n")
        archivo.write("Dimentions: " + str(len(weekdays)) + " -> "+ str(hidden_dim) + " -> " + str(out_dim))

    print(f"Text saved on '{name_metadataFile}'.")
