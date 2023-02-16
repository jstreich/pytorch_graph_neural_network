################################################################################
################################ Import libraries ##############################
################################################################################
print("Loading Libraries...")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


##### This script was originally drafted from ChatGPT, but took significant
# editing to get it to actually work, creating adjacency matrix, GraphNet,
# hyperparameters, plus modifying it to read in the input data, imports, etc.


################################################################################
################################## Load the data ###############################
################################################################################
print("Loading and Format Dataset...")

##### Read the graph file
data = pd.read_csv("n4096-l9.tsv", sep='\t', header=None)
data = pd.DataFrame(data)
print(data[0:])

##### Get the number of nodes
num_nodes = len(set(data[0]).union(data[1]))
nodes = set(data[0]).union(data[1])
print(num_nodes)

#### Adjacency matrix
adjacency_matrix = np.zeros((num_nodes, num_nodes))
adjacency_matrix = pd.DataFrame(adjacency_matrix)


##### get unique names in list and set row and column names
data_uniq = set(data[0]).union(data[1])
adjacency_matrix.columns = data_uniq
adjacency_matrix.index = data_uniq
print(adjacency_matrix[0:])


##### Populate adjecency matrix
for row in data.itertuples(index=False):
    adjacency_matrix.at[row[0], row[1]] = row[2]
    adjacency_matrix.at[row[1], row[0]] = row[2]


##### Set Adjacency matrix coding type
adjacency_matrix = adjacency_matrix.astype(np.float32)

##### Convert the adjacency matrix to a tensor
adjacency_matrix = torch.tensor(adjacency_matrix.values)


################################################################################
################################# Graph Model ##################################
################################################################################
print("Create Transformer Model...")


##### Graph Neural Network
class GraphNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GraphNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

##### Initialize Network
input_size = adjacency_matrix.shape[0]
hidden_size = 16
output_size = 2
model = GraphNet(input_size, hidden_size, output_size)


##### Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

##### Train Model
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(adjacency_matrix)
    loss = criterion(output, torch.tensor([0] * input_size))
    loss.backward()
    optimizer.step()
    print("Epoch = ", epoch, "Loss = ", loss)

###### Evaluate Model
with torch.no_grad():
    output = model(adjacency_matrix)
    predictions = np.argmax(output.numpy(), axis=1)

##### Model Accuracy
accuracy = accuracy_score(np.zeros(input_size), predictions)
print("Accuracy = ", accuracy)

################################################################################
############################# Create Graph Plot ################################
################################################################################



# # # # Note NetworkX and Scipy don't seem too compatible right now and this may not work # # # # 



##### Create Graph
G = nx.Graph()
G.add_nodes_from(nodes)
for row in data.itertuples(index=False):
    G.add_edge(row[0], row[1])

##### Cluster ID To Nodes
cluster_dict = {i: int(predictions[i]) for i in range(input_size)}
nx.set_node_attributes(G, cluster_dict, "cluster")
nx.set_node_attributes(G, cluster_dict)

##### Graph, nodes colored by Cluster ID
colors = [G.nodes[i]["cluster"] for i in G.nodes()]
nx.draw(G, node_color=colors, with_labels=True)
nx.draw(G, with_labels=True)
plt.savefig('mygraph_clusters2.pdf')
plt.show()

##### Pasted Code, needs editing
# colors = pred.cpu().numpy()
plt.figure(figsize=(10,10))
pos = nx.spring_layout(G)
nx.draw(G, pos, cmap='Set2', with_labels=False)
plt.savefig('mygraph_clusters1.pdf')

##### Save Model to File
torch.save(model.state_dict(), "graph_model.pt")
