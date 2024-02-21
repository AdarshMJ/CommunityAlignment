import warnings
warnings.filterwarnings('ignore')
import argparse
import torch
import os
from torch_geometric.datasets import TUDataset, Planetoid,WebKB,Actor, WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures, LargestConnectedComponents
from torch_geometric.utils import to_networkx,from_networkx,homophily
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F
from tqdm import tqdm
import networkx as nx
import numpy as np
import random
import pickle
import time
import csv
#from fastrewiringKupdates import *
from MinGapKupdates import *
from spectral_utils import *
from nodeli import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score as NMI


parser = argparse.ArgumentParser(description='Run BraessGCNPlanetoid script')
parser.add_argument('--dataset', type=str, help='Dataset to download')
parser.add_argument('--out', type=str, help='name of log file')
parser.add_argument('--max_iters', type=int, default=10, help='maximum number of Braess iterations')
parser.add_argument('--update_period', type=int, default=1, help='Times to recalculate criterion')
parser.add_argument('--dropout', type=float, default=0.4130296, help='Dropout = [Cora - 0.4130296, Citeseer - 0.3130296]')
parser.add_argument('--hidden_dimension', type=int, default=32, help='Hidden Dimension size')
parser.add_argument('--LR', type=float, default=0.01, help='Learning Rate = [0.01,0.001]')
parser.add_argument('--device',type=str,default='cpu')
args = parser.parse_args()


######### Hyperparams to use #############
## Cora --> Dropout = 0.4130296 ; LR = 0.01 ; Hidden_Dimension = 32
## Citeseer --> Dropout = 0.3130296 ; LR = 0.01 ; Hidden_Dimension = 32



filename = args.out
device = torch.device(args.device)

def get_graph_and_labels_from_pyg_dataset(dataset):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(dataset.x)))
    graph.add_edges_from(dataset.edge_index.T.numpy())

    labels = dataset.y.numpy()

    return graph, labels


# def visualize(h, color,legend_labels,title, filename,random_state = 13):
#     z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())
#     plt.figure(figsize=(5,5))
#     plt.xticks([])
#     plt.yticks([])
#     scatter = plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
#         # Add legend if legend_labels are provided
#     if legend_labels:
#         legend = plt.legend(handles= scatter.legend_elements()[0], title="Classes", labels=legend_labels)
#         plt.setp(legend.get_title(), fontsize='12')

#     if title:
#         plt.title(title, fontsize=14)
#     plt.savefig(filename)



def visualize(h, color, title, filename, random_state=13):
    z = TSNE(n_components=2, random_state=random_state).fit_transform(h.detach().cpu().numpy())
    plt.figure(figsize=(5, 5))
    plt.xticks([])
    plt.yticks([])
    scatter = plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    
    # Automatically generate legend labels based on unique classes
    unique_classes = list(set(color.tolist()))
    legend_labels = [f"Class {label}" for label in unique_classes]
    
    # Add legend
    legend = plt.legend(handles=scatter.legend_elements()[0], title="Classes", labels=legend_labels)
    plt.setp(legend.get_title(), fontsize='12')

    if title:
        plt.title(title, fontsize=14)
    plt.savefig(filename)

print(f"Downloading the dataset...")
##========================= Download Dataset ====================##
dataset = Planetoid(root = 'data/',name=args.dataset,transform=NormalizeFeatures())
#dataset = Actor(root = 'data/',transform=NormalizeFeatures())
#dataset = WikipediaNetwork(root = 'data/',name=args.filename,transform=NormalizeFeatures())
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
data = dataset[0]  # Get the first graph object.
print("Done!")
print()
print(data)
dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]

print()
print(f"Selecting the LargestConnectedComponent..")
transform = LargestConnectedComponents()
data = transform(dataset[0])
print(data)
print()

graph, labels = get_graph_and_labels_from_pyg_dataset(data)
print("Calculating Different Informativeness Measures...")
nodelibef = li_node(graph, labels)
edgelibef = li_edge(graph, labels)
hadjbef = h_adj(graph, labels)
print(f'node label informativeness: {nodelibef:.2f}')
print(f'adjusted homophily: {hadjbef:.2f}')
print(f'edge label informativeness: {edgelibef:.2f}')
print("=============================================================")
print()

##=========================##=========================##=========================##=========================
max_iterations = args.max_iters
update_period = args.update_period
print("Preparing the graph for pruning...")
nxgraph = to_networkx(data, to_undirected=True)
beforegap, _, _, _ = spectral_gap(nxgraph)
print(f"InitialGap = {beforegap}")
print()

print("Binarizing the graph...")
nodes = list(graph.nodes())
labels = data.y.numpy()

# Randomly select two classes
class_labels = list(set(labels))
selected_classes = random.sample(class_labels, 2)

# Binarize the labels
binary_labels = torch.tensor([1 if label in selected_classes else 0 for label in labels])

# Create a new data object with binarized labels
data = data.clone()
data.y = binary_labels
data.num_classes = 2
print(data)


print("Visualizing the graph...")
pos = nx.kamada_kawai_layout(nxgraph)
node_colors = [data.y[node] for node in nxgraph.nodes]
nx.draw(nxgraph, pos=pos, with_labels=False, node_color=node_colors, cmap='Set2')
plt.savefig(f"Max_{dataset_name}_GraphBeforePruning.jpg")
print("Done!")



start_algo = time.time()
newgraph = process_and_update_edges(nxgraph, rank_by_proxy_delete, "proxydelete",max_iter=max_iterations,updating_period=update_period)
newgraph.remove_edges_from(list(nx.selfloop_edges(newgraph)))
end_algo = time.time()
aftergap,_, _, _ = spectral_gap(newgraph)
print()
print(f"FinalGap = {aftergap}")
data_pruning = (f" Deleting time = {end_algo - start_algo}")

pos = nx.kamada_kawai_layout(newgraph)
node_colors = [data.y[node] for node in newgraph.nodes]
nx.draw(newgraph, pos=pos, with_labels=False, node_color=node_colors, cmap='Set2')
plt.savefig(f"Max_{dataset_name}_GraphAfterPruning.jpg")



##=========================##=========================##=========================##=========================
newdata = from_networkx(newgraph)
data.edge_index = torch.cat([newdata.edge_index])
print()
print()
print(data)
print("Calculating informativeness measure after pruning...")

graphaf, labelsaf = get_graph_and_labels_from_pyg_dataset(data)
nodeliaf = li_node(graphaf, labelsaf)
edgeliaf = li_edge(graphaf, labelsaf)
hadjaf = h_adj(graphaf, labelsaf)
print(f'node label informativeness: {nodeliaf:.2f}')
print(f'adjusted homophily: {hadjaf:.2f}')
print(f'edge label informativeness: {edgeliaf:.2f}')
print("=============================================================")
print()




##========================= Split the dataset into train/test/val ====================##
print("Splitting datasets train/val/test...")
transform2 = RandomNodeSplit(split="train_rest",num_splits = 1, num_val=0.2, num_test=0.2)
data  = transform2(data)
print(data)
print()

##========================= Training/Testing Initialization ====================##
p = args.dropout ### Dropout
lr = args.LR
data = data.to(device)
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, data.num_classes)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=p, training=self.training)
        x = self.conv2(x, edge_index)
        return x

##========================= Training/Testing Loop ====================##
train_losses = []
nmi_values = []
model = GCN(hidden_channels = 32)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
print("Visualizing the node embeddings before training...")
model.eval()
out = model(data.x, data.edge_index)
visualize(out, color=data.y,title='Embeddings before training',filename=f"Min_{dataset_name}_BeforeTrainingEmb_{max_iterations}.jpg")


def train():
  model.train()
  optimizer.zero_grad()  # Clear gradients.
  out = model(data.x, data.edge_index)  # Perform a single forward pass.
  loss = criterion(out[data.train_mask], data.y[data.train_mask])
  loss.backward()  # Derive gradients.
  optimizer.step()  # Update parameters based on gradients.
  return loss


@torch.no_grad()
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability. 
    test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    nmiout = NMI(data.y[data.test_mask].cpu(), out[data.test_mask].max(1)[1].cpu())
    return nmiout,test_acc*100

patience = 100
best_loss = 1
nmi_at_best_loss = 0
acc_at_best_loss = 0
for epoch in range(1, 1001):
    train_loss = train()
    train_losses.append(train_loss.detach().numpy())
    nmi, acc = test()
    print(f"Epoch: {epoch:03d}, Loss: {train_loss:.4f}, NMI: {nmi:.3f}, TestAcc: {acc: .3f}")
    if train_loss < best_loss:
        best_loss = train_loss
        nmi_at_best_loss = nmi
        acc_at_best_loss = acc
        patience = 100
    else:
        patience -= 1
    if patience == 0:
        break
print(f"NMI: {nmi_at_best_loss:.3f}, TestAcc: {acc_at_best_loss:.1f}")
print("===============================================================")

fig, ax = plt.subplots(figsize=(8, 6))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.plot(train_losses, label='Training Loss')
plt.title('Training Loss Over Epochs')
ax.set_xlabel('Epochs', fontsize=24, fontdict={'family': 'serif'})
ax.set_ylabel('Loss', fontsize=24, fontdict={'family': 'serif'})
plt.legend()
plt.savefig(f"Min_{dataset_name}_Loss.png")

print("Visualizing node embeddings after training...")
model.eval()
out = model(data.x, data.edge_index)
visualize(out, color=data.y,title=f'After training - NMI: {nmi:.3f} ACC: {acc_at_best_loss:.1f}',filename = f'Min_{dataset_name}_Testsetemb_{max_iterations}.jpg')
print("Writing Metrics...")
headers = ['Dataset','EdgesModified','GapBefore','GapAfter','BeforeNodeLI','BeforeAdjHomophily','AfterNodeLI','AfterAdjHomophily','TrainingLoss','NMI','TestAcc']
with open(filename, mode='a', newline='') as file:
              writer = csv.writer(file)
              if file.tell() == 0:
                      writer.writerow(headers)
              writer.writerow([args.dataset,args.max_iters,beforegap,aftergap,nodelibef,hadjbef,nodeliaf,hadjaf,train_loss,nmi,acc_at_best_loss])
