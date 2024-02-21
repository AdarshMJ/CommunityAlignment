import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score as NMI
import torch
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected,to_networkx,from_networkx
from torch_geometric.transforms import RandomNodeSplit
from sklearn.model_selection import train_test_split
from metrics import cluster_acc
import networkx as nx
from nodeli import *
from tqdm import tqdm
import argparse
import random
import numpy as np
from spectral_utils import *
import csv
random.seed(42)
np.random.seed(42)
torch.manual_seed(1234567)

parser = argparse.ArgumentParser(description='Run SBM')
parser.add_argument('--path', type=str, default='Dataset/', help='path to directory containing npz file')
parser.add_argument('--csvout', type=str, default='CommunityAlign/metrics.csv', help='CSV filename to record metrics')
parser.add_argument('--SBMPlotName', type=str, default='OriginalSBM', help='Saving the SBM plot')
parser.add_argument('--EmbeddingPlotBefore', type=str, default='OriginalPlotBefore', help='Saving the node embeddings before training')
parser.add_argument('--EmbeddingPlotAfter', type=str, default='OriginalPlotAfter', help='Saving the node embeddings after training')
args = parser.parse_args()

dataset_name = os.path.splitext(os.path.basename(args.path))[0]

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



def get_graph_and_labels_from_pyg_dataset(dataset):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(dataset.x)))
    graph.add_edges_from(dataset.edge_index.T.numpy())

    labels = dataset.y.numpy()

    return graph, labels


filename = args.csvout   


dataset,_ = torch.load(args.path)
x = dataset['x']
edge_index = dataset['edge_index']
y = dataset['y']
SBMdata = Data(x=x,edge_index=edge_index,y=y,num_classes=2)
print(SBMdata)

graph, labels = get_graph_and_labels_from_pyg_dataset(SBMdata)
print("Calculating Different Informativeness Measures...")
nodeli = li_node(graph, labels)
edgeli = li_edge(graph, labels)
hadj = h_adj(graph, labels)
print(f'node label informativeness: {nodeli:.2f}')
print(f'adjusted homophily: {hadj:.2f}')
print(f'edge label informativeness: {edgeli:.2f}')
print("=============================================================")
print()

print("Performing Train/Test splits...")
split = torch.randperm(SBMdata.num_nodes)
samples = int(0.8*len(split))
train_idx = split[:samples]
test_idx = split[samples:]
print(f"Number of training nodes - {len(train_idx)}")    
print(f"Number of testing nodes - {len(test_idx)}")     

gsbm = to_networkx(SBMdata,to_undirected=True)
print(gsbm)
gap_before,_,_,_ = spectral_gap(gsbm)
print(f"Spectral Gap = {gap_before}")
pos = nx.kamada_kawai_layout(gsbm)
node_colors = [SBMdata.y[node] for node in gsbm.nodes]
nx.draw(gsbm, pos=pos, with_labels=False, node_color=node_colors, cmap="Set2")
plt.savefig(f"Plots/Original/Original_{dataset_name}_SBM.jpg")



class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        #self.conv1 = GCNConv(newdata.num_features, hidden_channels)
        #self.conv2 = GCNConv(hidden_channels, newdata.num_classes)
        self.conv1 = GCNConv(SBMdata.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, SBMdata.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GCN(hidden_channels=512)
print(model)



print("Visualizing the node embeddings before training...")
model.eval()
out = model(SBMdata.x, SBMdata.edge_index)
visualize(out, color=SBMdata.y,title='Embeddings before training',filename=f"Plots/Original/{dataset_name}_BeforeTraining.jpg")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()


train_losses = []
nmi_values = []
def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(SBMdata.x, SBMdata.edge_index)  # Perform a single forward pass.

    # Use the CrossEntropyLoss directly without masking here
    loss = criterion(out[train_idx], SBMdata.y[train_idx])

    # Compute the gradients and update parameters
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def test():
    model.eval()
    clust = model(SBMdata.x, SBMdata.edge_index)
    return NMI(SBMdata.y[test_idx].cpu(), clust[test_idx].max(1)[1].cpu()),cluster_acc(SBMdata.y.cpu().numpy(), clust.max(1)[1].cpu().numpy())[0]

patience = 50
best_loss = 1
nmi_at_best_loss = 0
acc_at_best_loss = 0
for epoch in range(1, 1001):
    train_loss = train()
    train_losses.append(train_loss)
    nmi, acc = test()
    print(f"Epoch: {epoch:03d}, Loss: {train_loss:.4f}, NMI: {nmi:.3f}, ACC: {acc*100: .3f}")
    if train_loss < best_loss:
        best_loss = train_loss
        nmi_at_best_loss = nmi
        acc_at_best_loss = acc
        patience = 50
    else:
        patience -= 1
    if patience == 0:
        break
print(f"NMI: {nmi_at_best_loss:.3f}, ACC: {acc_at_best_loss*100:.1f}")
print("===============================================================")

fig, ax = plt.subplots(figsize=(8, 6))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.plot(train_losses, label='Training Loss')
ax.set_xlabel('Epochs', fontsize=24, fontdict={'family': 'serif'})
ax.set_ylabel('Loss', fontsize=24, fontdict={'family': 'serif'})
plt.legend()
plt.savefig(f"Plots/Original/{dataset_name}_Lossoriginal.png")

print("After training GCN, visualizing node embeddings...")
model.eval()
out = model(SBMdata.x, SBMdata.edge_index)
visualize(out[test_idx], color=SBMdata.y[test_idx],title=f'After training - NMI: {nmi:.3f} ACC: {acc_at_best_loss*100:.1f}',filename = f'Plots/Original/{dataset_name}_Testsetemb.jpg')
print("Writing Metrics...")
headers = ['Dataset','SpectralGap','NodeLI','EdgeLI','AdjHomophily','TrainingLoss','NMI','TestAcc']
with open(filename, mode='a', newline='') as file:
              writer = csv.writer(file)
              if file.tell() == 0:
                      writer.writerow(headers)
              writer.writerow([args.path,gap_before,nodeli,edgeli,hadj,train_loss,nmi,acc_at_best_loss])