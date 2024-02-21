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
from tqdm import tqdm
import networkx as nx
import argparse
from MinGapKupdates import *
from spectral_utils import *
import csv
from nodeli import *
from metrics import cluster_acc
import random
import numpy as np
random.seed(42)
np.random.seed(42)
torch.manual_seed(1234567)

parser = argparse.ArgumentParser(description='Run SBM')
parser.add_argument('--path', type=str, default='Dataset/SBM_30_0.3_0.03_1e-10.pt', help='path to directory containing npz file')
parser.add_argument('--update_period', type=int, default=1, help='Update Period of the Criterion')
parser.add_argument('--max_iters', type=int, default=10, help='Number of edges to add/delete')
parser.add_argument('--csvout', type=str, default='CommunityAlign/metrics.csv', help='CSV filename to record metrics')
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
nodelibef = li_node(graph, labels)
edgelibef = li_edge(graph, labels)
hadjbef = h_adj(graph, labels)
print(f'node label informativeness: {nodelibef:.2f}')
print(f'adjusted homophily: {hadjbef:.2f}')
print(f'edge label informativeness: {edgelibef:.2f}')
print("=============================================================")

print()
print("Pruning the graph for minimizing the spectral gap...")
SBMdatanx = to_networkx(SBMdata, to_undirected =True)
beforegap,_,_,_ = spectral_gap(SBMdatanx)
#proxykardeletegap,Gpr = minimize_and_update_edges(SBMdatanx)
Gpr  = process_and_update_edges(SBMdatanx, rank_by_proxy_delete, "proxydeleteSBM",max_iter=args.max_iters,updating_period=args.update_period)
Gpr.remove_edges_from(list(nx.selfloop_edges(Gpr)))
print()

pos = nx.kamada_kawai_layout(Gpr)
node_colors = [SBMdata.y[node] for node in Gpr.nodes]
nx.draw(Gpr, pos=pos, with_labels=False, node_color=node_colors, cmap='Set2')
plt.savefig(f"Plots/MinimizeGap/Min_{dataset_name}_GraphAfterPruning.jpg")

aftergap,_,_,_ = spectral_gap(Gpr)
print("Converting to PyG dataset")
newdata = from_networkx(Gpr)
newdata.x = SBMdata.x
newdata.y = SBMdata.y
newdata.num_classes = SBMdata.num_classes
print(newdata)


print("Calculating informativeness measure after pruning...")

graphaf, labelsaf = get_graph_and_labels_from_pyg_dataset(newdata)
nodeliaf = li_node(graphaf, labelsaf)
edgeliaf = li_edge(graphaf, labelsaf)
hadjaf = h_adj(graphaf, labelsaf)
print(f'node label informativeness: {nodeliaf:.2f}')
print(f'adjusted homophily: {hadjaf:.2f}')
print(f'edge label informativeness: {edgeliaf:.2f}')
print("=============================================================")
print()


print("Performing Train/Test splits...")
split = torch.randperm(SBMdata.num_nodes)
samples = int(0.6*len(split))
train_idx = split[:samples]
test_idx = split[samples:]
print(f"Number of training nodes - {len(train_idx)}")    
print(f"Number of testing nodes - {len(test_idx)}")   


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        #self.conv1 = GCNConv(newdata.num_features, hidden_channels)
        #self.conv2 = GCNConv(hidden_channels, newdata.num_classes)
        self.conv1 = GCNConv(newdata.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, newdata.num_classes)

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
out = model(newdata.x, newdata.edge_index)
visualize(out, color=newdata.y,title='Embeddings before training',filename=f"Plots/MinimizeGap/Min_{dataset_name}_EmbBefore.jpg")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
train_losses = []
nmi_values = []

def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(newdata.x, newdata.edge_index)  # Perform a single forward pass.

    # Use the CrossEntropyLoss directly without masking here
    loss = criterion(out[train_idx], newdata.y[train_idx])

    # Compute the gradients and update parameters
    loss.backward()
    optimizer.step()

    return loss.item()

def test():
    model.eval()
    clust = model(newdata.x, newdata.edge_index)
    return NMI(newdata.y[test_idx].cpu(), clust[test_idx].max(1)[1].cpu()),cluster_acc(newdata.y.cpu().numpy(), clust.max(1)[1].cpu().numpy())[0]

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
plt.title('Training Loss Over Epochs')
ax.set_xlabel('Epochs', fontsize=24, fontdict={'family': 'serif'})
ax.set_ylabel('Loss', fontsize=24, fontdict={'family': 'serif'})
plt.legend()
plt.savefig(f"Plots/MinimizeGap/Min_{dataset_name}_Loss250Min.png")

print("After training GCN, visualizing node embeddings...")
model.eval()
out = model(newdata.x, newdata.edge_index)
visualize(out[test_idx], color=newdata.y[test_idx],title=f'After training - NMI: {nmi:.3f} ACC: {acc_at_best_loss*100:.1f}',filename = f'Plots/MinimizeGap/Min_{dataset_name}_Testemb.jpg')
print("Writing Metrics...")
headers = ['Dataset','GapBefore','GapAfter','BeforeNodeLI','BeforeAdjHomophily','AfterNodeLI','AfterAdjHomophily','TrainingLoss','NMI','TestAcc']
with open(filename, mode='a', newline='') as file:
              writer = csv.writer(file)
              if file.tell() == 0:
                      writer.writerow(headers)
              writer.writerow([args.path,beforegap,aftergap,nodelibef,hadjbef,nodeliaf,hadjaf,train_loss,nmi,acc_at_best_loss])