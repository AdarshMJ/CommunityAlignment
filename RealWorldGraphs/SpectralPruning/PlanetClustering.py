from torch_geometric.data import Data
from torch_geometric.utils import to_undirected,to_networkx,from_networkx
from torch_geometric.datasets import TUDataset, Planetoid,WebKB,Actor, WikipediaNetwork,Coauthor
from torch_geometric.transforms import NormalizeFeatures, LargestConnectedComponents
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
import pickle
import networkx as nx
from spectralclustering import *
from fastrewiringKupdates import *
from MinGapKupdates import *
from spectral_utils import *
import matplotlib.pyplot as plt
import csv
import os
import warnings
warnings.filterwarnings('ignore')
import argparse

parser = argparse.ArgumentParser(description='Run SpectralClustering')
parser.add_argument('--method', type=str, help='Max/Min')
parser.add_argument('--dataset', type=str, help='Dataset to download')
parser.add_argument('--max_iter', type=int, help='Number of edges to modify')
parser.add_argument('--csv_file', type=str, help='Out file')
args = parser.parse_args()




max_iter = args.max_iter

pickle_file = f"/content/drive/MyDrive/ICLR2024/SBMExpts/CommunityAlign/PrunedGraphs/AddedGraphs/{args.dataset}_{args.max_iter}_{args.method}Add.pickle"
graph_file = f'{args.dataset}.jpg'
csv_file =args.csv_file



### Download the dataset###
dataset = Planetoid(root = 'data/',name=args.dataset,transform=NormalizeFeatures())
#dataset = WebKB(root = 'data/',name=args.dataset,transform=NormalizeFeatures())
#dataset = Coauthor(root = 'data/',name = args.dataset,transform=NormalizeFeatures())
data = dataset[0]  # Get the first graph object.
print()
transform = LargestConnectedComponents()
data = transform(dataset[0])
print(data)


if os.path.exists(pickle_file):
    print("Pickle file already exists. Loading the graph...")
    with open(pickle_file, "rb") as f:
        newgraph = pickle.load(f)
        gapafter,_,_,_ = spectral_gap(newgraph)
        print(f"Spectral Gap after pruning = {gapafter}")
        print("Performing spectral clustering...")
        clusters,labels = k_way_spectral(newgraph, dataset.num_classes)
        clustermod = maximize_modularity(newgraph)
        cluster_dict = {node: i for i, cluster in enumerate(clustermod) for node in cluster}
        cluster_list = [cluster_dict[node] for node in range(len(data.y))]
        ground_truth_labels = data.y.cpu()
        nmi_score_spectral = NMI(labels,ground_truth_labels)
        #mod_score_spectral = nx.community.modularity(newgraph,clusters)
        print("NMI Score from Spectral Pruning :", nmi_score_spectral)
        #print("Mod Score from Spectral Pruning :", mod_score_spectral)

        nmi_score_mod = NMI(cluster_list,ground_truth_labels)
        #mod_score_mod = nx.community.modularity(newgraph,clustermod)
        print("NMI Score from Modularity Maximization :", nmi_score_mod)
        #print("Mod Score from Modularity Maximization :", mod_score_mod)
else:
    print("Pickle file does not exist. Processing the edges...")
    nxgraph = to_networkx(data, to_undirected=True)
    if os.path.exists(graph_file):
      print("Not visualizing graph. It exists...")
    else:
          print("Visualizing the graph...")
          #pos = nx.kamada_kawai_layout(nxgraph)
          #node_colors = [data.y[node] for node in nxgraph.nodes]
          #nx.draw(nxgraph, pos=pos, with_labels=False, node_color=node_colors, cmap='Set2')
          #plt.savefig(f"{args.dataset}.jpg")

    print("Done!")
    gapbefore,_,_,_ = spectral_gap(nxgraph)
    print(f"Spectral Gap before pruning = {gapbefore}")
    
    if args.method == "Max":
          print("Pruning edges to maximize spectral gap...")
          newgraph = process_and_update_edges(nxgraph, rank_by_proxy_add, "proxyaddmax", max_iter=max_iter,updating_period=1)
          newgraph.remove_edges_from(list(nx.selfloop_edges(newgraph)))
          gapafter,_,_,_ = spectral_gap(newgraph)
          print(newgraph)
          print(f"Spectral Gap after pruning = {gapafter}")
          print("Saving the graph...")
          with open(pickle_file, "wb") as f:
            pickle.dump(newgraph, f)
          print("Done!")
          print("Performing spectral clustering...")
          clusters,labels = k_way_spectral(newgraph, dataset.num_classes)
          clustermod = maximize_modularity(newgraph)
          cluster_dict = {node: i for i, cluster in enumerate(clustermod) for node in cluster}
          cluster_list = [cluster_dict[node] for node in range(len(data.y))]
          ground_truth_labels = data.y.cpu()
          nmi_score_spectral = NMI(labels,ground_truth_labels)
          #mod_score_spectral = nx.community.modularity(newgraph,clusters)
          print("NMI Score from Spectral Pruning :", nmi_score_spectral)
          #print("Mod Score from Spectral Pruning :", mod_score_spectral)

          nmi_score_mod = NMI(cluster_list,ground_truth_labels)
          #mod_score_mod = nx.community.modularity(newgraph,clustermod)
          print("NMI Score from Modularity Maximization :", nmi_score_mod)
          #print("Mod Score from Modularity Maximization :", mod_score_mod)
      
        

    elif args.method == 'Min':
          print("Pruning edges to minimize spectral gap...")
          newgraph = min_and_update_edges(nxgraph, rank_by_proxy_add_min, "proxyaddmin", max_iter=max_iter)
          newgraph.remove_edges_from(list(nx.selfloop_edges(newgraph)))
          gapafter,_,_,_ = spectral_gap(newgraph)
          print(f"Spectral Gap after pruning = {gapafter}")
          print(newgraph)
          print("Saving the graph...")
          with open(pickle_file, "wb") as f:
            pickle.dump(newgraph, f)
          print("Done!")
          print("Performing spectral clustering...")
          clusters,labels = k_way_spectral(newgraph, dataset.num_classes)
          clustermod = maximize_modularity(newgraph)
          cluster_dict = {node: i for i, cluster in enumerate(clustermod) for node in cluster}
          cluster_list = [cluster_dict[node] for node in range(len(data.y))]
          ground_truth_labels = data.y.cpu()
          nmi_score_spectral = NMI(labels,ground_truth_labels)
          #mod_score_spectral = nx.community.modularity(newgraph,clusters)
          print("NMI Score from Spectral Pruning :", nmi_score_spectral)
          #print("Mod Score from Spectral Pruning :", mod_score_spectral)

          nmi_score_mod = NMI(cluster_list,ground_truth_labels)
          #mod_score_mod = nx.community.modularity(newgraph,clustermod)
          print("NMI Score from Modularity Maximization :", nmi_score_mod)
          #print("Mod Score from Modularity Maximization :", mod_score_mod)


    else :
      print("Invalid Method...")


#headers = ['Method','Dataset','EdgesModified','GapAfter','NMISpectral','NMIMod','ModSpectral','ModMod']
headers = ['Method','Dataset','EdgesModified','GapAfter','NMISpectral','NMIMod']
with open(csv_file, mode='a', newline='') as file:
              writer = csv.writer(file)
              if file.tell() == 0:
                      writer.writerow(headers)
              #writer.writerow([args.method,args.dataset,args.max_iter,gapafter,nmi_score_spectral,nmi_score_mod,mod_score_spectral,mod_score_mod])
              writer.writerow([args.method,args.dataset,args.max_iter,gapafter,nmi_score_spectral,nmi_score_mod])
