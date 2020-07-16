import csv
import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
import pandas as pd
import statistics
import seaborn as sns

dataset21_path = r'Part_2_1\\dataset\\'

# open the bipartite graph from the file where each row represents an edge

g = nx.read_edgelist(dataset21_path + 'User_Item_BIPARTITE_GRAPH___UserID__ItemID.tsv',
                    delimiter = "\t")

# create a set (in order to avoid repetition) with the items

f = open(dataset21_path + 'User_Item_BIPARTITE_GRAPH___UserID__ItemID.tsv')
read = csv.reader(f, delimiter = '\t')
item = {r[1] for r in read} 
f.close()


f = open(dataset21_path + 'User_Item_BIPARTITE_GRAPH___UserID__ItemID.tsv')
all_rows = f.read().split('\n') # couple of nodes (source --> destination)


# graph = {'user_id' : [item_id, ..]}

graph = {}

for row in all_rows:
    row = row.split("\t") # remove tab separator from the nodes
    if row[0] not in graph: # if the first node is not a key of graph
        try:
            graph[row[0]] = [row[1]] # put that nodes as key (source) and row[1] as value (destination)
        except:
            pass
    else:
        graph[row[0]].append(row[1]) # else if it already a key, append another destination in the values of that key
    

# list of all items
item_list = sorted([int(x) for x in item])

# creating the projected item-item graph
# by default the weight is the number of shared neighbors
proj = bipartite.generic_weighted_projected_graph(g, item)

adj_matrix = nx.adjacency_matrix(proj)
adj_matrix = adj_matrix.todense()



# personalized is a nested dict 
# the key is the user_id, the values are dicts with the item_id (as key) and 1 (as values) if the item_id belongs to the topic, 0 otherwise

personalized = {}

for user in graph:
    adj_dict = {}
    for i in range(len(item_list)):
        
        
        if str(item_list[i]) in graph[user]:
        
            adj_dict.update({str(item_list[i]): 1})
            
        else:
            adj_dict.update({str(item_list[i]): 0})


    personalized.update({user: adj_dict})
            

# topic_open is a dict with user_id as key and a list of tuples (item_id, rank) as values

topic_open = {}

for topic in personalized:
    
    topic_open[topic] = nx.pagerank(proj,  0.1, personalized[topic])
    
# removing links already visited by each user from the dict

for topic in topic_open:
    for item in topic_open[topic]:
        if item in graph[topic]:            
            topic_open[topic][item] = 0

# ordering by (descending) rank

for topic in topic_open:
    topic_open[topic] = sorted(topic_open[topic].items(), key=lambda x: x[1], reverse = True)


# opening ground truth file as dict
gt_dict = pd.read_csv(dataset21_path + "Ground_Truth___UserID__ItemID.tsv", sep = "\t", header = None)

gt_dict.rename(columns = {0: "user_id", 1: "item_id"}, inplace = True)
gt_dict = gt_dict.groupby("user_id")["item_id"].apply(list).to_dict()
gt_dict = dict((str(k), [str(x) for x in v]) for k,v in gt_dict.items())


# R precision su topic_open

precision_dict = {}

for topic in gt_dict:
    count = 0
    precision_dict[topic] = {}

    for rank in range(len(gt_dict[topic])):

        if str(topic_open[topic][rank][0]) in gt_dict[topic]:
            count += 1
            
    precision_dict[topic] = round(count/len(gt_dict[topic]), 3)

# average R_precision

R_precision = statistics.mean(list(precision_dict.values()))


# distribution plot of the R-PRECISION values

#plt.figure(figsize = (15, 8))
#sns.set_style("whitegrid", {'axes.grid' : False})

#sns.distplot(list(precision_dict.values()), color="r", kde=False).set_title("Distribution of R Precision")

#plt.xlabel("R precision", fontsize=15)
#plt.ylabel("Frequency", fontsize=15)
