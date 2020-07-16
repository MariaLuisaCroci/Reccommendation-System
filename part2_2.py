import csv
import pandas as pd
import statistics

dataset22_path = 'Part_2_2\\dataset\\'

# graph_dict = {user_id : [item1, item2, ...]}

graph_dict = pd.read_csv(dataset22_path + 'Base_Set___UserID__ItemID__PART_2_2.tsv',
                      sep = "\t", header = None)
graph_dict.rename(columns = {0: "user_id", 1: "item_id"}, inplace = True)
graph_dict = graph_dict.groupby("user_id")["item_id"].apply(list).to_dict()

# rank_dict = {item : [(item1, prob1), (item2, prob2), ...]}

f = open(dataset22_path + 'ItemID__PersonalizedPageRank_Vector.tsv')
reader = csv.reader(f, delimiter = '\t')
rank_dict = {int(rows[0]) : eval(rows[1]) for rows in reader} # list of tuples
f.close()


row_list = list(rank_dict.keys())

# from tuples to dict
for item in rank_dict:
    rank_dict[item] = dict(rank_dict[item])


personalized_pr =  {}

for user in graph_dict:
    denom = len(graph_dict[user])
    # creating a subdict with keys from row_list and 0 as initial value
    q = dict.fromkeys(row_list, 0)
    for item_id in graph_dict[user]:
        for col in rank_dict[item_id]:
            q[col] += (rank_dict[item_id][col]/ denom)
    personalized_pr.update({user : q})

# opening the ground truth file as dict
gt2_dict = pd.read_csv(dataset22_path + 'Ground_Truth___UserID__ItemID__PART_2_2.tsv', 
                       sep = "\t", header = None)
gt2_dict.rename(columns = {0: "user_id", 1: "item_id"}, inplace = True)
gt2_dict = gt2_dict.groupby("user_id")["item_id"].apply(list).to_dict()


# removing the links already visited by the user from the dict 

for topic in personalized_pr:
    for item in personalized_pr[topic]:
        if item in graph_dict[topic]:            
            personalized_pr[topic][item] = 0


# ordering by (descending) rank

for topic in personalized_pr:
    personalized_pr[topic] = sorted(personalized_pr[topic].items(), key=lambda x: x[1], reverse = True)


# R precision su personalized_pr

precision2_dict = {}

for topic in gt2_dict:
    count = 0
    precision2_dict[topic] = {}

    for rank in range(len(gt2_dict[topic])):

        if str(personalized_pr[topic][rank][0]) in gt2_dict[topic]:
            count += 1
            
    precision2_dict[topic] = round(count/len(gt2_dict[topic]), 3)

# average R 

R_precision = statistics.mean(list(precision2_dict.values()))