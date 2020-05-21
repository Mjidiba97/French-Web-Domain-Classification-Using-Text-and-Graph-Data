import csv
import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression

# Read training data
with open("train_labels.csv", 'r') as f:
    train_data = f.read().splitlines()

train_hosts = list()
y_train = list()
for row in train_data:
    host, label = row.split(",")
    train_hosts.append(host)
    y_train.append(label.lower())

# Read test data
with open("test_labels.csv", 'r') as f:
    test_hosts = f.read().splitlines()

# Create a directed, weighted graph
G = nx.read_weighted_edgelist('graph_data.txt', create_using=nx.DiGraph())

print(G.number_of_nodes())
print(G.number_of_edges())

# Create the training matrix. Each row corresponds to a web host.
# Use the following 3 features for each web host (unweighted degrees)
# (1) out-degree of node
# (2) in-degree of node
# (3) average degree of neighborhood of node
X_train = np.zeros((len(train_hosts), 3))
avg_neig_deg = nx.average_neighbor_degree(G, nodes=train_hosts)
for i in range(len(train_hosts)):
    X_train[i,0] = G.in_degree(train_hosts[i])
    X_train[i,1] = G.out_degree(train_hosts[i])
    X_train[i,2] = avg_neig_deg[train_hosts[i]]

# Create the test matrix. Use the same 3 features as above
X_test = np.zeros((len(test_hosts), 3))
avg_neig_deg = nx.average_neighbor_degree(G, nodes=test_hosts)
for i in range(len(test_hosts)):
    X_test[i,0] = G.in_degree(test_hosts[i])
    X_test[i,1] = G.out_degree(test_hosts[i])
    X_test[i,2] = avg_neig_deg[test_hosts[i]]

print("Train matrix dimensionality: ", X_train.shape)
print("Test matrix dimensionality: ", X_test.shape)

# Use logistic regression to classify the webpages of the test set
clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=2000)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)

# Write predictions to a file
with open('graph_baseline.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    lst = clf.classes_.tolist()
    lst.insert(0, "Host")
    writer.writerow(lst)
    for i,test_host in enumerate(test_hosts):
        lst = y_pred[i,:].tolist()
        lst.insert(0, test_host)
        writer.writerow(lst)
