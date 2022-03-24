import math
import random
import numpy as np

def euclid(p,q):
    x = p[0]-q[0]
    y = p[1]-q[1]
    return math.sqrt(x*x+y*y)
                
class Graph:

    # Complete as described in the specification, taking care of two cases:
    # the -1 case, where we read points in the Euclidean plane, and
    # the n>0 case, where we read a general graph in a different format.
    # self.perm, self.dists, self.n are the key variables to be set up.
    def __init__(self,n,filename):
        reader = open(filename,'r')
        line = reader.readline().rstrip('\n')
        nodes = []
        if int(n) == -1: #Euclidean File
            # for each line get coordinates, and calculate euclidean distances between all points and add to dist
            while True:
                if len(line) == 0:
                    break
                a = line.rstrip('\n')
                coordinate_string = None
                coordinate_string_array = []
                line_as_list = list(a)
                for i in range(len(line_as_list)):
                    if line_as_list[i] != ' ' and coordinate_string == None:
                        coordinate_string = line_as_list[i]
                    elif line_as_list[i] != ' ' and coordinate_string != None:
                        coordinate_string = coordinate_string + line[i]
                    elif line_as_list[i] == ' ' and coordinate_string != None:
                        coordinate_string_array.append(int(coordinate_string))
                        coordinate_string = None
                    if i == len(line_as_list) -1 and coordinate_string != None:
                        coordinate_string_array.append(int(coordinate_string))
                        coordinate_string = None
                nodes.append(coordinate_string_array)
                line = reader.readline()
            self.n = len(nodes)
            # Initialize Distances
            self.dists = np.zeros((self.n,self.n))
            # Initialize Permutation
            self.perm = [i for i in range(self.n)]
            self.perm.append(0)
            # Add Euclidean distances for each pair i j in dists
            for i in range(self.n):
                for j in range(self.n):
                    self.dists[i][j] = euclid(nodes[i],nodes[j])
                    self.dists[j][i] = euclid(nodes[i],nodes[j])
        else: # General File
            # For each line get three components, and add cost from i to j and j to i to dist
            self.n = int(n)
            while True:
                if len(line) == 0:
                    break
                a = line.rstrip('\n')
                nodes.append(a.split(' '))
                line = reader.readline()
            # Initialize Distances
            self.dists = np.zeros((self.n,self.n))
            # Initialize Permutation
            self.perm = [i for i in range(self.n)]
            self.perm.append(0)
            for i in range(len(nodes)):
                node1 = int(nodes[i][0])
                node2 = int(nodes[i][1])
                cost = int(nodes[i][2])
                self.dists[node1][node2] = cost
                self.dists[node2][node1] = cost
        reader.close()
        # dist now contains the costs dist[i][j] to get from node i to node j
        # perm is initialzed to perm[i] = i
        # n is now number of nodes
        

    # Complete as described in the spec, to calculate the cost of the
    # current tour (as represented by self.perm).
    def tourValue(self):
        # Iterate through current permutation and add up the costs of each step to find the total cost of the tour
        cost = 0
        for i in range(len(self.perm)):
            if (i == len(self.perm)-1):
                break
            else:
                node1 = self.perm[i]
                node2 = self.perm[i+1]
                cost += self.dists[node1][node2]
        return cost
    
    # Attempt the swap of cities i and i+1 in self.perm and commit
    # commit to the swap if it improves the cost of the tour.
    # Return True/False depending on success.
    def trySwap(self,i):
        copy = self.perm.copy()
        current_cost = self.tourValue()
        self.perm[i] = copy[(i+1) % self.n]
        self.perm[(i+1) % self.n] = copy[i]
        new_cost = self.tourValue()
        if new_cost < current_cost:
            # Swapped and update the start/end points
            self.perm[-1] = self.perm[0]
            return True
        else:
            # Revert to original permutation
            self.perm = copy[:]
            return False


    # Consider the effect of reversiing the segment between
    # self.perm[i] and self.perm[j], and commit to the reversal
    # if it improves the tour value.
    # Return True/False depending on success.              
    def tryReverse(self,i,j):
        copy = self.perm.copy()
        current_cost = self.tourValue()
        for _ in range(j-i+1):
            if i == 0:
                break
            self.perm[i] = copy[j]
            i +=1
            j -=1
        new_cost = self.tourValue()
        if new_cost < current_cost:
            # Reversed and update the start/end points
            self.perm[-1] = self.perm[0]
            return True
        else:
            # Revert to original permutation
            self.perm = copy[:]
            return False
        
    def swapHeuristic(self,k):
        better = True
        count = 0
        while better and (count < k or k == -1):
            better = False
            count += 1
            for i in range(self.n):
                if self.trySwap(i):
                    better = True

    def TwoOptHeuristic(self,k):
        better = True
        count = 0
        while better and (count < k or k == -1):
            better = False
            count += 1
            for j in range(self.n-1):
                for i in range(j):
                    if self.tryReverse(i,j):
                        better = True

    # Implement the Greedy heuristic which builds a tour starting
    # from node 0, taking the closest (unused) node as 'next'
    # each time.
    def Greedy(self):
        if self.n == 0:
            return
        self.perm[0] = 0
        startingNode = 0
        nextNode = startingNode
        visited = [startingNode]
        for i in range(1,self.n):
            # Use enumerate to get cost and node numbers
            neighbours = [(val,node) for (node,val) in enumerate(self.dists[nextNode]) if node not in visited ]
            # As mentioned in specs, take minimum cost neighbor as next node
            minCost,nextNode = min(neighbours)
            self.perm[i] = nextNode
            visited.append(nextNode)

    # Custom Hierarchical Clustering Algorithm
    def HierarchicalClustering(self):
        c = []
        hyperparameters = []
        for a in [0, 0.1, 0.5]:
            for b in [0, 0.1, 0.5]:
                for g in [0, 0.1, 0.5]:
                    for d in [0, 0.1, 0.5]:
                        for e in [0, 0.1, 0.5]:
                            p = [a,b,g,d,e]
                            A = self.HierarchicalClusteringGivenParameters(p)
                            self.perm = self.getPerm(A)
                            c.append(self.tourValue())
                            hyperparameters.append(p)
        index = np.argmax(c)
        A = self.HierarchicalClusteringGivenParameters(hyperparameters[index])
        self.perm = self.getPerm(A)
        show_graph(A)
    
    def HierarchicalClusteringGivenParameters(self, hyperparameters):
        # mean distance, standard deviation, and first priority power function
        mu = np.zeros(self.n)
        s = np.zeros(self.n)
        p = np.zeros(self.n)
        # Degrees of each node (number of connected neighbors)
        deg = np.zeros(self.n)
        # Hyperparameters
        alpha = hyperparameters[0]
        beta = hyperparameters[1]
        gamma = hyperparameters[2]
        delta = hyperparameters[3]
        epsilon = hyperparameters[4]
        # Initialize mean, sd, and priority
        for i in range(0,self.n):
            mu[i] = np.mean(np.delete(self.dists[i], i))
            s[i] = np.std(np.delete(self.dists[i], i))
            p[i] = (mu[i]**(alpha))*(s[i]**(beta))
        # A: Adjacency matrix to record connections
        # c2: Second priority power functions
        A = np.zeros((self.n, self.n))
        c2 = np.zeros((self.n, self.n))
        # jj: List of cities in decreasing order of priority
        jj = np.flip(np.argsort(p))
    
        for i in jj:
            # If no connections yet (always the case initially)
            if deg[i] == 0:
                # Calculate second power function, excluding diagonal distances (i,i)
                temp = (mu[i]**(delta))*(s[i]**(epsilon))/(np.delete(self.dists[i], i)**(gamma))
                c2[i] = np.insert(temp, i, 0)
                # If already has 2 connections set priority to 0
                for l in range(self.n):
                    if deg[l]>=2:
                        c2[i, l] = 0
                # Get highest priority neighbor
                j = np.argmax(c2[i])
                # If neighber also doesn't already have 2 connections
                if (deg[j] < 2) & (j != i):
                    # Add connection to adjacency matrix
                    A[i,j] = 1
                    A[j,i] = 1
                    # Update each node's degree (num of connected neighbors)
                    deg[i] += 1
                    deg[j] += 1
        
        # ---------------------------- Second Main Step ---------------------------- #
        
        # Reset second priority function matrix
        c2 = np.zeros((self.n, self.n))
        for i in jj:
            # For nodes that still only have one connected neighbor
            if deg[i] == 1:
                # Calculate second power function, excluding diagonal distances (i,i)
                temp = (mu[i]**(delta))*(s[i]**(epsilon))/(np.delete(self.dists[i], i)**(gamma))
                c2[i] = np.insert(temp, i, 0)
                # If already has 2 connections set priority to 0
                for l in range(self.n):
                    if deg[l]>=2:
                        c2[i, l] = 0
                
                for l in range(self.n):
                    if A[i,l] == 1:
                        c2[i,l] = 0
                    if deg[l] > 1:
                        c2[i,l] = 0
                # Highest priority node
                highc2 = np.flip(np.argsort(c2[i]))
                k = 0
                # Check that adding this node will not result in a cycle
                while (k < self.n-1) & (self.isCycle(i, highc2[k], A)):
                    k += 1
                # j: highest priority city with degree < 2 guaranteed to not create a cycle
                j = highc2[k]
                # If neighber also doesn't already have 2 connections
                if deg[j] <=1 & (j != i):
                    # Add connection to adjacency matrix
                    A[i,j] = 1
                    A[j,i] = 1
                    # Update each node's degree (num of connected neighbors)
                    deg[i] += 1
                    deg[j] += 1
        
        return A

    def isCycle(self, i, j, a):
        # Check whether adding connection (i,j) will result in a cycle
        A = a.copy()
        # Add connections to copy
        A[i,j] = 1
        A[j,i] = 1
        dp = np.array([i,j])
        size = len(dp)
        noCycle = True
        while (sum(A[dp[size-1]]) > 1) & (size<len(A)) & (noCycle):
            for l in range(len(A)):
                if (A[dp[size-1]][l] > 0) & (l != dp[size-2]):
                    nextNode = l
            # Add to direct path (dp)
            dp = np.append(dp,nextNode)
            size = len(dp)
            noCycle = (dp[size-1]!=dp[0])
        return(not noCycle)


    def getPerm(self, A):
        # Return tour from adjacency matrix
        for l in range(len(A)):
            if (A[0][l] > 0):
                nextNode = l
        dp = np.array([0,nextNode])
        size = len(dp)
        while (sum(A[dp[size-1]]) > 1) & (size<len(A)):
            for l in range(len(A)):
                if (A[dp[size-1]][l] > 0) & (l != dp[size-2]):
                    nextNode = l
            # Add to direct path (dp)
            dp = np.append(dp,nextNode)
            size = len(dp)
        dp = np.append(dp, 0)
        # Return direct path found in adjacency matrix
        return(dp)

import matplotlib.pyplot as plt
import networkx as nx
def show_graph(adjacency_matrix):
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=500)
    plt.show()


    


'''g=Graph(12,"twelvenodes")
g.tourValue()
g.swapHeuristic(12)
g.tourValue()'''