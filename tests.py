import math
import graph
import random

def randomEuclidean(max_value,num_of_nodes):
    result = []
    for i in range(num_of_nodes):
        x = random.randint(0,max_value)
        y = random.randint(0,max_value)
        result.append([x,y])
    outputToFile(result,"random_euclidean", max_value)

def randomMetric(max_value,num_of_nodes):
    result = [[None for _ in range(num_of_nodes)] for _ in range(num_of_nodes)]
    for i in range(num_of_nodes):
        for j in range(num_of_nodes):
            if i == j:
                result[i][j] = 0
                continue
            max = []
            for z in range(num_of_nodes):
                if result[i][z] == None:
                    value = random.randint(1,max_value)
                    result[i][z] = value
                    result[z][i] = value
                if result[z][j] == None:
                    value = random.randint(1, max_value)
                    result[z][j] = value
                    result[j][z] = value
                max.append(result[i][z] + result[z][j])
            value = random.randint(1,min(max))
            result[i][j] = value
            result[j][i] = value
    outputToFile(result, "random_metric", max_value)

def randomNonMetric(max_value,num_of_nodes):
    result = [[None for _ in range(num_of_nodes)] for _ in range(num_of_nodes)]
    for i in range(num_of_nodes):
        for j in range(num_of_nodes):
            if i == j:
                result[i][j] = 0
                continue
            result[i][j] = random.randint(1,max_value)
    outputToFile(result, "random_nonmetric", max_value)

def euclideanStraightLine(max_value,num_of_nodes):
    result = []
    for i in range(num_of_nodes+1):
        x = 0
        y = i
        result.append([x,y])
    random.shuffle(result)
    outputToFile(result,"eucliden_straightline", max_value)

def euclideanSquare(max_value,num_of_nodes):
    result = []
    for i in range(num_of_nodes+1):
        x = 0
        y = i
        result.append([x, y])
    for i in range(num_of_nodes+1):
        x = i
        y = 0
        result.append([x, y])
    for i in range(num_of_nodes+1):
        x = num_of_nodes
        y = i
        result.append([x, y])
    for i in range(num_of_nodes+1):
        x = i
        y = num_of_nodes
        result.append([x, y])
    random.shuffle(result)
    outputToFile(result, "eucliden_square", max_value)

#save graph to file
def outputToFile(result, file_name, max_value):
    if len(result[0]) == 2: #is euclidean
        final = ""
        f = open(file_name, "w")
        for r in result:
            final = final+(" ")
            x_count = len(str(r[0]))
            while x_count != len(str(max_value)):
                final = final + (" ")
                x_count += 1
            final = final + (str(r[0]))
            final = final + ("  ")
            y_count = len(str(r[1]))
            while y_count != len(str(max_value)):
                final = final + (" ")
                y_count += 1
            final= final + str(r[1])
            final = final + ("\n")
        f.write(final)
        f.close()
    else: #is metric/no metric
        final = ""
        f = open(file_name, "w")
        for i in range(len(result)):
            for j in range(len(result)):
                if i == j:
                    continue
                if j > i:
                    final = final + str(i)
                    final = final + " "
                    final = final + str(j)
                    final = final + " "
                    final = final + str(result[i][j])
                    final = final + "\n"
        f.write(final)
        f.close()

#finding the excat solution by listing all possible path and calculate their value
def exact_solution(graph):
    nodes = graph.n
    all_possile_path = []
    for i in range(nodes):
        search_rec([i],nodes, all_possile_path)
    values = []
    for path in all_possile_path:
        values.append(tour_value(path,graph))
    return min(values)

def tour_value(input_path,graph):
    final_path = input_path[:]
    final_path.append(input_path[0])
    value = 0
    for i in range(len(final_path)):
        if i == len(final_path) - 1:
            break
        node_current = final_path[i]
        node_next = final_path[i + 1]
        value += graph.dists[node_current][node_next]
    return value

def search_rec(array,n,result):
    if len(array) == n:
        result.append(array)
    for i in range(n):
        if i not in array:
            new = array[:]
            new.append(i)
            search_rec (new,n,result)

#Main method for testing and listing result
def testing(fileName,node_num,best_result = None):
    g = graph.Graph(node_num,fileName)
    g.swapHeuristic(g.n)
    swap = g.tourValue()
    g = graph.Graph(node_num, fileName)
    g.TwoOptHeuristic(g.n)
    two_op = g.tourValue()
    g = graph.Graph(node_num, fileName)
    g.Greedy()
    greedy = g.tourValue()
    g = graph.Graph(node_num, fileName)
    g.HierarchicalClustering()
    cluster = g.tourValue()
    if best_result == None:
        best_result = exact_solution(g)
    print("FileName:{0}. Swap Result:{1}. Two_Op Result:{2}. Greedy Result:{3}. Cluster Result:{4}. Best Result:{5}".
          format(fileName,swap,two_op,greedy,cluster,best_result))#Add time

    
#generate graphs
randomEuclidean(11,6)
randomMetric(11,6)
randomNonMetric(11,6)
euclideanStraightLine(70,70)
euclideanSquare(100,25)

#testing
testing("sixnodes",6)
testing("twelvenodes",12,-1)
testing("cities50",-1,-1)
testing("random_euclidean",-1)
testing("random_metric",6)
testing("random_nonmetric",6)
testing("eucliden_square",-1,100)
testing("eucliden_straightline",-1,140)