import numpy as np
import sys
import random
from sklearn.cluster import KMeans

def Data_Read(input_file,output_file):
    syn_input = np.genfromtxt(input_file, delimiter=',')
    syn_output= np.genfromtxt(output_file, delimiter=',').reshape([-1, 1])
#    letor_input= np.genfromtxt('datafiles/Querylevelnorm_X.csv', delimiter=',')
#    letor_output = np.genfromtxt('datafiles/Querylevelnorm_t.csv', delimiter=',').reshape([-1, 1])

    part_train= int(0.8*len(syn_input))
    part_valid= int(0.1*(len(syn_input)))+part_train
    part_test=len(syn_input)
    #print (syn_input)
    #print(syn_output)
    train_set=syn_input[:part_train,:]
    valid_set=syn_input[part_train:part_valid,:]
    test_set=syn_input[part_valid:part_test,:]

    inputs={}

    inputs["train"] = train_set
    inputs["validate"]=valid_set
    inputs["test"]= test_set


    train_set=syn_output[:part_train,:]
    valid_set=syn_output[part_train:part_valid,:]
    test_set=syn_output[part_valid:part_test,:]

    outputs={}
    outputs["train"] = train_set
    outputs["validate"] =valid_set
    outputs["test"]=test_set
    
    return inputs,outputs

def compute_design_matrix(X, centers, spreads):  # use broadcast
	basis_func_outputs = np.exp(
		np. sum(np.matmul(X - centers, spreads) * (X - centers), axis=2) / (-2)).T
# insert ones to the 1st col
	return np.insert(basis_func_outputs, 0, 1, axis=1)



def SGD_sol(learning_rate, minibatch_size, num_epochs, L2_lambda, design_matrix, output_data, weights):
	N, _ = design_matrix.shape
	for epoch in range(num_epochs):
		for i in range(int(N / minibatch_size)):
			lower_bound = i * minibatch_size
			upper_bound = min((i + 1) * minibatch_size, N)
			Phi = design_matrix[lower_bound: upper_bound, :]
			t = output_data[lower_bound: upper_bound, :]
			E_D = np.matmul((np.matmul(Phi, weights.T) - t).T, Phi)
			E = (E_D + L2_lambda * weights) / minibatch_size
			weights = weights - learning_rate * E
			# print weights
		# print(np.linalg.norm(E))
	return weights


def closed_form_sol(L2_lambda, design_matrix, output_data):
	return np.linalg.solve(L2_lambda * np.identity(design_matrix.shape[1]) + np.matmul(design_matrix.T, design_matrix), np.matmul(design_matrix.T, output_data)).flatten()

def Error(weights, design_matrix, output):
    return np.sum((np.matmul(design_matrix, weights.T) - output)**2)

def ErrorF(weights,lamda,ED,output):
    weights = weights.ravel()
    EW = np.dot(weights.T,weights)*(0.5)

    f = np.sqrt((ED+lamda*EW)/len(output))
    
    return f
    
    
def Early_Stop(design_matrix_train, design_matrix_validate, output_train, output_validate, patience, Threshold):
    j = 0
    num_epochs = 100
    v = sys.float_info.max
    best_weights = weights = np.zeros([1, len(design_matrix_train[0])])
    best_step = step = 0
    
    while j < patience:
        step = step + num_epochs
        weights = SGD_sol(learning_rate=1, minibatch_size=design_matrix_train.shape[0], num_epochs=num_epochs, L2_lambda=0.1, design_matrix=design_matrix_train, output_data=output_train, weights=weights)
        v_prim = Error(weights, design_matrix_validate, output_validate)
        
        print("Step: " + str(step) + "\nValidate Error: "+ str(v_prim) + "\n Current weights: " + str(weights))
        if v_prim - v < -Threshold:
            j = 0
            best_weights = weights
            best_step = step
            v = v_prim
        else:
            j = j+1
    return best_step, best_weights

####################################old Evaluation_Early_Stop
'''
def kmeans(train_data, nclusters,D):

    #print(inputs["train"])
    #def keams():

    km = KMeans(n_clusters = nclusters ).fit(train_data)
    centers = km.cluster_centers_
    for i in range(0,len(centers)):
            matrix =  train_data - centers[i]  #get the
            matrix = np.square(matrix)
            dia = np.sum(matrix, axis = 0)
            dia = np.divide(dia,D-1)
            a = np.zeros((D,D))
            np.fill_diagonal(a,dia)
           # print covmatrix
            #print dia
            #print a
            if(i == 0):
                spreads = np.array([a])
            else:
                spreads = np.concatenate((spreads,np.array([a])),axis = 0)
            
    return centers, spreads
'''
####################

def kMeans(M,data):
        kmeans = KMeans(n_clusters=M,random_state=0)
        kmeans.fit(data)
        centroid = kmeans.cluster_centers_

        labels=kmeans.labels_
        clusters = {}
        for label in range(0,M):
                cluster_index = np.where(labels == label)[0]
                cluster_data = []
                for p in cluster_index :
                    cluster_data.append(data[p])
                cluster_data = np.array(cluster_data)
                clusters[label] = cluster_data
        return centroid,clusters


def give_spreads(cluster,M):
        spread = []
        for i in range(M):
            spread.append(np.linalg.pinv(np.cov((np.array(cluster[i]).T))))
        return spread

#################

def Evaluation(input_file, output_file, n_center):
    inputs, outputs = Data_Read(input_file, output_file)
    _, D = inputs["train"].shape
    
    centers,cluster = kMeans(n_center,inputs["train"] )

    spreads = give_spreads(cluster,n_center)
    
    centers = centers[:, np.newaxis, :]
    # shape = [M, D, D]
    
    centers = np.around(centers, decimals=6)
    spreads=np.around(spreads,decimals=6)
    
    #print(centers)
    design_matrix = {}
    design_matrix["test"] = compute_design_matrix(inputs["test"][np.newaxis, :, :], centers, spreads)
    
    #print(spreads)
    #centers = np.array([np.ones((D)) * 1, np.ones((D)) * 0.5, np.ones((D)) * 1.5])
    #centers = np.array([np.ones((D)) * 1, np.ones((D))* 0.5])
    #centers = centers[:, np.newaxis, :]
    #print(centers.shape)
    #print(centers)
    
    design_matrix["validate"] = compute_design_matrix(inputs["validate"][np.newaxis, :, :], centers, spreads)

    # shape = [M, D, D]
    #spreads = np.array([np.identity(D),np.identity(D),np.identity(D)]) * 0.5  # shape = [1, N, D]
    #print centers
    #print spreads
    design_matrix["train"] = compute_design_matrix(inputs["train"][np.newaxis, :, :], centers, spreads)
    
    best_step, best_weights = Early_Stop(design_matrix["train"], design_matrix["validate"], outputs["train"], outputs["validate"], 10, 0.01)
    print('\n ----------------------------\n')
    print("For data set " + input_file +" and " +output_file )


    print("Training step: " + str(best_step) + "\n Weights: " + str(best_weights))

    ED = Error(best_weights, design_matrix["test"], outputs["test"])
   # print("The test error is: " + str(ED))
    EF = ErrorF(best_weights, 0.1, ED , outputs["test"] )
    EF= np.around(EF, decimals=5)
    print( "The Error RMS  is "+str(EF))

    
    return

#Evaluation("datafiles/input.csv", "datafiles/output.csv",5)
Evaluation("datafiles/Querylevelnorm_X.csv", "datafiles/Querylevelnorm_t.csv",15)


###############################
'''

inputs, outputs = Data_Read("data/input", "data/output")

_, D = inputs["train"].shape
# Assume we use 3 Gaussian basis functions M = 3
# shape = [M, 1, D]
#print(inputs["train"])
# print(km.cluster_centers_)
#print(kmc.shape)
centers = np.array([np.ones((D)) * 1, np.ones((D)) * 0.5, np.ones((D)) * 1.5])
#print(centers.shape)
#print(centers)
centers = centers[:, np.newaxis, :]
print(centers)
# shape = [M, D, D]
spreads = np.array([np.identity(D), np.identity(D), np.identity(D)]) * 0.5  # shape = [1, N, D]
print(spreads)


print("=============================================")

km = KMeans(n_clusters = 3 ,random_state = 0).fit( inputs["train"])
centers = km.cluster_centers_

for i in range(0,len(centers)):
    matrix =  inputs["train"] - centers[i]  #get the

    matrix = np.square(matrix)

    dia = np.sum(matrix, axis = 0)

    dia = np.divide(dia,D-1)
    
    a = np.zeros((D,D))

    np.fill_diagonal(a,dia)
    

    
    print matrix
    print dia
    print a

    if(i == 0):
        spreads = np.array([a])
    else:
        spreads = np.concatenate((spreads,np.array([a])),axis = 0)


centers = centers[:, np.newaxis, :]
'''
#print centers

#print spreads
'''

a=np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5]]).T
b = np.cov(first.T)
'''

