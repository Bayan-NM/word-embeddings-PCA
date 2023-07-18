import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

def read_embeddings(file):
	
	lines=open("../data/%s"%file).read().splitlines()
	X = np.zeros((len(lines),300), dtype=float)
	Y=[]
	for i,line in enumerate(lines):
		line=line.strip().split(" ")
		embed=np.array([float(i) for i in line[1:]])
		X[i]=embed
		Y.append("%s"%file)		
	return X,Y

def PCA(X,num_components):
	mean=np.mean(X,axis=0)
	centered_X = X - mean
	print(centered_X.shape)
	cov = np.cov(centered_X,rowvar=False)

	eigen_values , eigen_vectors = np.linalg.eigh(cov)
	sorted_index = np.argsort(eigen_values)[::-1]
	sorted_eigenvalue = eigen_values[sorted_index]
	sorted_eigenvectors = eigen_vectors[:,sorted_index]

	eigenvector_subset = sorted_eigenvectors[:,0:num_components]

	X_reduced = np.dot(eigenvector_subset.transpose() , centered_X.transpose() ).transpose()

	return X_reduced


if __name__=="__main__":
	X_country,Y_1 = read_embeddings("countries")
	X_animal,Y_2 = read_embeddings("animals")
	X_veggies,Y_3 = read_embeddings("veggies")
	X_fruit,Y_4 = read_embeddings("fruits")
	
	X=np.concatenate((X_country,X_animal,X_veggies,X_fruit))
	print(X.shape)
	Y_1.extend(Y_2)
	Y_1.extend(Y_3)
	Y_1.extend(Y_4)

	X_reduced=PCA(X,2)

	dic={}
	dic['PCA 1']=X_reduced[:,0]
	dic['PCA 2']=X_reduced[:,1]
	dic['word-type']=Y_1
	
	sns.set_style("darkgrid")
	sns.scatterplot(
	    x="PCA 1", y="PCA 2",
	    hue="word-type",
	    palette=sns.color_palette("hls", len(set(Y_1))),
	    data=dic,
	    alpha=0.3
	)
	plt.show()
