# word-embeddings-PCA
Word Embeddings PCA Visualization
---------------------------------------------------------------------------------------
This code reads in word embeddings from four different datasets: animals, countries, veggies, and fruits. It then performs PCA (Principal Component Analysis) on each dataset to reduce the dimensionality of the embeddings from 300 features to just 2. Finally, the reduced embeddings are plotted in a scatter plot with different colors for each category.

Requirements
--------------------
*numpy

*matplotlib

*panda

Usage
--------------------
1 - Place the data files (animals.csv, countries.csv, veggies.csv, and fruits.csv) in the same directory as the code.

2- Run the code using python word_embeddings_pca.py.

3-The scatter plot will be displayed showing the reduced embeddings for each category.

PCA Algorithm
-------------------------------------------------------------------------------------
The PCA algorithm used in this code follows these steps:

1-Normalize the data by subtracting the mean and dividing by the standard deviation.

2-Calculate the covariance matrix of the normalized data.

3-Calculate the eigenvectors and eigenvalues of the covariance matrix.

4-Sort the eigenvectors and eigenvalues in descending order.

5-Select the top k eigenvectors (where k is the desired number of dimensions for the reduced representation).

6-Project the normalized data onto the selected eigenvectors to get the reduced representation.

Output
---------------------------------------------------------------------------------------

The output of this code is a scatter plot showing the reduced embeddings for each category. The x and y axes represent the two dimensions obtained from PCA. Each point on the plot represents a word in the dataset, and the color of the point indicates which category the word belongs to.



