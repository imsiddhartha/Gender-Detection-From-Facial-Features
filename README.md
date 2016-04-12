# Gender-Detection-From-Facial-Features
Gender Detection From Facial Features using various methods like Eigenfaces,K-means,PCA then SVM

The objective of this project is to identify the gender of a person by looking at his/her photograph. This is a case of supervised learning where the algorithm is first trained on a set of female and male faces, and then used to classify new data.We have not taken genders other than Male and Female into account

First,We tried to identify gender from facial features, we are often curious about what features of the face are most important in determining gender.
We also use coloured (RGB) and B/W versions of the givenimages.Colour images have been compressed to 140x140 pixels and B/W to 64x48 pixels.

In this project, the following methods were used for classification:

1.Eigenface Method:

First I have applied Principle Component Analysis (PCA) to reduce the dimensionality.Then the eigen matrix returned by pca method is used for finding Eigenfaces for training data.
Now,for each image in test dataset,first calculate its eigenfaces matrix and then apply 1-NN algorithm to classify the image.

2.K-means:

I have applied K-means directly on the pixel data.From these,I got 10 clusters for female faces and 10 for male faces. Now, call these the 10 most representative female and male faces.Now, run the K Nearest Neighbours algorithm to classify the test images.Here K was chosen to be 5.

3.SVM Method:

The PCA was applied to reduce dimensionality of the vectors that serve as inputs to the SVM.Here,I have used Svm library from sklearn in python.


