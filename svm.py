from sklearn import svm
from numpy import array, dot, mean, std, empty, argsort ,size ,shape ,transpose
from numpy.linalg import eigh, solve
from numpy.random import randn
from matplotlib.pyplot import subplots, show
import Image
import numpy as np
import os
from sklearn.decomposition import PCA
from skimage import exposure
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


			
dirname="aligneddataset/colored/jpg4/"

count=0
size=48,64		#for color pics change it to 140,140
#print size[0],size[1]
for filename in os.listdir(dirname):
	count=count+1
	image_file=Image.open(dirname+filename)
	size=image_file.size

total=count
cc=0	
while cc<5:
	print "fold",cc
	data=np.zeros((total,size[0]*size[1]))
	count=0
	label=[]
	j=0

	for filename in os.listdir(dirname):
		label.append(filename[7])
		image_file=Image.open(dirname+filename)	
		temp = array(exposure.equalize_hist(image_file.convert('1'))) #equialize histogram
		A = np.asarray(temp).reshape(-1)
		#print temp.shape,A.shape
		data[count,:]=A	
		count=count+1	
	traindata,testdata,trainglabel,testlabel=train_test_split(data,label,test_size=0.2,random_state=30)
	k=150				#751/10=75

	pca = PCA(n_components=k)
	pca.fit(traindata)
	red_mat=pca.transform(traindata)

	pca = PCA(n_components=k)
	pca.fit(testdata)
	red_mat2=pca.transform(testdata)

	clf = svm.SVC(kernel='poly', C=1.0, gamma=0.10000000000000001)
	trainlabels=np.array(trainglabel[:])
	testlabels=np.array(testlabel[:])
	clf.fit(traindata,trainlabels)

	res=clf.predict(testdata)

	correct=0
	incoreect=0
	m=0
	f=0
	#print res

	#print "RBF Kernel"
	cm_linear=confusion_matrix(testlabels,res)
	accuracy_linear=accuracy_score(testlabels,res)
	report_linear=classification_report(testlabels,res)
	print(cm_linear)
	print "ACCURACY"
	print(accuracy_linear)
	print "REPORT"
	print(report_linear)
	cc=cc+1
