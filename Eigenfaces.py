from numpy import array, dot, mean, std, empty, argsort ,size ,shape ,transpose
from numpy.linalg import eigh, solve
from numpy.random import randn
from matplotlib.pyplot import subplots, show
import Image
import numpy as np
import os

def pca(data,k,m):
    """
        Principal component analysis using eigenvalues
        note: this mean-centers and auto-scales the data (in-place)
    """
    mean_mat =data-m
    C = np.dot(mean_mat.transpose(),mean_mat)
    E, V = eigh(C)							#E eigen values and V is eigen vectors	
    key = argsort(E)[::-1][:k]					#key will have indices of array
  
    E, V = E[key], V[:, key]
    V=V.T									#eigen matrix ka transpose bhj rha hu i.e. k*48
    #print "Dim of eigen matrix",V.shape,data.shape
    U = np.dot(V,data.transpose())					# U is projection matrix
    return U,V	

#----------------Start of program---------------

dirname="aligneddataset/bw/jpg4/"
print dirname
count=0
size=48,48								#for color pics change it to 140,140

for filename in os.listdir(dirname):
	count=count+1
	image_file=Image.open(dirname+filename)
	size=image_file.size					#initializing size of the image

total=count								#total no. of images
data=np.zeros((total//2,size[0]*size[1]))			#trainging data is half the aize of total no. of images

count=0
trainglabel=[]							#training label
testlabel=[]							#testing label
trainf=[]								#training filename
testf=[]								#training filename,used later for tesing each test image
j=0

for filename in os.listdir(dirname):
	
	if count < total//2:
		trainglabel.append(filename[7])
		trainf.append(filename)
		#print filename[7]
		image_file=Image.open(dirname+filename)	
		temp = array(image_file.convert('1'))
		A = np.asarray(temp).reshape(-1)
		#print temp.shape,A.shape
		data[count,:]=A
		
	else:
		testf.append(filename)
		testlabel.append(filename[7])
	count=count+1	
#print label
print "------------------Start---------------------------"
m=mean(data, 0)
#print "Dim of mean ",m.shape
k=200				#751/10=75
projmatrix,eigenmatrix = pca(data,k,m)
np.save("eigmat1.txt",eigenmatrix)				#saving eigen values for backup
#print projmatrix.shape
#print "Eigen Matrix",eigenmatrix.shape			#Eigen Matrix of size==k X size[0]*size[1]

eigenfaces=eigenmatrix*m					#eigenfaces=eigenmatrix*mean of data
#print eigenfaces.shape	
wttraining=np.dot(eigenfaces,data.transpose()).transpose()			#sizze of weight for trainging data == SIZE(training X k)

#print wttraining.shape
#print data.shape[0],len(trainglabel),len(trainf),len(testlabel),len(testf)

male=0
female=0
correct=0
incorrect=0

for j in range(len(testf)):
	testimg=Image.open(dirname+testf[j])
	testimg=np.asarray(testimg).reshape(-1)
	#print eigenfaces.shape,testimg.transpose().shape
	wttest=np.dot(eigenfaces,testimg.transpose()).transpose()			#size of weight for teseting data==SIZE(1 X K)
	#print wttest.shape 

	cur=1000000000
	mini=1000000000
	maxi=0
	ind=0
	
	for i in range(len(data)):
		cur=np.linalg.norm(wttraining[i,:]-wttest[:])				#Dist of ith-EigenFace with current test data
		if mini>cur:
			mini=cur
			ind=i
	if trainglabel[ind]==testlabel[j]:							#ind is the Least distance,i.e.predicted value

		correct=correct+1

		if testlabel[j]=='m':
			male=male+1
		else:
			female=female+1

	else:
		incorrect=incorrect+1		

print "Correct,Incorrect,Predicted-Male,Total Male,Predicted-female,Total Female"	
print correct,incorrect,male,testlabel.count('m'),female,testlabel.count('f')
acc=correct*1.0/(incorrect+correct)
print "Accuracy:",acc
print "Male Accuracy:",(male*1.0)/testlabel.count('m')
print "Male Error:",1-(male*1.0)/testlabel.count('m')
print "Female Accuracy:",(female*1.0)/testlabel.count('f')
print "Female Error:",1-(female*1.0)/testlabel.count('f')
print "done"

