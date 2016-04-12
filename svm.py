from sklearn import svm
from numpy import array, dot, mean, std, empty, argsort ,size ,shape ,transpose
from numpy.linalg import eigh, solve
from numpy.random import randn
from matplotlib.pyplot import subplots, show
import Image
import numpy as np
import os
from sklearn.decomposition import PCA


def pca(data,k,m):
    """
        Principal component analysis using eigenvalues
        note: this mean-centers and auto-scales the data (in-place)
    """
    #print mean(data, 0).shape
    print "Dim of data",data.shape
    mean_mat =data-m
    
    #x=mean_mat.transpose()
    print mean_mat.shape,mean_mat.transpose().shape
    C = np.dot(mean_mat.transpose(),mean_mat)
    print "Scatter matrix ka dim",C.shape
    #data -= mean(data, 0)
    #data /= std(data, 0)
    #C = cov(data)

    E, V = eigh(C)					#E eigen values and V is eigen vectors	
    key = argsort(E)[::-1][:k]			#key will have indices of array
  
    E, V = E[key], V[:, key]
    #print "---------"
    #print data.shape
    #print C.shape
    V=V.T							#eigen matrix ka transpose bhj rha hu i.e. k*48
    print "Dim of eigen matrix",V.shape,data.shape
    #print E.shape
    U = np.dot(V,data.transpose())					# U is projection matrix
    #print "Size of U"
    #print U.shape
    #print U
    return U,V	


dirname="aligneddataset/bw/jpg1/"

count=0
size=48,48		#for color pics change it to 140,140
print size[0],size[1]
for filename in os.listdir(dirname):
	count=count+1
	image_file=Image.open(dirname+filename)
	size=image_file.size

total=count	
traindata=np.zeros((total//2+1,size[0]*size[1]))
testdata=np.zeros((total//2,size[0]*size[1]))
count=0
trainglabel=[]
testlabel=[]
trainf=[]
testf=[]
j=0
print total
for filename in os.listdir(dirname):
	
	if count <= total//2:
		trainglabel.append(filename[7])
		trainf.append(filename)
		#print filename[7]
		image_file=Image.open(dirname+filename)	
		temp = array(image_file.convert('1'))
		A = np.asarray(temp).reshape(-1)
		#print temp.shape,A.shape
		traindata[count,:]=A
		
	else:
		#print j
		testf.append(filename)
		testlabel.append(filename[7])
		image_file=Image.open(dirname+filename)	
		temp = array(image_file.convert('1'))
		A = np.asarray(temp).reshape(-1)
		testdata[j,:]=A
		j=j+1
		
	count=count+1	
#print label
print traindata.shape
print count
#print data[0]
#print data[1]
print "------------------Start PCA---------------------------"
#m=mean(traindata, 0)
#print "Dim of mean ",m.shape
k=150				#751/10=75
print k
#projmatrix,eigenmatrix = pca(traindata,k,mean(traindata, 0))

#projmatrix2,eigenmatrix2 = pca(testdata,k,mean(testdata, 0))
#print projmatrix.shape,projmatrix2.shape

pca = PCA(n_components=k)
pca.fit(traindata)
red_mat=pca.transform(traindata)
red_mat2=pca.transform(testdata)
print red_mat.shape,red_mat2.shape

print len(trainglabel),len(testlabel)
#print eigenmatrix.shape,eigenmatrix2.shape
print "--------------------------PCA Done----------------------"


print "---------------------SVM-----------------------"
clf = svm.SVC()
trainlabels=np.array(trainglabel[:])
#clf = svm.SVC(decision_function_shape='ovo')
testlabels=np.array(testlabel[:])
#trainglabel=[]
#testlabel=[]
clf.fit(red_mat,trainlabels)

#dec = clf.decision_function([[1]])
#print dec.shape[1]

res=clf.predict(red_mat2)

correct=0
incoreect=0
m=0
f=0
#print res
print "RBF Kernel"
for i in range(len(res)):
	if res[i]==testlabels[i]:
		correct=correct+1
		if res[i]=='m':
			m=m+1
		else:
			f=f+1	
	else:
		incoreect=incoreect+1
print correct,incoreect
print m,f
print "Accuracy:",correct*1.0/(incoreect+correct)
print "Men Accuracy:",(m*1.0)/testlabel.count('m')
print "Men Error:",1-(m*1.0)/testlabel.count('m')
print "female Error:",1-(f*1.0)/testlabel.count('f')

