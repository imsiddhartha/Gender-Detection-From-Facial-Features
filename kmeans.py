from numpy import array, dot, mean, std, empty, argsort ,size ,shape ,transpose
from numpy.linalg import eigh, solve
from numpy.random import randn
from matplotlib.pyplot import subplots, show
import Image
import numpy as np
import os
import collections
import operator

def kmeans(d,m,f,k):						#d-data ,m-initial mean of the data,f-number of data items,k-K for K-means
	print f,d.shape
	col = np.zeros((d.shape[0], 1))
	p=0

	while 1:
		
		flag=0						#flag used for terminating from while Loop
		j=0
		
		for cell in d:
			
			i=0
			ind=0
			cur=0
			mini=1000000
			
			for mean in m:
			
				cur=np.linalg.norm(cell[:]-mean)		#distance b/w current item and mean
				#print cur,i
				if cur < mini:					#ind will have ith cluster to which current data itm is assigned
					ind=i
					mini=cur
				i=i+1
			
			if col[j] != ind:
				flag=1 
			
			col[j]=ind
			j=j+1
		
		if flag==0:
			break
		
		for i in range(k):						#recalculating the means
			count=1
			mean=np.zeros((1, len(d[0])))
			for cell in d:
				if col[count-1]==i:
					mean=mean+cell[:]
					count=count+1
			
			if count !=1:
				mean=mean/count
				m[i:i+1]=mean[:]
		p=p+1
	
	#print "No of iterations:",p
	return m


#------------start of program---------------

dirname="aligneddataset/bw/jpg4/"
#dirname="aligneddataset/colored/jpg5/"
print dirname
count=0
size=48,48
#print size[0],size[1]
m=0
f=0

for filename in os.listdir(dirname):
	
	image_file=Image.open(dirname+filename)
	size=image_file.size					#size == size of the image file,will be used to initializing Array
	if filename[7]=='m':
		m=m+1
	else:
		f=f+1
	count=count+1
print "Size of image",size
print "Total,Males,Females",count,m,f

maledata=np.zeros((200,size[0]*size[1]))			#for training data 200 male and 200 female faces
femaledata=np.zeros((200,size[0]*size[1]))	

fmean=np.zeros((10,size[0]*size[1]))
mmean=np.zeros((10,size[0]*size[1]))

m=0
f=0
count=0
testfilenm=[]							#This list will be containing all the files that are in test-data set
testlabels=[]							#This list will be containing all the labels for a file
for filename in os.listdir(dirname):

	image_file=Image.open(dirname+filename)	
	temp = array(image_file.convert('1'))
	A = np.asarray(temp).reshape(-1)
	if filename[7]=='m':
		if m <200:						#if m <200 then put it into traiging male data
			maledata[m,:]=A
		else:							#else put it into teset data,Similar for female faces
			testfilenm.append(filename)
			testlabels.append('m')
		m=m+1
	else:
		if f<200:
			femaledata[f,:]=A
		else:
			testfilenm.append(filename)
			testlabels.append('f')
		f=f+1
	count=count+1

print "Total,Males,Females in training data","400","200","200"
print "Total,Males,Females in test data",len(testlabels),m-200,f-200

for i in range(10):						#initializing means of both males and females
	fmean[i,:]=femaledata[i*10,:]
	mmean[i,:]=maledata[i*10,:]
print "Called"
fmean=kmeans(femaledata,fmean,f,10)				#10 most representative female faces
mmean=kmeans(maledata,mmean,m,10)				#10 most representative male faces

print "Done"

k=5									#K for k-nearest neighbour


cmale=0								#cmale and cfemale for correct classified male and female faces
cfemale=0

imale=0								#imale and ifemale for incorrect classified male and female faces
ifemale=0			

correct=0
incorrect=0

ptr=0									#counter for accessing testlabels

for filename in testfilenm:
	image_file=Image.open(dirname+filename)
	temp = array(image_file.convert('1'))
	test = np.asarray(temp).reshape(-1)

	mini=100000000
	result={}
	
	for  j in range(len(fmean)*2):
	
		if j <len(fmean):
			cur=np.linalg.norm(fmean[j]-test)
			result[cur,j]=0				#cur,j for making each key unique
		else:	
			cur=np.linalg.norm(mmean[j-10]-test)
			result[cur,j]=1				#key<10 for female and >10 for male

	od = collections.OrderedDict(sorted(result.items()))	#Sorting dictionary on keys
	m=0
	f=0
	count=0
	
	for k, v in od.iteritems():
		if count >=5:
			break
		if v==0:
			f=f+1
		if v==1:
			m=m+1
		count=count+1
	if m>f:						#if num of males>females in k-NN then
		if testlabels[ptr]=='m':		#check test label if it is m then correct classified
			correct=correct+1
			cmale=cmale+1
		else:						#else incorrect classified
			incorrect=incorrect+1	
			imale=imale+1
	else:
		if testlabels[ptr]=='f':
			correct=correct+1
			cfemale=cfemale+1
		else:
			incorrect=incorrect+1
			ifemale=ifemale+1	
	ptr=ptr+1

print correct,incorrect
print cmale,cfemale,testlabels.count('m'),testlabels.count('f')
print "Accuracy:",correct*1.0/(incorrect+correct)
print "Men Accuracy:",cmale*1.0/testlabels.count('m')
print "Men Error:",1-cmale*1.0/testlabels.count('m')
print "Female Accuracy:",cfemale*1.0/testlabels.count('f')
print "Female Error:",1-cfemale*1.0/testlabels.count('f')

