import os
import numpy as np
np.random.seed()
import pylab as plt
import math
from parameters import params
import sys

def createStimRandomPos(nr):
	img = np.zeros((params['x_max'],params['y_max']), np.uint8)
	centerList=np.zeros((params['num_lines'],2))

	counter=1000
	while counter==1000:
		for line in range(params['num_lines']):
			x=np.random.randint(params['border'],params['x_max']-params['border'])
			y=np.random.randint(params['border'],params['y_max']-params['border'])
			center=np.array([y,x])
			counter=0
			while (np.sum(np.sqrt(np.sum((centerList-center)**2,1))<=2*params['border'])>0) and counter<1000:
				print (counter)
				x=np.random.randint(params['border'],params['x_max']-params['border'])
				y=np.random.randint(params['border'],params['y_max']-params['border'])
				center=np.array([y,x])
				counter+=1
			if counter==1000:
				break
			img[x,y]=params['phi'][line]
		
			centerList[line,:]=center

	return [img,centerList]

def gridStim(nr, phi=[50,50,50,50,50,50,50,50,55]):
	img = np.zeros((params['x_max'],params['y_max']), np.uint8)
	centerList=np.zeros((9,2))
	centerListShuff=np.zeros((9,2))
	
	x_c=params['x_max']//2
	y_c=params['y_max']//2
	
	gapX=8
	gapY=8

	idxC=0
	for idxA in [-1,0,1]:
		y=y_c+idxA*gapY
		x_offset=np.random.randint(-gapX//4,gapX//4)
		for idxB in [-1,0,1]:
			x=int(x_c+idxB*gapX)+x_offset
			centerList[idxC,:]=np.array([y,x])
			idxC+=1
	
	idxA=0
	idxList=np.arange(9)
	np.random.shuffle(idxList)
	for idxB in idxList:
		img[int(centerList[idxB,1]),int(centerList[idxB,0])]=phi[idxA]
		centerListShuff[idxA]=centerList[idxB]
		idxA+=1

	return [img,centerListShuff,phi]

def createStimT1(nr):
	return gridStim(nr)
def createStimT1Reversed(nr):
	return gridStim(nr, [60,60,60,60,60,60,60,60,55])
def createStimJustT(nr):
	return gridStim(nr, [60,60,60,60,60,60,60,60,60])
def createStimJustD(nr):
	return gridStim(nr, [30,30,30,30,30,30,30,30,30])
def createStimStandard(nr):
	return gridStim(nr, [30,30,30,30,30,30,30,30,60])
def createStimHeterogenD(nr):
	return gridStim(nr, [30,30,30,30,45,45,45,45,60])
def createStimLinearSep(nr):
	return gridStim(nr, [30,30,30,30,60,60,60,60,45])

def createStimT2(nr):
	phi=[30,50,55,60,80]
	img = np.zeros((params['x_max'],params['y_max']), np.uint8)
	centerList=np.zeros((5,2))
	centerListShuff=np.zeros((5,2))
	
	x_c=params['x_max']//2
	y_c=params['y_max']//2
	
	radiusX=8
	radiusY=8

	offset=np.random.uniform(0,np.pi*2)
	for idx in [0,1,2,3,4]:
		theta=((idx+1)/5.0)*2*np.pi+offset
		x=int(x_c+radiusX*np.cos(theta))
		y=int(y_c+radiusY*np.sin(theta))
		centerList[idx,:]=np.array([y,x])
	
	idxA=0
	idxList=np.arange(5)
	np.random.shuffle(idxList)
	for idxB in idxList:
		img[int(centerList[idxB,1]),int(centerList[idxB,0])]=phi[idxA]
		centerListShuff[idxA]=centerList[idxB]
		idxA+=1

	return [img,centerListShuff,phi]


def gauss(x,m,sig):
	xs=x.shape[0]
	ys=x.shape[1]
	x=np.reshape(x,xs*ys).astype(float)
	ret=[]
	for val in x:
		if val>0:
			ret.append(np.exp(-0.5*((val-m)/float(sig))**2))
		else:
			ret.append(0)
	"""
	if m==60:	
		plotX=np.arange(-90,90)
		plotY=np.exp(-0.5*((plotX-0)/float(sig))**2)
		plt.plot(plotX,plotY)
		plt.show()
	"""
	ret=np.array(ret)
	ret=np.reshape(ret,(xs,ys))
	return ret


N=10



filters=params['possibleOr']
v1Act= np.zeros((params['x_max'],params['y_max'],1,16))
sigV1 = ['','normal','wide','small'][int(sys.argv[1])]
sigV1_Value = {'normal': 35, 'wide': 47.43, 'small': 22.14, '': 35}[sigV1]
print(sigV1_Value)
for stimNr in range(400):

	if stimNr>=0 and stimNr<=49:
		[img,centerList,phi]=createStimT1(stimNr)
	elif stimNr>=50 and stimNr<=99:
		[img,centerList,phi]=createStimT2(stimNr)
	elif stimNr>=100 and stimNr<=149:
		[img,centerList,phi]=createStimStandard(stimNr)
	elif stimNr>=150 and stimNr<=199:
		[img,centerList,phi]=createStimJustT(stimNr)
	elif stimNr>=200 and stimNr<=249:
		[img,centerList,phi]=createStimJustD(stimNr)
	elif stimNr>=250 and stimNr<=299:
		[img,centerList,phi]=createStimHeterogenD(stimNr)
	elif stimNr>=300 and stimNr<=349:
		[img,centerList,phi]=createStimLinearSep(stimNr)
	elif stimNr>=350 and stimNr<=399:
		[img,centerList,phi]=createStimT1Reversed(stimNr)	
	
	for filterIdx in range(16):
		v1Act[:,:,0,filterIdx]=gauss(img,filters[filterIdx],sigV1_Value)#V4 Orientation-Tuning bandwidth ca 52 (median laut Visual Properties of Neurons in Area V4 of the Macaque: Sensitivity to Stimulus Form) --> normal=35, small=22.14, wide=47.43

	"""
	plt.figure("img")	
	plt.imshow(img)
	
	plt.figure("V1")
	for i in range(16):
		plt.subplot(4,4,i+1)
		plt.imshow(v1Act[:,:,0,i],vmin=0,vmax=1)
	
	plt.show()
	"""

	try:
		os.mkdir('new_stims_'+sigV1)
	except:
		if os.path.isdir('new_stims')==False:
			print('could not create new_stims_'+sigV1+' folder')
	np.save('new_stims_'+sigV1+'/img'+str(stimNr)+'.npy',img)
	np.save('new_stims_'+sigV1+'/input'+str(stimNr)+'.npy',v1Act)
	np.save('new_stims_'+sigV1+'/lineCoords'+str(stimNr)+'.npy',centerList)
	np.save('new_stims_'+sigV1+'/phi'+str(stimNr)+'.npy',phi)

	





