#!/usr/bin/evn python

"""
ENPM808F: Robot Learning
Spring 2019
CMAC Implementation - Discrete as well as Continuous

Author(s): 
Abhishek Kathpal (akathpal@terpmail.umd.edu) 
UID: 114852373
M.Eng. Robotics
University of Maryland, College Park
"""

import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt
import argparse



def train_cmac(trainLabel,trainData,g=35,lr=0.01,maxEpochs=5000,r=1):
	# Selecting random 35 weights from 0 to 1 
	weight = np.random.rand(35)
	pred=np.zeros((len(trainLabel),1))

	epochLoss = []
	numEpochs = []
	maxEpochs = 5000
	num = 0
	counter = 0
	while counter!=len(trainData) and num<maxEpochs:
		counter = 0
		num = num+1
		numEpochs.append(num)
		loss = []
		loss_abs = []
		for i in range(len(trainData)):
			w = 0
			start = int(math.floor((i)*35/70))
			if r==1:
				end = start+g # Discrete CMAC
			else:
				end = start+g+1 # Continuous CMAC
			if end>34:
				end=35

			for j in range(start,end):
				if j == start:
					w = w + weight[j]*r
				elif j == end-1:
					w = w + weight[j]*(1-r)
				else:
					w = w + weight[j]

			pred[i] = w*trainData[i]
			l = trainLabel[i] - pred[i]
			loss.append(l)
			for k in range(start,end):
				if k== start:
					weight[k] = weight[k] + lr*loss[i]*r/g
				elif k == end-1:
					weight[k] = weight[k] + lr*loss[i]*(1-r)/g
				else:
					weight[k] = weight[k] + lr*loss[i]/g

		epochLoss.append(abs(np.mean(loss)))
		# print("Epoch "+str(num))

		counter =0
		for i in range(len(trainData)):
			if abs(loss[i])<0.1:
				counter = counter +1
	return epochLoss,numEpochs,pred,weight


def test(testLabel,testData,weight,g,r=1):
	predTest = []
	acc = []
	for i in range(len(testLabel)):
		w = 0
		start = int(math.floor(i*35/30))
		if r==1:
			end = start+g
		else:
			end = start+g+1
		if end>34:
			end = 35
		for j in range(start,end):
			if j == start:
				w = w + weight[j]*r
			elif j == end-1:
				w = w + weight[j]*(1-r)
			else:
				w = w + weight[j]

		predTest.append(w*testData[i])
		l = abs((testLabel[i] - predTest[i])*100/testLabel[i])
		acc.append(100-l)
		

	return predTest,acc
	

def main():

	Parser = argparse.ArgumentParser()
	Parser.add_argument('--Method', default="Continuous", help='Continuous/Discrete')
	# Parser.add_argument('--Generalization',default=35,help='Any int- 3 to 35')
	Parser.add_argument('--Ratio',default=0.5,help='Only required for Continuous Model (0,1) ')

	Args = Parser.parse_args()
	Method = Args.Method
	# g = Args.Generalization
    
	theta = math.pi
	trainData = []
	testData = []
	trainLabel = []
	testLabel = []

	# 70:30 split for train and test data
	for u in range(70):
		trainData.append(u*theta/70)

	for u in range(30):
		testData.append(u*theta/30)

	for i in trainData:
	    trainLabel.append(math.sin(i))

	for i in testData:
	    testLabel.append(math.sin(i))

	# print(trainData)
	# print(trainLabel)
	
	for g in range(3,36,8):
		lr = 0.01

		if Method.lower() == "continuous":
			print("Continuous CMAC"+"\n")
			r=Args.Ratio
			print("Ratio ="+str(r)+"\n")
		else:
			print("Discrete CMAC"+"\n")
			r=1

		print("Generalization ="+str(g)+"\n")

		epochLoss,numEpochs,pred,weight = train_cmac(trainLabel,trainData,g,lr,r)
		
		# print(max(epochLoss))
		fig = plt.figure()
		plt.title(Method+"CMAC Loss"+"-- g="+str(g)+" r="+str(r))
		plt.plot(numEpochs[1:100],epochLoss[1:100])
		plt.xlabel("Epochs")
		plt.ylabel("Loss")
		# plt.show()
		plt.savefig(Method+" Loss-- g="+str(g)+" r="+str(r)+".jpg", dpi=fig.dpi)
		plt.close()

		predTest,acc = test(testLabel,testData,weight,g,r)
		fig = plt.figure()
		plt.title(Method+" CMAC"+"-- g="+str(g)+" r="+str(r))

		plt.ylabel("Predicted Output")
		plt.step(testData,predTest,'g--')
		# plt.scatter(testData,predTest)
		plt.scatter(testData,testLabel)
		# plt.show()
		plt.savefig(Method+"-- g="+str(g)+" r="+str(r)+".jpg",dpi = fig.dpi)
		plt.close()

		fig = plt.figure()
		plt.title(Method+"CMAC Accuracy"+"-- g="+str(g)+" r="+str(r))
		n = np.linspace(1,30,30)
		plt.plot(n,acc)
		plt.ylabel("Test Accuracy")
		# plt.show()
		plt.savefig(Method+" Accuracy-- g="+str(g)+" r="+str(r)+".jpg", dpi=fig.dpi)
		plt.close()

if __name__ == '__main__':
    main()