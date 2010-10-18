#!/usr/bin/env python

#general Python imports
from numpy import *
import scipy.misc.pilutil as pilutil
from pylab import *
from scipy.ndimage import *
from pysparse import spmatrix as spmatrix
#from scipy.sparse import *

import time

#module is designed to create a graph where pixels are nodes and edges are
#a measure of intensity difference between adjacent pixels

#initialize level 1 variables
class SWA():
    
    def SetImage(self,inImage):
        self.Image=inImage.copy()
        self.Intensity=self.Image.flatten() #linear representation
        self.Level=1
        self.NumNodes=self.Image.size
        self.U=zeros(self.NumNodes,dtype=bool)
        #user defined parameters
        self.alpha = 8 #intensity scale factor
        self.alpha2 = 4 #coarse-level intensity rescaling factor
        self.beta=200#coarse-level variance rescaling factor
        self.theta=0.025#coarsening strength threshold
        self.gamma = 0.05#saliency threshold
        self.d1=0.15#sharpening threshold
        self.sigma=2#min level segment detection threshold
        self.rho=2#variance rescaling threshold level
        
        #setup edge or "coupling" matrix
        numRow=self.Image.shape[0]
        numCol=self.Image.shape[1]
        #there will be N*(N-1) connections in each direction of a square matrix
        #self.A=lil_matrix((self.NumNodes,self.NumNodes))
        #self.A=spmatrix.ll_mat_sym(self.NumNodes,numRow*(numCol-1)+numCol*(numRow-1))
        self.A=spmatrix.ll_mat(self.NumNodes,self.NumNodes,numRow*(numCol-1)+numCol*(numRow-1))
        
        print 'calculating A'
        curr=time.time()
        #calculate all the neighbor in each direction
        horizNeighbors=asarray(exp(-self.alpha*abs(self.Image[:,0:numCol-1]-self.Image[:,1:numCol])),dtype=float64)
        vertNeighbors=asarray(exp(-self.alpha*abs(self.Image[0:numRow-1,:]-self.Image[1:numRow,:])),dtype=float64)
        print 'numNeighbors=%d',vertNeighbors.size+horizNeighbors.size
        #now populate the sparse matrix A with the neighbor values
        for a in range(horizNeighbors.shape[0]):
            for b in range(horizNeighbors.shape[1]):
                i=a*numCol+b
                j=i+1
                self.A[j,i]=self.A[i,j]=horizNeighbors[a,b]
                
        for a in range(vertNeighbors.shape[0]):
            for b in range(vertNeighbors.shape[1]):
                i=a*numCol+b
                j=i+numCol
                self.A[i,j]=self.A[j,i]=vertNeighbors[a,b]
       
        print '%f seconds'%(time.time()-curr)
        #variance matrix
        self.S=spmatrix.ll_mat(self.NumNodes,self.NumNodes)
        
        print 'calculating L'
        curr=time.time()
        #set up weighted boundary length matrix L,correct diagonals

        self.L=self.A.copy()
        self.L.scale(-1.)
        #allocate buffers
        theOnes=ones(self.NumNodes,dtype=float64)
        theDiagIdx=arange(self.NumNodes)
        theDiag=zeros(self.NumNodes,dtype=float64)
        theSums=zeros(self.NumNodes,dtype=float64)
        self.A.take(theDiag,theDiagIdx,theDiagIdx)
        self.A.matvec(theOnes,theSums)
        self.L.put(theSums-theDiag,theDiagIdx,theDiagIdx)
        
        
        print '%f seconds'%(time.time()-curr)
        
        print 'calculating V'
        curr=time.time()
        #setup area matrix V
        self.V=self.A.copy()
        self.V.generalize() #0 if Aij ==0
        theNons=self.V.keys()
        #theNons=zip(theNons[0],theNons[1])
        self.V.put(1.0,theNons[0],theNons[1])#1 if Aij!=0
        #for item in theNons:
        #    self.V[item]=1 #1 if Aij!=0
        print '%f seconds'%(time.time()-curr)
        
        print 'calculating G'
        curr=time.time()
        #setup the boundary length matrix G,correct diagonals
        self.G=self.V.copy()
        self.G.scale(-1.)
        
        self.V.take(theDiag,theDiagIdx,theDiagIdx)
        self.V.matvec(theOnes,theSums)
        self.G.put(theSums-theDiag,theDiagIdx,theDiagIdx)
        
        print '%f seconds'%(time.time()-curr)
        
        #calculate the initial saliency vector Gamma
        print 'calculating Gamma:'
        curr=time.time()
        self.Saliency = zeros(self.NumNodes,dtype=float64)
        theDiag2=zeros(self.NumNodes,dtype=float64)
        self.G.take(theDiag,theDiagIdx,theDiagIdx)
        self.L.take(theDiag2,theDiagIdx,theDiagIdx)
        self.Saliency=theDiag2/theDiag
        print '%f seconds'%(time.time()-curr)


    #recursive function that modifies self.U, the segment assignments
    #returns U, the 
    def imageVCycle(self):
        if self.Level<=self.sigma:
            self.Saliency+=numpy.inf
        C = self.coarsenAMG()
        #if self.NumNodes ==len(C): #no further coarsening obtained
        #    return ones(self.NumNodes,self.NumNodes)
        #else:
        #    junk=1
        
        return C
            
    #uses A, Saliency (Gamma), gamma, theta to coarsen graph, reducing the
    #number of C points per level
    def coarsenAMG(self):
        print 'calculating AHat'
        curr=time.time()
        
        M=self.A.shape[0] #number or rows in A
        AHat=self.A.copy() #0 if Aij ==0
        #zero out diagonal
        theRange=arange(M,dtype=int)
        theSums=zeros(M,dtype=float64)
        theBuff=zeros(M,dtype=float64)
        AHat.put(theSums,theRange,theRange)
        #calculate row sums for weakness criterion
        theOnes=ones(M,dtype=float64)
        theOnesInt=ones(M,dtype=int)
        AHat.matvec(theOnes,theSums)#calcs row sums, stores in theSums
        #eliminate (zero out) Connections from A that are "weak"
        theNons=AHat.keys()
        #theNons=zip(theNons[0],theNons[1])
        theSums = theSums*self.theta #scale theSums by coarsening threshold
        theLambda=zeros(M,dtype=float64) #number of nonzeros in each row
        #theValues=AHat.values()
        #theItems=AHat.items()
        #theGreater = [ item[0][0] for item in theItems if item[1]>theSums[item[0][0]] ]
        #theLesser = [ item[0] for item in theItems if item[1]<=theSums[item[0][0]] ]
        #
        #for idx in theGreater:
        #    theLambda[idx]+=1
        #
        #for item in theLesser:
        #    AHat[item]=0
            
        for item in AHat.items():
            theIdx=item[0][0]
            if item[1] < theSums[theIdx]:
                AHat[item[0]]=0 #zero out weak connections
            else:
                theLambda[theIdx]+=1 #update number of strong connections for i
        
        #take each column and numpy compare to 0 AHat.matvec_transp(theOnes,theLambda)
        #for colNum in range(M):
        #    AHat.take(theBuff,colNum*theOnesInt,theRange)
        #    theLambda[colNum]=(theBuff>theSums[colNum]).sum()

        T=zeros(M,dtype=int8) #tracks which set node is assigned,0=unassigned,1=C Point, 2=F Point
        #test nodes for saliency
        salientNodes = self.Saliency<self.gamma #error with inital Saliency calc?
        T+=salientNodes#assigns val of 1 to salientNodes
        theLambda[salientNodes]=0
        
        C=None
        
        print '%f seconds Setup time'%(time.time()-curr)
        
        print 'starting on unassigned nodes'
        #while there are unassigned nodes remaining
        counter=0
        sortTime=0
        kGenTime=0
        kFiltTime=0
        hFiltTime=0
        
        curr=time.time()
        #init a priority queue based on lambda value
        theZeros=where(T==0)[0]#find locations of zeros
        lambdaVals=theLambda[theZeros]
        sortFunc=lambda a,b: cmp(a[0],b[0]) or cmp(b[1],a[1])
        combo = sorted(zip(lambdaVals,theZeros),cmp=sortFunc,reverse=False)
        print '%f seconds to generate Priority Queue'%(time.time()-curr) #typically 4 seconds
        curr=time.time()
        while 0 in T:
            if counter%1000==0:
                print '%d of %d'%(counter,len(where(T==0)[0]))
                print 'sortTime=%f\nkGenTime=%f\nkFiltTime=%f\nhFiltTime=%f'%(sortTime,kGenTime,kFiltTime,hFiltTime)
                print time.time()-curr
            counter+=1
            
            #temp=time.time()
            #theZeros=where(T==0)[0]#find locations of zeros
            #lambdaVals=theLambda[theZeros]
            #default python sorts on first tuple val, then second in case of ties
            #we need descend on tuple val1, ascend on ties
            #sortFunc=lambda a,b: cmp(a[0],b[0]) or cmp(b[1],a[1])
            #combo = sorted(zip(lambdaVals,theZeros),cmp=sortFunc,reverse=True)
            #combo.sort(cmp=sortFunc)
            j=combo.pop()[1]#find max lambda when T==0, node will influence most neighbors
            #sortTime+=time.time()-temp
            
            #temp=time.time()
            T[j]=1
            theLambda[j]=0#point is now a C Point
            #print 'j=%d and lambda=%d'%(j,combo[0][0])
            #K is the set of all unassigned(T=0) points influenced strongly by j
            theOnes = ones(M,dtype=int32)
            AHat.take(theSums,theRange,theOnes*j)
            theStrongConnect=where(theSums>0)[0]
            theUnassigned = where(T[theStrongConnect]==0)[0]
            K=theStrongConnect[theUnassigned]#selects only item from theStrong that have T==0
            #kGenTime+=time.time()-temp
            #print 'K vector has %d points'%len(K)
            
            #temp=time.time()
            T[K]=2
            for k in K:
                combo.remove((theLambda[k],k))
                theLambda[k]=0
                AHat.take(theSums,theOnes*k,theRange)
                theStrongConnect=where(theSums>0)[0]
                theUnassigned = where(T[theStrongConnect]==0)[0]
                H=theStrongConnect[theUnassigned]
                
                temp2=time.time()
                for h in H:
                    theIndex=combo.index((theLambda[h],h))
                    del combo[theIndex]
                    theLambda[h]+=1
                    combo.insert(theIndex,(theLambda[h],h))
                #hFiltTime+=time.time()-temp2
            #kFiltTime+=time.time()-temp
            
            C=where(T==1)[0]
            
        print '%f seconds'%(time.time()-curr)
        return C

    
    #final assignment of pixels to most appropriate segment
    def assignPixels(self):
        return
 
if __name__ == "__main__":
    #anImage = pilutil.imread('testImage.png',flatten=True)
    anImage = pilutil.imread('bob00177.jpg',flatten=True)
    theSWA=SWA()
    theSWA.SetImage(anImage)
    C=theSWA.imageVCycle()
    theSWA.assignPixels(C)
    print 'finished setup'

