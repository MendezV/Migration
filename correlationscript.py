#import pandas as pd
#import seaborn as sns
import csv
import math
#import vincent
#import json
import numpy as np
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt

a=list(csv.reader(open('BEA_County_Population_1969_2015(1).csv')))

c=(a[1875][8:-1])
#plt.plot(np.diff((np.array(c, dtype=np.int))))
#plt.show()
corr=np.correlate(np.diff((np.array(c, dtype=np.int))),np.diff((np.array(c, dtype=np.int))),'full');
corr=(corr[(np.floor(np.size(corr)+1)/2.0):-1])



#plt.plot(corr/np.float(corr[0]))
#plt.show()


##basics
counter=0
Meanfit=[]
Meansamp=[]
Stdfit=[]
Stdsamp=[]
Meanplus=[]
Meanneg=[]
Stdplus=[]
Stdneg=[]

CORR=np.zeros(43)

from scipy.special import erf
from scipy.optimize import curve_fit

def erfunc(x, mu, sig):
	return 0.5*(1+erf((x-mu)/(sig*np.sqrt(2))))
counts=0.0

for i in range(1,np.size(np.array(a).T[1,:]) ):
	
	#print i
	c= np.array([x for x in a[i][7:-2] if x!='(NA)' ] , dtype=np.int) #####key formatting thingy
	if (np.size(c)>45):
		counter+=1
		diffs=np.diff((np.array(c, dtype=np.int)))
		#corr=np.correlate(diffs,np.array(c, dtype=np.int),'full');
		#corr=np.correlate(diffs,np.diff(diffs),'full');
		#corr=np.correlate(diffs,diffs,'full')
		#corr=np.correlate(((np.array(c, dtype=np.int))),((np.array(c, dtype=np.int))),'full');
		#corr=(corr[(np.floor(np.size(corr)-1)/2.0):-1])
		#plt.plot(diffs) ###uncomment for corr plot
		#plt.title('difference in the logarithm of the population vs time '+a[i][1]+' '+str(c[0]))
		#plt.show()
		bins = np.arange(np.floor(diffs.min()),np.ceil(diffs.max()))
		values, base = np.histogram(diffs, bins=bins, density=1)
		u, indices = np.unique(np.cumsum(values), return_index=True)
		init=bins[:-1]
		x=[init[j] for j in indices]
		#print values
		superior_params, extras = curve_fit(erfunc, x, u, p0=[np.mean(diffs),np.std(diffs)])
		Meansamp.append(np.mean(diffs))
		Meanfit.append(superior_params[0])
		Stdsamp.append(np.std(diffs))
		Stdfit.append(superior_params[1])
		if (np.mean(diffs)>0):
			Meanplus.append(np.mean(diffs))
			Stdplus.append(np.std(diffs))
		else:
			Meanneg.append(np.mean(diffs))
			Stdneg.append(np.std(diffs))
		corr=np.correlate(diffs,diffs,'full');
		corr=(corr[(np.floor(np.size(corr)+1)/2.0):-1])
		corr=corr/float(corr[0])
		CORR+=corr
		counts+=1.0

CORR=CORR/counts

print counter

#plt.scatter(Meanfit,Stdfit)


plt.scatter(Meansamp,Meanfit)
plt.plot(Meansamp,Meansamp,c='r')
plt.title('Fit Mean vs Sample Mean ')
plt.show()
plt.scatter(Stdsamp,Stdfit)
plt.plot(Stdsamp,Stdsamp,c='r')
plt.title('Fit Standard Deviation Vs Sample Standard Deviation')
plt.show()

plt.scatter(np.log10(np.abs(Meansamp)),np.log10(Stdsamp),c=Meansamp)
plt.colorbar()
plt.title('Log10 of the Std vs Log10 of the absolute value of the mean')
plt.show()

plt.scatter(np.log10(np.abs(Meanplus)),np.log10(Stdplus),c=(Meanplus))
plt.colorbar()
plt.title('Log10 of the Std vs Log10 of the absolute value of positive means')
plt.show()

plt.scatter(np.log10(np.abs(Meanneg)),np.log10(Stdneg),c=(Meanneg))
plt.colorbar()
plt.title('Log10 of the Std vs Log10 of the absolute value of negative means')
plt.show()
print max(np.log10(np.abs(Meanneg))),min(np.log10(np.abs(Meanneg)))


plt.plot(CORR,c='r')
plt.title('Average correlation function vs time lag')
plt.show()

##differences
difplus= np.array([x for x in Meansamp if x>0 ] , dtype=np.float) #####key formatting thingy
difneg= np.array([x for x in Meansamp if x<0 ] , dtype=np.float) #####key formatting thingy
##correlation plots
counter=0
for i in range(1,np.size(np.array(a).T[1,:]) ):

	#print i
	c= np.array([x for x in a[i][7:-2] if x!='(NA)' ] , dtype=np.int) #####key formatting thingy
	if (c[0]>1500000):
		counter+=1
		diffs=np.diff(np.log(np.array(c, dtype=np.int)))
			#corr=np.correlate(diffs,np.array(c, dtype=np.int),'full');
			#corr=np.correlate(diffs,np.diff(diffs),'full');
		corr=np.correlate(diffs,diffs,'full')
		#corr=np.correlate(((np.array(c, dtype=np.int))),((np.array(c, dtype=np.int))),'full');
		corr=(corr[(np.floor(np.size(corr)-1)/2.0):-1])
		#plt.plot(corr/np.float(corr[0])) ###uncomment for corr plot
		#plt.title('correlation function in differences vs time lag '+a[i][1]+' '+str(c[0]))
		#plt.show()

print counter


##frequency analysis
counter=0
for i in range(1,np.size(np.array(a).T[1,:]) ):
	
	#print i
	c= np.array([x for x in a[i][7:-2] if x!='(NA)' ] , dtype=np.int) #####key formatting thingy
	if (c[0]>1500000):
		counter+=1
		diffs=np.diff((np.array(c, dtype=np.int)))
		#corr=np.correlate(diffs,np.array(c, dtype=np.int),'full');
		#corr=np.correlate(diffs,np.diff(diffs),'full');
		corr=np.correlate(diffs,diffs,'full')
		#corr=np.correlate(((np.array(c, dtype=np.int))),((np.array(c, dtype=np.int))),'full');
		n=len(diffs)
		dt=1
		fft_x = fft(diffs) / n # FFT Normalized
		freq = fftfreq(n, dt) # Recuperamos las frecuencias
		fft_x_shifted = np.fft.fftshift(fft_x)
		#print np.int((np.size(corr)-1)/2.0),np.size(corr),np.size(abs(fft_x_shifted))
		freq_shifted = np.fft.fftshift(freq)
		plt.scatter(np.log10(freq_shifted[(np.floor(np.size(freq_shifted)-1)/2.0):-1]),np.log10(abs(fft_x_shifted)[(np.floor(np.size(freq_shifted)-1)/2.0):-1]))
		#plt.plot(freq_shifted[(np.floor(np.size(freq_shifted)-1)/2.0):-1],np.log10(abs(fft_x_shifted)[(np.floor(np.size(freq_shifted)-1)/2.0):-1]))
		#plt.plot(freq_shifted,np.real(fft_x_shifted))
		#plt.plot(freq_shifted,np.imag(fft_x_shifted))
		#plt.plot(np.log10(freq_shifted[(np.floor(np.size(freq_shifted)-1)/2.0):-1]),-np.log10(freq_shifted[(np.floor(np.size(freq_shifted)-1)/2.0):-1])+np.log10(freq_shifted[(np.floor(np.size(freq_shifted)-1)/2.0)+1])+np.log10(abs(fft_x_shifted)[(np.floor(np.size(freq_shifted)-1)/2.0)]),c='r')
		#plt.title('power spectrum of the differences vs frequency '+a[i][1]+' '+str(c[0]))
		#plt.show()


print counter




a=list(csv.reader(open('BEA_County_Population_1969_2015(1).csv')))




##stability of std deviations analysis
counter=0
for i in range(1,np.size(np.array(a).T[1,:]) ):
	
	
	c= np.array([x for x in a[i][7:-2] if x!='(NA)' ] , dtype=np.int) #####key formatting thingy
	if (c[0]>1500000):
		counter+=1
		diffs=np.diff(np.log(np.array(c, dtype=np.int)))
		sampest=np.array([np.std(diffs[:j]) for j in range(2,np.size(diffs))]) #uncomment for variable size std plot
		#plt.plot(np.sqrt(sampest))
		#plt.title('Standard dev vs time '+a[i][1]+' '+str(c[0]))
		#plt.show()
		#sampest=np.array([np.std(np.random.choice(diffs,15, replace=False)) for j in range(2,np.size(diffs))])
		#plt.plot(np.sqrt(sampest))
		#plt.title('Standard dev vs sample number '+a[i][1]+' '+str(c[0]))
	#plt.show()

print counter




