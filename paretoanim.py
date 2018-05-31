import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
from scipy import stats

np.set_printoptions(threshold=np.nan)


def find_first(item, vec):
	"""return the index of the first occurence of item in vec"""
	for i in range(np.size(vec)):
		if vec[i]== item:
			return i
	return -1

a=list(csv.reader(open('BEA_County_Population_1969_2015(1).csv')))
#a=filter(lambda l: l != '(NA)', a)
['100000000' if x=='(NA)' else x for x in a]
lag=46
dats=(np.array(a).T[7+lag,1:]).astype(int)
b=dats
ubound=11000000 ##to avoid overflow values

#print np.size(b)
for i in range(np.size(dats)):
	if (dats[i]>ubound):
		b=np.delete(b,find_first(dats[i], b))
		#print np.size(b)

#print np.size(b),'perr'
bins = np.arange(np.floor(b.min()),np.ceil(b.max()))
#bins = np.linspace(np.floor(b.min()),np.ceil(b.max()),100)
values, base = np.histogram(b, bins=bins, density=1)
plt.loglog((bins[:-1]),(1-np.cumsum(values)))
plt.title('plot cummulative distribution vs population '+str(1969+lag))
plt.show()
print np.sum(values)

plt.plot((bins[:-1]),(1-np.cumsum(values)))
plt.title('plot cummulative distribution vs population '+str(1969+lag))
plt.show()
print np.sum(values)


listb=np.array(sorted(list(b)))
listb=listb[::-1]
plt.scatter(   np.log10(listb) , np.log10( np.arange(1,1+np.size(listb)) )   )
plt.title('scatter log log rank of the city vs population '+str(1969+lag))

plt.show()



Counties=[]
a=list(csv.reader(open('BEA_County_Population_1969_2015(1).csv')))
for i in range(1,np.size(np.array(a).T[1,:]) ):
	
	#print i
	c= np.array([x for x in a[i][7:-2] if x!='(NA)' ] , dtype=np.int) #####key formatting thingy
	if (c[-2]>10000):
		Counties.append(c[-3])

listb=np.array(sorted(Counties))
listb=listb[::-1]
x=np.log10(listb[:100])
y=np.log10( np.arange(1,1+100) )
plt.scatter(  x  ,  y  )
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
plt.plot(x,slope*x+intercept)
plt.title('scatter log log rank of the city vs population 2006')

plt.show()
print slope


listb=np.array(sorted(list(b)))
listb=listb[::-1]
x=np.log10(listb[:150])
y=np.log10( np.arange(1,1+150) )
plt.scatter(  x  ,  y  )
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
plt.plot(x,slope*x+intercept)
plt.title('top 150 scatter log log rank of the city vs population '+str(1969+lag))

plt.show()

print slope

totalpop=[]
meanpop=[]
tophund=[]
for lagg in range(0,45):
	dats=(np.array(a).T[7+lagg,1:]).astype(int)
	b=dats
	ubound=11000000 ##to avoid overflow values
	for i in range(np.size(dats)):
		if (abs(dats[i])>ubound):
			b=np.delete(b,find_first(dats[i], b))
	tophund.append(np.sum((np.array(sorted(list(b)))[::-1])[:400]))
	totalpop.append(np.sum(b))
	meanpop.append(np.mean(b))

plt.plot(totalpop)
plt.title('total population vs time')
plt.show()

plt.plot(meanpop)
plt.title('mean population vs time')
plt.show()


plt.plot(tophund)
plt.title('population in th top 400 largest counties vs time')
plt.show()
