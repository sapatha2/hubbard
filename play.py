from ed import hamiltonian, build_dm, sub_sampling
from pyscf import gto, ao2mo, fci
import numpy as np 
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt 

from itertools import chain, combinations
def powerset(iterable):
  "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
  s = list(iterable)
  return chain.from_iterable(combinations(s, r) for r in range(2,len(s)+1))

Sz=0
U = 8
norb = 4
nelec = (2 + Sz, 2 - Sz)
method = 'unif'
h1,eri = hamiltonian(U)
e,ci = fci.direct_uhf.kernel(h1,eri,norb=norb,nelec=nelec,nroots=10**5)
prop = build_dm(e,ci,norb,nelec)

pset = powerset(np.arange(len(e)))
for x in pset:
  print(x)
exit(0)

samples = sub_sampling(e,ci,np.arange(len(e)),1000,norb,nelec,method=method)

prop['type']='ED'
samples['type']=method

df = pd.concat((prop,samples),axis=0)
sns.pairplot(df,hue='type',markers=['o','.'])
plt.show()
