from ed import hamiltonian, build_dm, sub_sampling
from pyscf import gto, ao2mo, fci
import numpy as np 
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt 

df = None
for U in [0,2,5,8,12]: 
  Sz = 0
  norb = 4
  nelec = (2 + Sz, 2 - Sz)
  method = 'push'
  h1,eri = hamiltonian(U)
  e,ci = fci.direct_uhf.kernel(h1,eri,norb=norb,nelec=nelec,nroots=10**5)
  prop = build_dm(e,ci,norb,nelec)
  prop['energy'] -= min(prop['energy'])
  
  for beta in np.arange(0,0.5,0.1):
    size=np.exp(-beta*prop['energy'])
    size/=np.sum(size)
    
    small = pd.DataFrame({'size': size*40/max(size)})
    small['beta'] = np.around(beta,2)
    small['U'] = U
    small = pd.concat((prop,small),axis=1)

    if(df is None): df = small
    else: df = pd.concat((df,small),axis=0)

g = sns.FacetGrid(df, col='U', row='beta')
g = g.map(plt.scatter, 'dt', 'dU', 'size')
plt.savefig('eff_eig_full.pdf',bbox_inches='tight')
plt.close()
