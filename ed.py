#4 site Hubbard model @ half filling has 70 states
#Nelec          Nstates           N degen
#0 2 0 2        6                 1
#1 1 1 1        1                 16
#1 2 1 0        12                4
#Total is 6*1 + 1*16 + 12*4 = 70

import numpy as np
from pyscf import gto, ao2mo, fci
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
import statsmodels.api as sm
from sklearn.metrics import r2_score
def hamiltonian(U):
  '''
  input:
    U - U/t
  output:
    h1 and eri for ED
  '''

  #Hamiltonian (2x2)
  #  .[0]   .[1]
  #  .[3]   .[2]
  norb=4
  h1 = np.zeros((norb,norb))
  eri = np.zeros((3,norb,norb,norb,norb))

  for i in range(norb-1):
    h1[i,i+1] = h1[i+1,i]= -2.
  h1[norb-1,0]= -2.
  h1[0,norb-1]= -2.
  for i in range(4):
    eri[1][i,i,i,i] = U
  return [h1,h1], eri

def build_dm(e,ci,norb,nelec):
  '''
  Given a list of ci and energy,
  construct a data frame
  with DM elements and energy
  '''

  mol = gto.Mole()
  cis = fci.direct_uhf.FCISolver(mol)
  
  dt = []
  dU = []
  dJ = []
  for p in range(len(ci)):
    dm = cis.make_rdm12s(ci[p],norb,nelec)
    dm1 = np.array(dm[0])
    dm2 = np.array(dm[1])
    dm1 = dm1[0]+dm1[1]

    val1 = 0
    for i in range(norb-1):
      val1 += 2*(dm1[i,i+1] + dm1[i+1,i])
    val1 += 2*(dm1[norb-1,0])
    val1 += 2*(dm1[0,norb-1])
    dt.append(val1)
    
    val2 = 0
    for i in range(4):
      val2 += dm2[1][i,i,i,i]
    dU.append(val2)

    val3 = 0
    for i in range(norb):
      j = np.mod(i+1,norb)
      ###### ---------------- DONT COPY THIS TO BIGGER CELLS ------------- ##########
      val3 +=  4*(0.25*dm2[0][i,i,j,j] +  #4 is the correct factor
                  0.25*dm2[2][i,i,j,j] -  #to get the 4t^2/U term
                  0.25*dm2[1][i,i,j,j] - 
                  0.25*dm2[1][j,j,i,i] - 
                  0.50*dm2[1][j,i,i,j] - 
                  0.50*dm2[1][i,j,j,i])
      ###### ---------------- DONT COPY THIS TO BIGGER CELLS ------------- ##########
    dJ.append(val3)
  return pd.DataFrame({'energy':e,'dt':dt,'dU':dU,'dJ':dJ})

def sub_sampling(e,ci,subset,N,norb,nelec,method='unif',beta=None,dt=None):
  #Generate samples 
  '''
  input:
    e - list of energies from ED
    ci - list of CI coefficients from ED
    subset - list of CIs to sample the span of 
    N - number of samples to draw
    method - method for sampling
  '''

  if(method=='unif'):
    return unif(e,ci,subset,N,norb,nelec)
  elif(method=='boltz'):
    return boltz(e,ci,subset,N,norb,nelec,beta,dt)
  else:
    return -1

def unif(e,ci,subset,N,norb,nelec):
  '''
  input:
    e - list of energies from ED
    ci - list of CI coefficients from ED
    subset - list of CIs to sample the span of 
    N - number of samples to draw
  '''
  
  print("Uniform sampling: "+str(N))
  basis = np.array(ci)[subset]
  e_basis = np.array(e)[subset]
  
  coeffs = np.random.normal(size=(N,subset.shape[0]))
  coeffs = preprocessing.normalize(coeffs, norm='l2')

  sampled_ci = np.einsum('ij,jkl->ikl',coeffs,basis)
  sampled_e = np.dot(coeffs**2,e_basis)
  
  return build_dm(sampled_e,sampled_ci,norb,nelec)

def boltz(e,ci,subset,N,norb,nelec,beta,dt):
  '''
  input:
    e - list of energies from ED
    ci - list of CI coefficients from ED
    subset - list of CIs to sample the span of 
    N - number of samples to draw
    beta - temperature for boltzmann sample
    dt - timestep, adjusted so acceptance is 50%
  '''
  
  print("Boltzmann sampling: "+str(N))
  basis = np.array(ci)[subset]
  e_basis = np.array(e)[subset]
  coeffs = []
  curr = np.zeros(subset.shape[0])
  curr[0] = 1
  e_curr = np.dot(curr**2,e_basis)/np.dot(curr,curr)
  coeffs.append(curr)

  acc_ratio = 0
  for i in range(N): 
    print(e_curr)
    prop = curr + dt*np.random.normal(size=(subset.shape[0])) #MC move 
    e_prop = np.dot(prop**2,e_basis)/np.dot(prop,prop)
    acc = (np.exp(-beta*(e_prop - e_curr)) > np.random.uniform(size=1))
    if(acc): 
      coeffs.append(prop)
      curr = prop
      e_curr = np.dot(curr**2,e_basis)/np.dot(curr,curr)
      acc_ratio += 1
    else: 
      coeffs.append(curr)

  coeffs = preprocessing.normalize(coeffs, norm='l2')
  sampled_ci = np.einsum('ij,jkl->ikl',coeffs,basis)
  sampled_e = np.dot(coeffs**2,e_basis)

  print("Acceptance ratio: "+str(acc_ratio/N))
  return build_dm(sampled_e,sampled_ci,norb,nelec)

#t or J model only, WLS on eigenstates
Sz=0
R2_J = np.zeros((20,10))
R2_t = np.zeros((20,10))
for U in np.arange(20):
  norb = 4
  nelec = (2 + Sz, 2 - Sz)
  h1,eri = hamiltonian(U)
  e,ci = fci.direct_uhf.kernel(h1,eri,norb=norb,nelec=nelec,nroots=10**5)
  prop = build_dm(e,ci,norb=norb,nelec=nelec)

  #J, strong coupling limit
  X=prop['dJ']
  y=prop['energy']
  X=sm.add_constant(X)
  for beta in np.arange(10):
    w = np.exp(-beta*(y-min(y)))
    ols=sm.WLS(y,X,weights=w).fit()
    R2_J[U,beta] = ols.rsquared

  #T, weak coupling limit
  X=prop['dt']
  y=prop['energy']
  X=sm.add_constant(X)
  for beta in np.arange(10):
    w = np.exp(-beta*(y-min(y)))
    ols=sm.WLS(y,X,weights=w).fit()
    R2_t[U,beta] = ols.rsquared

#Plot results
fig, axes = plt.subplots(nrows=3,ncols=1,sharey=True,sharex=True,figsize=(3,9))
ax = axes[0]
ax.matshow(R2_J.T,vmax=1,vmin=-1,cmap=plt.cm.bwr)
ax.set_xlabel('U')
ax.set_ylabel('1/kT')
ax.set_title('Model: J')

ax = axes[1]
ax.matshow(-1*R2_t.T,vmax=1,vmin=-1,cmap=plt.cm.bwr)
ax.set_xlabel('U')
ax.set_ylabel('1/kT')
ax.set_title('Model: t')

ax = axes[2]
mat = np.maximum(R2_J,R2_t)
ind = np.where(mat == R2_t)
mat[ind]*=-1
ax.matshow(mat.T,vmax=1,vmin=-1,cmap=plt.cm.bwr)
ax.set_xlabel('U')
ax.set_ylabel('1/kT')
ax.set_title('Model: Best of t (b) or J (r)')
ax.xaxis.tick_bottom()

plt.suptitle('WLS on eigenstates')
plt.savefig('eff_eig.pdf',bbox_inches='tight')
exit(0)
