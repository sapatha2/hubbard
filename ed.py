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
  for i in range(len(ci)):
    dm = cis.make_rdm12s(ci[i],norb,nelec)
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
      val3 += 2.*(0.25*dm2[0][i,i,j,j] + 
                  0.25*dm2[0][j,j,i,i] +
                  0.25*dm2[2][i,i,j,j] + 
                  0.25*dm2[2][j,j,i,i] - 
                  0.25*dm2[1][i,i,j,j] - 
                  0.25*dm2[1][j,j,i,i] - 
                  0.50*dm2[1][j,i,i,j] - 
                  0.50*dm2[1][i,j,j,i])
    dJ.append(val3)

  return pd.DataFrame({'energy':e,'dt':dt,'dU':dU,'dJ':dJ})

def sub_sampling(e,ci,subset,N,norb,nelec,method='unif'):
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

#FCI solver
Sz=0
U=8
norb = 4
nelec = (2 + Sz, 2 - Sz)
h1,eri = hamiltonian(U)
e,ci = fci.direct_uhf.kernel(h1,eri,norb=norb,nelec=nelec,nroots=10**5)

#Plot dt, dU, energy for eigenstates/samples
method = 'unif'
prop = build_dm(e,ci,norb=norb,nelec=nelec)
sns.pairplot(prop)
plt.show()
#samples = sub_sampling(e,ci,np.arange(6),1000,norb=norb,nelec=nelec,method=method)

'''
prop['type']='ED'
samples['type']=method
df = pd.concat((samples,prop.iloc[:6]),axis=0)
sns.pairplot(df,hue='type',markers=['.','o'])
plt.show()
'''