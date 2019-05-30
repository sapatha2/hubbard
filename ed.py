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
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

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

if __name__=='__main__':
  #eig_vs_U.pdf
  '''
  Sz=0
  energies = []
  for U in np.arange(0,12.1,0.1):
    norb = 4
    nelec = (2 + Sz, 2 - Sz)
    h1,eri = hamiltonian(U)
    e,ci = fci.direct_uhf.kernel(h1,eri,norb=norb,nelec=nelec,nroots=10**5)
    energies.append(e-min(e))
  energies=np.array(energies).T

  fig, ax = plt.subplots(nrows=1,ncols=2)
  Us = np.arange(0,12.1,0.1)
  for x in energies:
    ax[0].plot(Us,x,'k')
  ax[0].set_xlabel('U/t')
  
  beta = np.arange(0,3.25,0.25)
  temp_matrix = np.zeros((300,len(beta)))
  for b in beta:
    temp_matrix[:,int(b*4)] = np.exp(-np.linspace(30,0,300)*b)
  ax[1].imshow(temp_matrix,vmin=0,vmax=1,cmap=plt.cm.pink)
  ax[1].set_xlabel('1/kT')
  ax[0].set_ylabel('E - E_0')
  ax[1].set_yticks(np.arange(0,350,50))
  ax[1].set_yticklabels(np.arange(30,-5,-5))
  
  ax[1].set_xticks([0,4,8,12])
  ax[1].set_xticklabels([0,1,2,3])
  ax[1].set_aspect(0.1)
  plt.show()
  exit(0)
  '''

  #eff_eig.pdf
  Sz=0
  betas = np.arange(0,3.25,0.25)
  Us = np.arange(13)
  R2_J = np.zeros((len(Us),len(betas)))
  R2_U = np.zeros((len(Us),len(betas)))
  R2_t = np.zeros((len(Us),len(betas)))
  v_J = np.zeros((len(Us),len(betas)))
  v_U = np.zeros((len(Us),len(betas)))
  v_t = np.zeros((len(Us),len(betas)))
  
  Uz = 0
  for U in Us:
    norb = 4
    nelec = (2 + Sz, 2 - Sz)
    h1,eri = hamiltonian(U)
    e,ci = fci.direct_uhf.kernel(h1,eri,norb=norb,nelec=nelec,nroots=10**5)
    prop = build_dm(e,ci,norb=norb,nelec=nelec)

    #J, strong coupling limit
    y=prop['energy']
    z=0
    for beta in betas:
      print(U,beta)
      X=prop['dJ']
      X=sm.add_constant(X)
      w = np.exp(-beta*(y-min(y)))
      ols=sm.WLS(y,X,weights=w).fit()
      R2_J[Uz,z] = ols.rsquared
      v_J[Uz,z] = ols.params[1]
   
      X=prop['dt']
      X=sm.add_constant(X)
      w = np.exp(-beta*(y-min(y)))
      ols=sm.WLS(y,X,weights=w).fit()
      R2_t[Uz,z] = ols.rsquared
      v_t[Uz,z] = ols.params[1]
    
      X=prop['dU']
      X=sm.add_constant(X)
      w = np.exp(-beta*(y-min(y)))
      ols=sm.WLS(y,X,weights=w).fit()
      R2_U[Uz,z] = ols.rsquared
      v_U[Uz,z] = ols.params[1]
      z+=1
    Uz+=1
  
  #Plot results
  fig, axes = plt.subplots(nrows=4,ncols=2,sharey=True,sharex=True,figsize=(6,15))
  ax = axes[0,0]
  ax.matshow(R2_J.T,vmax=1,vmin=0.5,cmap=plt.cm.Blues)
  ax.set_xlabel('U')
  ax.set_ylabel('1/kT')
  ax.set_title('Model: J')

  ax = axes[0,1]
  v_J = np.ma.masked_where((R2_J < 0.4),v_J)
  ax.matshow(v_J.T,vmin=-0.25,vmax=0.25,cmap=plt.cm.RdGy)
  ax.set_xlabel('U')
  ax.set_ylabel('1/kT')
  ax.set_title('Model: J (-0.25,0.25)')

  ax = axes[1,0]
  ax.matshow(R2_U.T,vmax=1,vmin=0.5,cmap=plt.cm.Greens)
  ax.set_xlabel('U')
  ax.set_ylabel('1/kT')
  ax.set_title('Model: U')

  ax = axes[1,1]
  v_U = np.ma.masked_where((R2_U < 0.4),v_U)
  ax.matshow(v_U.T,vmin=-10,vmax=10,cmap=plt.cm.RdGy)
  ax.set_xlabel('U')
  ax.set_ylabel('1/kT')
  ax.set_title('Model: U (-10, 10)')

  ax = axes[2,0]
  ax.matshow(R2_t.T,vmax=1,vmin=0.5,cmap=plt.cm.Reds)
  ax.set_xlabel('U')
  ax.set_ylabel('1/kT')
  ax.set_title('Model: t')

  ax = axes[2,1]
  v_t = np.ma.masked_where((R2_t < 0.4),v_t)
  ax.matshow(-v_t.T,vmin=-1,vmax=1,cmap=plt.cm.RdGy)
  ax.set_xlabel('U')
  ax.set_ylabel('1/kT')
  ax.set_title('Model: t (1, -1)')
  
  ax = axes[3,0]
  plot_mat = np.ma.masked_where((R2_J < R2_t)|(R2_J < R2_U),R2_J)
  im = ax.imshow(plot_mat.T,vmax=1,vmin=0.5,cmap=plt.cm.Blues)
  plot_mat = np.ma.masked_where((R2_U < R2_t)|(R2_U < R2_J),R2_U)
  im = ax.imshow(plot_mat.T,vmax=1,vmin=0.5,cmap=plt.cm.Greens)
  plot_mat = np.ma.masked_where((R2_t < R2_U)|(R2_t < R2_J),R2_t)
  im = ax.imshow(plot_mat.T,vmax=1,vmin=0.5,cmap=plt.cm.Reds)

  ax.set_xlabel('U')
  ax.set_ylabel('1/kT')
  ax.set_title('Model: Best of J (b), U(g), t (r)')
  ax.xaxis.tick_bottom()
  ax.set_yticks([0,4,8,12])
  ax.set_yticklabels([0,1,2,3])
  
  plt.suptitle('WLS R2 and Param values on eigenstates')
  plt.savefig('eff_eig.pdf',bbox_inches='tight')
  exit(0)
