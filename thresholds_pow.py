import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.sandbox.stats.multicomp import multipletests
import joblib
import time
from joblib import Parallel, delayed
import multiprocessing
## Calculate Score Quantile
def scores_ord_bayes(P_sort,theta,th):
    dens_sort = np.exp((norm.ppf(P_sort) * theta) - (0.5 * (theta ** 2)));
    K = np.size(P_sort);
    score_mat = np.zeros((K-1,K-1));
    score_mat[0,:] = ((np.delete(dens_sort,K-1)*(np.delete(P_sort,K-1)<alpha)*(1+dens_sort[K-1]))+((1+np.delete(dens_sort,K-1))*(P_sort[K-1]<alpha)*dens_sort[K-1]))/4
    for j in 1+np.arange(K-2):
        score_mat[j,0:K-j-1] = (((score_mat[j-1,0:K-j-1]>th[j-1])*dens_sort[0:K-j-1]*np.prod(1+dens_sort[K-j-1:K]))/(2**(j+2)))+(((1+dens_sort[0:K-j-1])/2)*score_mat[j-1,K-j-1]*(score_mat[j-1,K-j-1]>th[j-1]))
    return(score_mat[K-2,0])
## Calculate Decision
def decision_bayes(P,theta,th):
    P_sort = np.sort(P);
    dens_sort = np.exp((norm.ppf(P_sort) * theta) - (0.5 * (theta ** 2)));
    K = np.size(P_sort);
    score_mat = np.zeros((K-1,K-1));
    score_mat[0,:] = ((np.delete(dens_sort,K-1)*(np.delete(P_sort,K-1)<alpha)*(1+dens_sort[K-1]))+((1+np.delete(dens_sort,K-1))*(P_sort[K-1]<alpha)*dens_sort[K-1]))/4
    for j in 1+np.arange(K-2):
        score_mat[j,0:K-j-1] = (((score_mat[j-1,0:K-j-1]>th[j-1])*dens_sort[0:K-j-1]*np.prod(1+dens_sort[K-j-1:K]))/(2**(j+2)))+(((1+dens_sort[0:K-j-1])/2)*score_mat[j-1,K-j-1]*(score_mat[j-1,K-j-1]>th[j-1]));
    D_sort= np.append(1*(np.rot90(score_mat).diagonal()>th)[::-1],int(P_sort[K-1]<alpha));
    D_sort = np.cumprod(D_sort);
    return(D_sort[np.argsort(np.argsort(P))])
## Calculate Score Quantile
def scores_ord_pi1(P_sort,theta,th):
    dens_sort = np.exp((norm.ppf(P_sort) * theta) - (0.5 * (theta ** 2)));
    K = np.size(P_sort);
    score_mat = np.zeros((K-1,K-1));
    score_mat[0,:] = ((np.delete(dens_sort,K-1)*(np.delete(P_sort,K-1)<alpha))+((P_sort[K-1]<alpha)*dens_sort[K-1]))
    for j in 1+np.arange(K-2):
        score_mat[j,0:K-j-1] = ((score_mat[j-1,0:K-j-1]>th[j-1])*dens_sort[0:K-j-1])+(score_mat[j-1,K-j-1]*(score_mat[j-1,K-j-1]>th[j-1]))
    return(score_mat[K-2,0])
## Calculate Decision
def decision_pi1(P,theta,th):
    P_sort = np.sort(P);
    dens_sort = np.exp((norm.ppf(P_sort) * theta) - (0.5 * (theta ** 2)));
    K = np.size(P_sort);
    score_mat = np.zeros((K-1,K-1));
    score_mat[0,:] = ((np.delete(dens_sort,K-1)*(np.delete(P_sort,K-1)<alpha))+((P_sort[K-1]<alpha)*dens_sort[K-1]))
    for j in 1+np.arange(K-2):
        score_mat[j,0:K-j-1] = ((score_mat[j-1,0:K-j-1]>th[j-1])*dens_sort[0:K-j-1])+(score_mat[j-1,K-j-1]*(score_mat[j-1,K-j-1]>th[j-1]))
    D_sort= np.append(1*(np.rot90(score_mat).diagonal()>th)[::-1],int(P_sort[K-1]<alpha));
    D_sort = np.cumprod(D_sort);
    return(D_sort[np.argsort(np.argsort(P))])
## Calculate Last Step Score Quantile
def scores_hom_bayes(P,theta):
    P_sort = np.sort(P);
    _, p_values, _, _ = multipletests(np.delete(P_sort,0), alpha=alpha, method='hommel')
    score = np.sum(np.append((multipletests(np.delete(P_sort,1), alpha=alpha, method='hommel')[0]*1)[0],1*(p_values < alpha))*(np.apply_along_axis(np.prod,1,-np.eye(np.size(P_sort))+1+np.exp((norm.ppf(P_sort) * theta) - (0.5 * (theta ** 2))))/(2**np.size(P)))); #pi_bayes
    return(score)
# Calculate Dcision
def decisions_hom_bayes(P,theta,th):
#     pass;
    P_sort = np.sort(P);
    _, p_values, _, _ = multipletests(np.delete(P_sort,0), alpha=alpha, method='hommel')
    score_ind = np.sum(np.append((multipletests(np.delete(P_sort,1), alpha=alpha, method='hommel')[0]*1)[0],1*(p_values < alpha))*(np.apply_along_axis(np.prod,1,-np.eye(np.size(P_sort))+1+np.exp((norm.ppf(P_sort) * theta) - (0.5 * (theta ** 2))))/(2**np.size(P)))); #pi_bayes
    decisions = np.append((multipletests(np.delete(P_sort,1), alpha=alpha, method='hommel')[0]*1)[0],1*(p_values < alpha))*1*(score_ind>th)
    return(decisions[np.argsort(np.argsort(P))])
## Calculate Last Step Score Quantile
def scores_hom_pi1(P,theta):
    P_sort = np.sort(P);
    _, p_values, _, _ = multipletests(np.delete(P_sort,0), alpha=alpha, method='hommel')
    score = np.sum(np.append((multipletests(np.delete(P_sort,1), alpha=alpha, method='hommel')[0]*1)[0],1*(p_values < alpha))*(np.exp((norm.ppf(P_sort) * theta) - (0.5 * (theta ** 2))))); #pi_1
    return(score)
# Calculate Dcision
def decisions_hom_pi1(P,theta,th):
#     pass;
    P_sort = np.sort(P);
    _, p_values, _, _ = multipletests(np.delete(P_sort,0), alpha=alpha, method='hommel')
    score_ind = np.sum(np.append((multipletests(np.delete(P_sort,1), alpha=alpha, method='hommel')[0]*1)[0],1*(p_values < alpha))*(np.exp((norm.ppf(P_sort) * theta) - (0.5 * (theta ** 2))))); #pi_1
    decisions = np.append((multipletests(np.delete(P_sort,1), alpha=alpha, method='hommel')[0]*1)[0],1*(p_values < alpha))*1*(score_ind>th)
    return(decisions[np.argsort(np.argsort(P))])

#################
target_power = 0.9;
alpha= 0.05; K=10; theta = norm.ppf(alpha/K)-norm.ppf(target_power); B=2*(10**7); 
##################
np.random.seed(1024)
start_time = time.time()
####### Threshold with Improved Hommel bayes #######
U = np.random.uniform(size=(B, K));
U = U[np.apply_along_axis(lambda p: np.min(p)<alpha , axis=1, arr=U)];
s = np.array(Parallel(n_jobs=-1)(delayed(scores_hom_bayes)(p, theta) for p in U))
thhom_bayes = np.quantile(np.concatenate([np.zeros(B - len(s)), s]), 1 - alpha)

print('pass')
####### Threshold with Improved Hommel pi_1 #######
U = np.random.uniform(size=(B, K));
U = U[np.apply_along_axis(lambda p: np.min(p)<alpha , axis=1, arr=U)];
s = np.array(Parallel(n_jobs=-1)(delayed(scores_hom_pi1)(p, theta) for p in U))
thhom_pi1 = np.quantile(np.concatenate([np.zeros(B - len(s)), s]), 1 - alpha)

print('pass')
######### Threshold with Bottom-up bayes ##########
th_bayes = []
for k in np.arange(2,K+1):
    print(k)
    U = np.random.uniform(size=(int(B*10), k));
    U = U[np.apply_along_axis(lambda p: np.min(p)<alpha , axis=1, arr=U)];
    U = np.apply_along_axis(lambda p: np.sort(p) , axis=1, arr=U)
    s = np.array(Parallel(n_jobs=-1)(delayed(scores_ord_bayes)(p, theta, th_bayes) for p in U))
    th_bayes = np.append(th_bayes,np.quantile(np.concatenate([np.zeros(int(B*10) - len(s)), s]), 1 - alpha));

print('pass')
######### Threshold with Bottom-up pi1 ##########
th_pi1 = []
for k in np.arange(2,K+1):
    print(k)
    U = np.random.uniform(size=(B, k));
    U = U[np.apply_along_axis(lambda p: np.min(p)<alpha , axis=1, arr=U)];
    U = np.apply_along_axis(lambda p: np.sort(p) , axis=1, arr=U)
    s = np.array(Parallel(n_jobs=-1)(delayed(scores_ord_pi1)(p, theta, th_pi1) for p in U))
    th_pi1 = np.append(th_pi1,np.quantile(np.concatenate([np.zeros(B - len(s)), s]), 1 - alpha));
 

print('pass')

end_time = time.time()
elapsed_time = end_time - start_time

print("Elapsed time:", elapsed_time, "seconds")    
np.save('thresholds_pow_90_10.npy',np.array([thhom_bayes,thhom_pi1,th_bayes,th_pi1],dtype=object))
