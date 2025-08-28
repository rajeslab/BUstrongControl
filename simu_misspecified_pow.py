import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.sandbox.stats.multicomp import multipletests
import joblib
import time
from joblib import Parallel, delayed
import multiprocessing
from random import randint
import rpy2
import rpy2.robjects as robjects
#from rpy2.robjects import pandas2ri
from rpy2.robjects.vectors import FloatVector
# Load the required R package
robjects.r('''
library(elitism)
''')
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

def GouConsonantRej(p_values,level):
    # Convert NumPy array to an R numeric vector
    p_values_r = robjects.FloatVector(p_values)
    
    # Assign to R environment
    robjects.r.assign("p_values", p_values_r)
    
    # Run the R function and capture the output
    mtp_output = robjects.r('''
    mtp_results <- elitism::p.adjust(p_values, method = "gtxr")
    mtp_results  # Return the result back to Python
    ''')
    return(1*(np.array(mtp_output)<level))

################ First Situation ###############
################# Thresholds #################
x = np.load('thresholds_pow_50_10.npy',allow_pickle=True)
th_bayes = x[2]
th_pi1 = x[3]
thhom_bayes = x[0]
thhom_pi1 = x[1]

all_list=[];
all_result = [];

reps = int(1e6);
theta_process = norm.ppf(alpha/K)-norm.ppf(0.5);
######################## Parameters #########################
alpha= 0.05;K=10;theta_simu = norm.ppf(alpha/K)-norm.ppf(0.5);

l = [];
#np.random.seed(1011)
start_time = time.time()
def process_rep(rep):
    np.random.seed(rep);
    h = np.concatenate([np.ones(K0), np.zeros(max(K - K0,0))])
    if(K0==K+1):
        h= np.random.binomial(1,0.5,size = K)
    x = theta_simu * h + np.random.normal(0, 1, K)
    p = norm.cdf(x, 0, 1)
    H = np.zeros(K)
    H[:] = h
    D_bu_bayes = np.zeros(K)
    D_bu_pi1 = np.zeros(K)
    D_imphom_bayes = np.zeros(K)
    D_imphom_pi1 = np.zeros(K)
    D_hom = np.zeros(K)
    D_gou = np.zeros(K)
    if np.min(p) > alpha:
        pass  
    else:
        D_bu_bayes = decision_bayes(p, theta_process, th_bayes);
        D_bu_pi1 = decision_pi1(p, theta_process, th_pi1);
        D_imphom_bayes = decisions_hom_bayes(p, theta_process, thhom_bayes)
        D_imphom_pi1 = decisions_hom_pi1(p, theta_process, thhom_pi1)
        _, p_values, _, _ = multipletests(p, alpha=alpha, method='hommel')
        D_hom = np.asarray(p_values <= alpha, dtype=int)
        D_gou = GouConsonantRej(p,alpha)
    result = [H,D_bu_bayes,D_bu_pi1,D_imphom_bayes,D_imphom_pi1,D_hom,D_gou]
    return result
Result = np.empty((K+1,13))
for K0 in np.delete(np.arange(K+2),0):
    print(K0)
    # Arrays to store results
    if __name__ == "__main__":
        num_processes = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(process_rep, range(reps))
        results_array = np.array(results)
        del(results)

    l.append(results_array)
    # Arrays to store results
    H = np.zeros((reps, K))
    D_bu_bayes = np.zeros((reps, K))
    D_bu_pi1 = np.zeros((reps, K))
    D_imphom_bayes = np.zeros((reps, K))
    D_imphom_pi1 = np.zeros((reps, K))
    D_hom = np.zeros((reps, K))
    D_gou = np.zeros((reps, K))
    ##################
    for rep in range(reps):
        H[rep]= results_array[rep][0]
        D_bu_bayes[rep]= results_array[rep][1]
        D_bu_pi1[rep]= results_array[rep][2]
        D_imphom_bayes[rep]= results_array[rep][3]
        D_imphom_pi1[rep]= results_array[rep][4]
        D_hom[rep]= results_array[rep][5]
        D_gou[rep]= results_array[rep][6]
    
    Result[K0-1,:]    = np.array([int(K0),np.mean(np.sum(D_bu_bayes * H, axis=1) / max(1, K0)), np.mean(np.sum(D_bu_bayes * (1 - H), axis=1) > 0),
                                np.mean(np.sum(D_bu_pi1 * H, axis=1) / max(1, K0)),np.mean(np.sum(D_bu_pi1 * (1 - H), axis=1) > 0),
                                np.mean(np.sum(D_imphom_bayes * H, axis=1) / max(1, K0)), np.mean(np.sum(D_imphom_bayes * (1 - H), axis=1) > 0),
                                np.mean(np.sum(D_imphom_pi1 * H, axis=1) / max(1, K0)), np.mean(np.sum(D_imphom_pi1 * (1 - H), axis=1) > 0),
                            np.mean(np.sum(D_hom * H, axis=1) / max(1, K0)), np.mean(np.sum(D_hom * (1 - H), axis=1) > 0),
                                 np.mean(np.sum(D_gou * H, axis=1) / max(1, K0)), np.mean(np.sum(D_gou * (1 - H), axis=1) > 0)])


end_time = time.time()
elapsed_time = end_time - start_time

print("Elapsed time:", elapsed_time, "seconds")

Result[K0-1,:] = Result[K0-1,:]*np.array([(K/2)/K0,max(1, K0)/(K/2),1,max(1, K0)/(K/2),1,max(1, K0)/(K/2),1,max(1, K0)/(K/2),1,max(1, K0)/(K/2),1,max(1, K0)/(K/2),1]);
#np.save('results_array_bool_25_15_5.npy',np.array(l, dtype=bool))
all_list.append(np.array(l, dtype=bool));
all_result.append(Result);
# df = pd.DataFrame(Result)
# df.to_csv('all_result_25_15_5.csv', index=False)


################ Second Situation ###############

######################## Parameters #########################
alpha= 0.05;theta_simu = norm.ppf(alpha/K)-norm.ppf(0.7);
#K=5;
#theta_process = -1.5;

l = [];
#np.random.seed(1011)
start_time = time.time()
Result = np.empty((K+1,13))
for K0 in np.delete(np.arange(K+2),0):
    print(K0)
    # Arrays to store results
    if __name__ == "__main__":
        num_processes = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(process_rep, range(reps))
        results_array = np.array(results)
        del(results)

    l.append(results_array)
    # Arrays to store results
    H = np.zeros((reps, K))
    D_bu_bayes = np.zeros((reps, K))
    D_bu_pi1 = np.zeros((reps, K))
    D_imphom_bayes = np.zeros((reps, K))
    D_imphom_pi1 = np.zeros((reps, K))
    D_hom = np.zeros((reps, K))
    D_gou = np.zeros((reps, K))
    ##################
    for rep in range(reps):
        H[rep]= results_array[rep][0]
        D_bu_bayes[rep]= results_array[rep][1]
        D_bu_pi1[rep]= results_array[rep][2]
        D_imphom_bayes[rep]= results_array[rep][3]
        D_imphom_pi1[rep]= results_array[rep][4]
        D_hom[rep]= results_array[rep][5]
        D_gou[rep]= results_array[rep][6]
    
    Result[K0-1,:]    = np.array([int(K0),np.mean(np.sum(D_bu_bayes * H, axis=1) / max(1, K0)), np.mean(np.sum(D_bu_bayes * (1 - H), axis=1) > 0),
                                np.mean(np.sum(D_bu_pi1 * H, axis=1) / max(1, K0)),np.mean(np.sum(D_bu_pi1 * (1 - H), axis=1) > 0),
                                np.mean(np.sum(D_imphom_bayes * H, axis=1) / max(1, K0)), np.mean(np.sum(D_imphom_bayes * (1 - H), axis=1) > 0),
                                np.mean(np.sum(D_imphom_pi1 * H, axis=1) / max(1, K0)), np.mean(np.sum(D_imphom_pi1 * (1 - H), axis=1) > 0),
                            np.mean(np.sum(D_hom * H, axis=1) / max(1, K0)), np.mean(np.sum(D_hom * (1 - H), axis=1) > 0),
                                 np.mean(np.sum(D_gou * H, axis=1) / max(1, K0)), np.mean(np.sum(D_gou * (1 - H), axis=1) > 0)])


end_time = time.time()
elapsed_time = end_time - start_time

print("Elapsed time:", elapsed_time, "seconds")

Result[K0-1,:] = Result[K0-1,:]*np.array([(K/2)/K0,max(1, K0)/(K/2),1,max(1, K0)/(K/2),1,max(1, K0)/(K/2),1,max(1, K0)/(K/2),1,max(1, K0)/(K/2),1,max(1, K0)/(K/2),1]);
#np.save('results_array_bool_25_15_5.npy',np.array(l, dtype=bool))
all_list.append(np.array(l, dtype=bool));
all_result.append(Result);
# df = pd.DataFrame(Result)
# df.to_csv('all_result_25_15_5.csv', index=False)


################ Third Situation ###############

######################## Parameters #########################
alpha= 0.05;theta_simu = norm.ppf(alpha/K)-norm.ppf(0.9);
#K=5;
#theta_process = -1.5;

l = [];
#np.random.seed(1011)
start_time = time.time()
Result = np.empty((K+1,13))
for K0 in np.delete(np.arange(K+2),0):
    print(K0)
    # Arrays to store results
    if __name__ == "__main__":
        num_processes = multiprocessing.cpu_count()
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(process_rep, range(reps))
        results_array = np.array(results)
        del(results)

    l.append(results_array)
    # Arrays to store results
    H = np.zeros((reps, K))
    D_bu_bayes = np.zeros((reps, K))
    D_bu_pi1 = np.zeros((reps, K))
    D_imphom_bayes = np.zeros((reps, K))
    D_imphom_pi1 = np.zeros((reps, K))
    D_hom = np.zeros((reps, K))
    D_gou = np.zeros((reps, K))
    ##################
    for rep in range(reps):
        H[rep]= results_array[rep][0]
        D_bu_bayes[rep]= results_array[rep][1]
        D_bu_pi1[rep]= results_array[rep][2]
        D_imphom_bayes[rep]= results_array[rep][3]
        D_imphom_pi1[rep]= results_array[rep][4]
        D_hom[rep]= results_array[rep][5]
        D_gou[rep]= results_array[rep][6]
    
    Result[K0-1,:]    = np.array([int(K0),np.mean(np.sum(D_bu_bayes * H, axis=1) / max(1, K0)), np.mean(np.sum(D_bu_bayes * (1 - H), axis=1) > 0),
                                np.mean(np.sum(D_bu_pi1 * H, axis=1) / max(1, K0)),np.mean(np.sum(D_bu_pi1 * (1 - H), axis=1) > 0),
                                np.mean(np.sum(D_imphom_bayes * H, axis=1) / max(1, K0)), np.mean(np.sum(D_imphom_bayes * (1 - H), axis=1) > 0),
                                np.mean(np.sum(D_imphom_pi1 * H, axis=1) / max(1, K0)), np.mean(np.sum(D_imphom_pi1 * (1 - H), axis=1) > 0),
                            np.mean(np.sum(D_hom * H, axis=1) / max(1, K0)), np.mean(np.sum(D_hom * (1 - H), axis=1) > 0),
                                 np.mean(np.sum(D_gou * H, axis=1) / max(1, K0)), np.mean(np.sum(D_gou * (1 - H), axis=1) > 0)])


end_time = time.time()
elapsed_time = end_time - start_time

print("Elapsed time:", elapsed_time, "seconds")

Result[K0-1,:] = Result[K0-1,:]*np.array([(K/2)/K0,max(1, K0)/(K/2),1,max(1, K0)/(K/2),1,max(1, K0)/(K/2),1,max(1, K0)/(K/2),1,max(1, K0)/(K/2),1,max(1, K0)/(K/2),1]);
#np.save('results_array_bool_25_15_5.npy',np.array(l, dtype=bool))
all_list.append(np.array(l, dtype=bool));
all_result.append(np.array(Result));
# df = pd.DataFrame(Result)
# df.to_csv('all_result_25_15_5.csv', index=False)
reps = int(1e7)
#np.random.seed(1011)
K0 = 0;
# Arrays to store results
if __name__ == "__main__":
    num_processes = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_rep, range(reps))
    results_array = np.array(results)
    del(results)

all_list.append(np.array(results_array, dtype=bool))
# Arrays to store results
H = np.zeros((reps, K))
D_bu_bayes = np.zeros((reps, K))
D_bu_pi1 = np.zeros((reps, K))
D_imphom_bayes = np.zeros((reps, K))
D_imphom_pi1 = np.zeros((reps, K))
D_hom = np.zeros((reps, K))
D_gou = np.zeros((reps, K))
    ##################
for rep in range(reps):
    H[rep]= results_array[rep][0]
    D_bu_bayes[rep]= results_array[rep][1]
    D_bu_pi1[rep]= results_array[rep][2]
    D_imphom_bayes[rep]= results_array[rep][3]
    D_imphom_pi1[rep]= results_array[rep][4]
    D_hom[rep]= results_array[rep][5]
    D_gou[rep]= results_array[rep][6]
    
all_result.append(np.array([int(K0),np.mean(np.sum(D_bu_bayes * H, axis=1) / max(1, K0)), np.mean(np.sum(D_bu_bayes * (1 - H), axis=1) > 0),
                                np.mean(np.sum(D_bu_pi1 * H, axis=1) / max(1, K0)),np.mean(np.sum(D_bu_pi1 * (1 - H), axis=1) > 0),
                                np.mean(np.sum(D_imphom_bayes * H, axis=1) / max(1, K0)), np.mean(np.sum(D_imphom_bayes * (1 - H), axis=1) > 0),
                                np.mean(np.sum(D_imphom_pi1 * H, axis=1) / max(1, K0)), np.mean(np.sum(D_imphom_pi1 * (1 - H), axis=1) > 0),
                            np.mean(np.sum(D_hom * H, axis=1) / max(1, K0)), np.mean(np.sum(D_hom * (1 - H), axis=1) > 0),
                           np.mean(np.sum(D_gou * H, axis=1) / max(1, K0)), np.mean(np.sum(D_gou * (1 - H), axis=1) > 0)]))
print("done")

print(all_result[3]);
np.save('results_pow_50_10_gr.npy',np.array(all_result,dtype=object))
# np.save('results_array_bool_25_10_r.npy',np.array(all_list,dtype=bool))
print(all_result)
