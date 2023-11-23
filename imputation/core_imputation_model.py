import numpy as np
import os
import scipy as sp
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
from numpy.linalg import LinAlgError

############### Core Factor Imputation Model ####################


def get_optimal_A(B, A, present, cl, L, idxs=[], reg=0):
    """
    Get optimal A for cl = AB given that X is (potentially) missing data
    Parameters
    ----------
        B : matrix B
        A : matrix A, will be overwritten
        present: boolean mask of present data
        cl: matrix cl
        idxs: indexes which to impute
        reg: optinal regularization penalty
    """
    A[:,:] = np.nan
    for i in idxs:
        present_i = present[i,:]
        Xi = cl[i,:]
        Xi = Xi[present_i]
        Bi = B[:,present_i]
        assert np.all(~np.isnan(Bi)) and np.all(~np.isinf(Bi))
        effective_reg = reg 
        lmbda = effective_reg * np.eye(Bi.shape[1])
        try:
            A[i,:] = Bi @ np.linalg.lstsq(Bi.T @ Bi / L + lmbda, Xi / L, rcond=0)[0]
        except LinAlgError as e:
            lmbda = np.eye(Bi.shape[1])
            A[i,:] = Bi @ np.linalg.lstsq(Bi.T @ Bi / L + lmbda, Xi / L, rcond=0)[0]
    return A


def fit_factors_and_loadings(char_panel, min_chars, K, num_months_train,
                      reg=0.01,
                      time_varying_lambdas=False,
                      eval_data=None,
                      run_in_parallel=True):
    """
    Fit the cross-sectional Factors and Loadings using the regularized method
    Parameters
    ----------
        char_panel : the panel over which to fit the model T x N x L
        min_chars : the minimum number of observations required for a stock to include it in the sample
        K : the number of cross-sectional factors
        num_months_train : if fitting a global model, the number of months over which to fit the loadings
        reg=0.01 : the regularization strength
        time_varying_lambdas=False : whethere or not to allow the loadings to vary over time
        eval_data=None : Optional, can pass in an "eval" set for the data and print metrics
        run_in_parallel=True : Whether or not to use joblib to parallelize the factor estimation step
    """
    char_panel = np.copy(char_panel)
    missing_mask_overall = np.isnan(char_panel)
    missing_counts = np.sum(missing_mask_overall, axis=2)
    return_mask = np.sum(~missing_mask_overall, axis=2) >= min_chars
    
    char_panel[np.sum(~np.isnan(missing_mask_overall), axis=2) < min_chars] = np.nan
    imputed_chars = np.copy(char_panel)
    
    T, N, L = char_panel.shape
    
    resid_cov_mats = [None for _ in range(T)]
    mu = np.zeros((T, L), dtype=float)
    lmbda, cov_mat = estimate_lambda(imputed_chars, num_months_train=num_months_train,
                            K=K, min_chars=min_chars,
                                    time_varying_lambdas=time_varying_lambdas)

    assert np.sum(np.isnan(lmbda)) == 0, f"lambda should contain no nans, {np.argwhere(np.isnan(lmbda))}"

    gamma_ts = np.zeros((char_panel.shape[0], char_panel.shape[1], K))
    gamma_ts[:,:] = np.nan        

    def get_gamma_t(ct, present, to_impute, lmbda, time_varying_lambdas, t):

        if time_varying_lambdas:
            gamma_t = lmbda[t].T.dot(ct.T).T # gamma_t = ct @ lmbda[t]
            gamma_t = get_optimal_A(lmbda[t].T, gamma_t, present, ct, L=L,
                                    idxs=to_impute, reg=reg)
        else:
            gamma_t = lmbda.T.dot(ct.T).T # gamma_t = ct @ lmbda
            gamma_t = get_optimal_A(lmbda.T, gamma_t, present, ct, L=L, idxs=to_impute, reg=reg)
        return gamma_t

    if run_in_parallel:
        gammas = [x for x in Parallel(n_jobs=30, verbose=5)(delayed(get_gamma_t)(
            ct = char_panel[t], 
            present = ~np.isnan(char_panel[t]),
            to_impute = np.argwhere(return_mask[t]).squeeze(),
            lmbda=lmbda,
            time_varying_lambdas=time_varying_lambdas, t=t,
        ) for t in range(T))]
    else:
        gammas = [get_gamma_t(
            ct = char_panel[t], 
            present = ~np.isnan(char_panel[t]),
            to_impute = np.argwhere(return_mask[t]).squeeze(),
            lmbda=lmbda,
            time_varying_lambdas=time_varying_lambdas, t=t,
        ) for t in range(T)]

    for t in tqdm(range(T)):
        gamma_ts[t, return_mask[t]] = gammas[t][return_mask[t]]

    if time_varying_lambdas:
        new_imputation = np.concatenate([np.expand_dims(x @ l.T, axis=0) for x,l in zip(gamma_ts, lmbda)], axis=0)
    else:
        new_imputation = np.concatenate([np.expand_dims(x @ lmbda.T, axis=0) for x in gamma_ts], axis=0)

    resids = char_panel - new_imputation
    print(f"resids rmse are ", np.sqrt(np.nanmean(np.square(resids))))
    if eval_data is not None:
        print(f"eval resids rmse are ", np.sqrt(np.nanmean(np.square(new_imputation - eval_data))))

    imputed_chars[missing_mask_overall] = new_imputation[missing_mask_overall]
        
            
    return gamma_ts, lmbda


def estimate_lambda(char_panel, num_months_train, K, min_chars,
                    time_varying_lambdas=False):
    """
    Fit the cross-sectional Loadings using the XP method
    Parameters
    ----------
        char_panel : the panel over which to fit the model T x N x L
        num_months_train : if fitting a global model, the number of months over which to fit the loadings
        K : the number of cross-sectional factors
        min_chars : the minimum number of observations required for a stock to include it in the sample
        time_varying_lambdas=False : whethere or not to allow the loadings to vary over time
        run_in_parallel=True : Whether or not to use joblib to parallelize the factor estimation step
    """

    min_char_mask = np.expand_dims(np.sum(~np.isnan(char_panel), axis=2) >= min_chars, axis=2)

    cov_mats = []
    first_warn = True

    for t in range(num_months_train):
        cov_mats.append(get_cov_mat(char_panel[t]))
        
    cov_mats_sum = sum(cov_mats) * (1 / len(cov_mats))

    if time_varying_lambdas:
        lmbda = []
        cov_mat = []
        printed = False

        for t in range(len(cov_mats)):
            cov_mats_sum = cov_mats[t]
            eig_vals, eig_vects = np.linalg.eigh(cov_mats_sum)
            idx = np.abs(eig_vals).argsort()[::-1]
            lmbda.append(eig_vects[:, idx[:K]] * np.sqrt(np.maximum(eig_vals[idx[:K]].reshape(1, -1), 0)))
            assert np.all(~np.isnan(lmbda[-1])), lmbda
            cov_mat.append(cov_mats_sum)
    else:
        tgt_mat = cov_mats_sum
        eig_vals, eig_vects = np.linalg.eigh(tgt_mat)
#         print(eig_vals)

        idx = np.abs(eig_vals).argsort()[::-1]
        lmbda = eig_vects[:, idx[:K]] * np.sqrt(eig_vals[idx[:K]].reshape(1, -1))
        
        cov_mat = tgt_mat

    return lmbda, cov_mat


def get_cov_mat(char_matrix):
    """
    Calculate the covariance matrix of a partially observed panel using the method from Xiong & Pelger
    Parameters
    ----------
        char_matrix : the panel over which to calculate the covariance N x L
    """
    ct_int = (~np.isnan(char_matrix)).astype(float)
    ct = np.nan_to_num(char_matrix)
    mu = np.nanmean(char_matrix, axis=0).reshape(-1, 1)
    temp = ct.T.dot(ct) 
    temp_counts = ct_int.T.dot(ct_int)
    sigma_t = temp / temp_counts - mu @ mu.T
    return sigma_t


def get_sufficient_statistics_last_val(characteristics_panel, max_delta=None,
                                      residuals=None):
    """
    Get the last observed value for a panel time series 
    Parameters
    ----------
        characteristics_panel : the time series panel, T x N x L
        max_delta=None : Optional, the maximum lag which is allowed for a previously observed value
        residuals=None : Optional, residuals T x N x L, the residuals the factor model applied to the time
                        series panel
    """
    T, N, L = characteristics_panel.shape
    last_val = np.copy(characteristics_panel[0])
    if residuals is not None:
        last_resid = np.copy(residuals[0])

    lag_amount = np.zeros_like(last_val)
    lag_amount[:] = np.nan
    if residuals is None:
        sufficient_statistics = np.zeros((T,N,L, 1), dtype=float)
    else:
        sufficient_statistics = np.zeros((T,N,L, 2), dtype=float)
    sufficient_statistics[:,:,:,:] = np.nan
    deltas = np.copy(sufficient_statistics[:,:,:,0])
    for t in tqdm(range(1, T)):
        lag_amount += 1
        sufficient_statistics[t, :, :, 0] = np.copy(last_val)
        deltas[t] = np.copy(lag_amount)
        present_t = ~np.isnan(characteristics_panel[t])
        last_val[present_t] = np.copy(characteristics_panel[t, present_t])
        if residuals is not None:
            sufficient_statistics[t, :, :, 1] = np.copy(last_resid)
            last_resid[present_t] = np.copy(residuals[t, present_t])
        lag_amount[present_t] = 0
        if max_delta is not None:
            last_val[lag_amount >= max_delta] = np.nan
    return sufficient_statistics, deltas


def impute_chars(char_data, imputed_chars, residuals=None, 
                 suff_stat_method='None', constant_beta=False):
    """
    run the imputation for a given configuration
    Parameters
    ----------
        char_data : the time series panel, T x N x L
        imputed_chars: the cross-sectional imputation of the time series panel
        residuals=None : Optional, residuals T x N x L, the residuals the factor model applied to the time
                        series panel
        suff_stat_method=None : Optional, the type of information to add to the cross sectional panel in the 
                        imputation
        constant_beta=False: whether or not to allow time variation in the loadings of the model
    """
    if suff_stat_method == 'last_val':
        suff_stats, _ = get_sufficient_statistics_last_val(char_data, max_delta=None,
                                                                           residuals=residuals)
        if len(suff_stats.shape) == 3:
            suff_stats = np.expand_dims(suff_stats, axis=3)
                
    elif suff_stat_method == 'None':
        suff_stats = None
            
    if suff_stats is None:
        return imputed_chars
    else:
        return impute_beta_combined_regression(
            char_data, imputed_chars, sufficient_statistics=suff_stats, 
            constant_beta=constant_beta
        )

def impute_beta_combined_regression(characteristics_panel, xs_imps, sufficient_statistics=None, 
                                    constant_beta=False, get_betas=False):
    """
    run the imputation regression for a given configuration
    Parameters
    ----------
        char_data : the time series panel, T x N x L
        xs_imps: the cross-sectional imputation of the time series panel
        sufficient_statistics=None : Optional, the information to add to the cross sectaial panel in the imputation
        constant_beta=False: whether or not to allow time variation in the loadings of the model
        get_betas=False: whether or not to return the learned betas
        
    """
    T, N, L = characteristics_panel.shape
    K = 0
    if xs_imps is not None:
        K += 1
    if sufficient_statistics is not None:
        K += sufficient_statistics.shape[3]

    betas = np.zeros((T, L, K))
    imputed_data = np.copy(characteristics_panel)
    imputed_data[:,:,:]=np.nan
    
    for l in tqdm(range(L)):
        fit_suff_stats = []
        fit_tgts = []
        inds = []
        curr_ind = 0
        all_suff_stats = []
        
        for t in range(T):
            inds.append(curr_ind)
            
            if xs_imps is not None:
                suff_stat = np.concatenate([xs_imps[t,:,l:l+1], sufficient_statistics[t,:,l]], axis=1)
            else:
                suff_stat = sufficient_statistics[t,:,l]
            
            available_for_imputation = np.all(~np.isnan(suff_stat), axis=1)
            available_for_fit = np.logical_and(~np.isnan(characteristics_panel[t,:,l]),
                                                  available_for_imputation)
            all_suff_stats.append(suff_stat)

            fit_suff_stats.append(suff_stat[available_for_fit, :])
            fit_tgts.append(characteristics_panel[t,available_for_fit,l])
            curr_ind += np.sum(available_for_fit)
        
        
        inds.append(curr_ind)
        fit_suff_stats = np.concatenate(fit_suff_stats, axis=0)
        fit_tgts = np.concatenate(fit_tgts, axis=0)
        
        if constant_beta:

            beta = np.linalg.lstsq(fit_suff_stats, fit_tgts, rcond=None)[0]
                
            betas[:,l,:] = beta.reshape(1, -1)
        else:
            for t in range(T):
                beta_l_t = np.linalg.lstsq(fit_suff_stats[inds[t]:inds[t+1]],
                                       fit_tgts[inds[t]:inds[t+1]], rcond=None)[0]
                betas[t,l,:] = beta_l_t
                if np.any(np.isnan(beta_l_t)):
                    print("should be no nans, t=", t,)
                
        for t in range(T):
            beta_l_t = betas[t,l]
            suff_stat = all_suff_stats[t]
            available_for_imputation = np.all(~np.isnan(suff_stat), axis=1)
            imputed_data[t,available_for_imputation,l] = suff_stat[available_for_imputation,:] @ beta_l_t
            
    if get_betas:
        return imputed_data, betas
    else:
        return imputed_data


def get_oos_estimates_given_loadings(chars, reg, Lmbda, time_varying_lmbda=False, get_factors=False):
    """
    Generate the finite-sample correction to the cross-sectionally imputed data
    Parameters
    ----------
        chars : the time series panel, T x N x L
        Lmbda : the loadings in the Xiong - Pelger model
        time_varying_lmbda=False: whether or the loadings are time varying
        get_factors=False: whether or not to return the factors, or the imputed values        
    """
    C = chars.shape[-1]
    def impute_t(t_chars, reg, C, Lmbda, get_factors=False):
        if not get_factors:
            imputation = np.copy(t_chars) * np.nan
        else:
            imputation = np.zeros((t_chars.shape[0], t_chars.shape[1], Lmbda.shape[1])) * np.nan
        mask = ~np.isnan(t_chars)
        net_mask = np.sum(mask, axis=1)
        K = Lmbda.shape[1]
        for n in range(t_chars.shape[0]):
            if net_mask[n] == 1:
                imputation[n,:] = 0
            elif net_mask[n] > 1:
                for i in range(C):
                    tmp = mask[n, i]
                    mask[n,i] = False
                    y = t_chars[n, mask[n]]
                    X = Lmbda[mask[n], :]
                    L = np.eye(K) * reg
                    params = np.linalg.lstsq(X.T @ X + L, X.T @ y, rcond=None)[0]
                    if get_factors:
                        imputation[n,i] = params
                    else:
                        imputation[n,i] = Lmbda[i] @ params
                    
                    mask[n,i] = tmp
        return np.expand_dims(imputation, axis=0)
    chars = [chars_t for chars_t in chars]
    
    if time_varying_lmbda:
        imputation = list(Parallel(n_jobs=60)(delayed(impute_t)(chars_t, reg, C, l, get_factors=get_factors) 
                                              for chars_t, l in tqdm(zip(chars, Lmbda))))
    else:
        imputation = list(Parallel(n_jobs=60)(delayed(impute_t)(chars_t, reg, C, Lmbda, get_factors=get_factors)
                                              for chars_t in tqdm(chars)))
    return np.concatenate(imputation, axis=0)


def run_imputation(characteristics, n_xs_factors=20, time_varying_loadings=False,
                   xs_factor_reg=0.01, use_bw_ts_info=False,
                   include_ts_residuals=True, min_xs_obs=1):
    """
    run the imputation as described in the paper
    Parameters
    ----------
        characteristics : the time series panel, T x N x L
        n_xs_factors: the number of cross-sectional factors to use
        time_varying_loadings=False: whether or not to allow time variation in the loadings of the model
        xs_factor_reg=0.01 : the regularization to apply in the factor estimation
        use_bw_ts_info=False : whether or not to use past time series information
        include_ts_residuals=True : whether or not to include residuals from the XS model in the time series regression
        min_xs_obs=1 : the minimum number of observations a stock needs in a month to be included
        
    """
    T, N, L = characteristics.shape
    
    gamma_ts, lmbda = fit_factors_and_loadings(
        char_panel=characteristics, 
        min_chars=min_xs_obs, 
        K=n_xs_factors, 
        num_months_train=T,
        reg=xs_factor_reg,
        time_varying_lambdas=time_varying_loadings,
        eval_data=None,
        run_in_parallel=True
    )
    
    if time_varying_loadings:
        xs_imputations = np.concatenate([np.expand_dims(g @ l.T, axis=0) for g, l in zip(gamma_ts, lmbda)], axis=0)
    else:
        xs_imputations = np.concatenate([np.expand_dims(g @ lmbda.T, axis=0) for g in gamma_ts], axis=0)
    
    oos_xs_imputations = get_oos_estimates_given_loadings(
        characteristics,
        reg=xs_factor_reg, 
        Lmbda=lmbda, 
        time_varying_lmbda=time_varying_loadings)
    
    residuals = characteristics - xs_imputations
    
    if use_bw_ts_info:
        global_bw = impute_chars(
            characteristics,
            oos_xs_imputations, 
            residuals if include_ts_residuals else None,
            suff_stat_method='last_val', 
            constant_beta=~time_varying_loadings
        )
        return global_bw
    
    return oos_xs_imputations


def impute_data_xs(characteristics, n_xs_factors=20, time_varying_loadings=False,
                   xs_factor_reg=0.01, use_bw_ts_info=False,
                   include_ts_residuals=True, min_xs_obs=1):
    """
    run the cross-sectional imputation as described in the paper
    Parameters
    ----------
        characteristics : the time series panel, T x N x L
        n_xs_factors: the number of cross-sectional factors to use
        time_varying_loadings=False: whether or not to allow time variation in the loadings of the model
        xs_factor_reg=0.01 : the regularization to apply in the factor estimation
        min_xs_obs=1 : the minimum number of observations a stock needs in a month to be included
        
    """
    return run_imputation(characteristics, n_xs_factors=n_xs_factors, 
                          time_varying_loadings=time_varying_loadings,
                   xs_factor_reg=xs_factor_reg, use_bw_ts_info=False,
                   include_ts_residuals=True, min_xs_obs=min_xs_obs)


def impute_data_bxs(characteristics, n_xs_factors=20, time_varying_loadings=False,
                   xs_factor_reg=0.01, use_bw_ts_info=False,
                   include_ts_residuals=True, min_xs_obs=1):
    """
    run the imputation as described in the paper using backwards time series information
    Parameters
    ----------
        characteristics : the time series panel, T x N x L
        n_xs_factors: the number of cross-sectional factors to use
        time_varying_loadings=False: whether or not to allow time variation in the loadings of the model
        xs_factor_reg=0.01 : the regularization to apply in the factor estimation
        min_xs_obs=1 : the minimum number of observations a stock needs in a month to be included
        
    """
    return run_imputation(characteristics, n_xs_factors=n_xs_factors, 
                          time_varying_loadings=time_varying_loadings,
                   xs_factor_reg=xs_factor_reg, use_bw_ts_info=True,
                   include_ts_residuals=True, min_xs_obs=min_xs_obs)
    
    