from tqdm.notebook import tqdm
import numpy as np

from imputation import core_utils

def fit_factors_and_loadings(char_panel, return_panel, min_chars, K, num_months_train,
                       reg=0, time_varying_lambdas=False,
                        adaptive_reg=False, min_reg=None, max_reg=None, window_size=None):
    """
    Parameters
    ----------
        char_panel : input TxNxC panel of characteristics
        return_panel : input TxN panel of returns
        num_months_train : number of months to fit lambda over if it is not a time varying estimatiion
        min_chars: the minimum numbr of characteristics required to fit the stock factors
        K : number of factors
        reg: regularization for factor regression step
        time_varying_lambdas: flag indicating whether or note lambda should be time varying
        adaptive_reg: flag indicating whether to perform adaptive regularization as outlined in the paper
        min_reg: used as the minimum value of the adaptive regularization
        max_reg: used as the mnaximum value of the adaptive regularization
    """
    
    if adaptive_reg:
        assert min_reg is not None and max_reg is not None

    lmbda = estimate_lambda(char_panel, return_panel, num_months_train=num_months_train,
                            K=K, min_chars=min_chars,
                           time_varying_lambdas=time_varying_lambdas, 
                            ignore_nans=False)
    
    assert np.sum(np.isnan(lmbda)) == 0, f"lambda should contain no nans, {np.argwhere(np.isnan(lmbda))}"
    T, N, L = char_panel.shape
    gamma_ts = np.zeros((char_panel.shape[0], char_panel.shape[1], K))
    missing_counts = np.sum(np.isnan(char_panel), axis=2)
    
    # select the stocks which we should impute, present returns and enough characteristics
    return_mask = np.logical_and(np.sum(np.logical_and(~np.isnan(char_panel),
                                                       ~np.isinf(char_panel)), axis=2) >= min_chars,
                                 ~np.isnan(return_panel))

    for t in tqdm(range(char_panel.shape[0])):
        ct = np.nan_to_num(char_panel[t]) * 1.0
        present = ~np.isnan(char_panel[t])

        to_impute = np.argwhere(return_mask[t]).squeeze()
        
        if time_varying_lambdas:
            gamma_t = lmbda[t].T.dot(ct.T).T
            gamma_t = core_utils.get_optimal_A(lmbda[t].T, gamma_t, present, ct,
                                               idxs=to_impute, min_chars=min_chars,
                                               reg=reg, adaptive_reg=adaptive_reg,
                                               min_reg=min_reg, max_reg=max_reg)
        else:
            gamma_t = lmbda.T.dot(ct.T).T
            gamma_t = core_utils.get_optimal_A(lmbda.T, gamma_t, present, ct,
                                               idxs=to_impute, min_chars=min_chars,
                                               reg=reg, adaptive_reg=adaptive_reg,
                                               min_reg=min_reg, max_reg=max_reg)
            
        gamma_ts[t, return_mask[t]] = gamma_t[return_mask[t]]
        gamma_ts[t, ~return_mask[t]] = np.nan        
    
    return gamma_ts, lmbda, return_mask, missing_counts


def estimate_lambda(char_panel, return_panel, num_months_train, K, min_chars,
                   time_varying_lambdas=False, ignore_nans=False):
    """
    Parameters
    ----------
        char_panel : input TxNxC panel of characteristics
        return_panel : input TxN panel of returns
        min_chars: the minimum numbr of characteristics required to fit the stock factors
        num_months_train : number of months to fit lambda over
        time_varying_lambdas: flag indicating whether or note lambda should be time varying
        K : number of factors
    """
    chars = np.array(['A2ME', 'AC', 'AT', 'ATO', 'B2M', 'BETA_d', 'BETA_m', 'C2A',
       'CF2B', 'CF2P', 'CTO', 'D2A', 'D2P', 'DPI2A', 'E2P', 'FC2Y',
       'HIGH52', 'INV', 'IdioVol', 'LEV', 'ME', 'NI', 'NOA', 'OA', 'OL',
       'OP', 'PCM', 'PM', 'PROF', 'Q', 'R12_2', 'R12_7', 'R2_1', 'R36_13',
       'R60_13', 'RNA', 'ROA', 'ROE', 'RVAR', 'S2P', 'SGA2S', 'SPREAD',
       'SUV', 'TURN', 'VAR'])
    monthly_chars = ['BETA_d', 'BETA_m', 'D2P', 'IdioVol', 'ME', 'TURN',
                     'R2_1', 'R12_2', 'R12_7', 'R36_13', 'R60_13', 'HIGH52', 'RVAR', 'SPREAD',  'SUV',  'VAR']

    min_char_mask = np.expand_dims(np.logical_and(np.sum(~np.isnan(char_panel), axis=2) >= min_chars,
                                                  ~np.isnan(return_panel)), axis=2)
    cov_mats = []
    first_warn = True
    for t in range(num_months_train):
        
        ct = char_panel[t]

        cov_mat = core_utils.get_cov_mat(ct,
                              np.nan_to_num(ct) * min_char_mask[t] * 1.0,
                              np.logical_and(~np.isnan(ct), min_char_mask[t]) * 1.0
                             )
        if ignore_nans:
            cov_mats.append(np.nan_to_num(cov_mat))

        elif np.sum(np.isnan(cov_mat)) > 0 and first_warn:
            print("nans in the covariance matrix")
            first_warn = False

        elif np.sum(np.isnan(cov_mat)) == 0:
            cov_mats.append(np.nan_to_num(cov_mat))

    cov_mats_sum = sum(cov_mats) * (1 / len(cov_mats))
    
    if time_varying_lambdas:
        lmbda = []
        printed = False
        for t in range(len(cov_mats)):
            tgt_mat = cov_mats[t]
            eig_vals, eig_vects = np.linalg.eigh(tgt_mat)
            idx = np.abs(eig_vals).argsort()[::-1]
            eig_vects = eig_vects[:, ]
            lmbda.append(eig_vects[:, idx[:K]])
    else:
        tgt_mat = cov_mats_sum
        eig_vals, eig_vects = np.linalg.eigh(tgt_mat)

        idx = np.abs(eig_vals).argsort()[::-1]
        eig_vects = eig_vects[:, ]

        lmbda = eig_vects[:, idx[:K]]                
    return lmbda


############### Sufficient Statistics Beta Regression ####################

def get_sufficient_statistics_xs(gamma_ts, characteristics_panel):
    return gamma_ts, None


def get_sufficient_statistics_last_val(characteristics_panel, max_delta=None):
    '''
    get the last value and the lag coresponding to it for the characteristics panel passed in
    '''
    T, N, L = characteristics_panel.shape
    last_val = np.copy(characteristics_panel[0])
    print(last_val.shape)
    lag_amount = np.zeros_like(last_val)
    lag_amount[:] = np.nan
    sufficient_statistics = np.zeros((T,N,L), dtype=float)
    sufficient_statistics[:,:,:] = np.nan
    deltas = np.copy(sufficient_statistics)
    for t in tqdm(range(1, T)):
        lag_amount += 1
        sufficient_statistics[t] = np.copy(last_val)
        deltas[t] = np.copy(lag_amount)
        present_t = ~np.isnan(characteristics_panel[t])
        last_val[present_t] = np.copy(characteristics_panel[t, present_t])
        lag_amount[present_t] = 0
        if max_delta is not None:
            last_val[lag_amount >= max_delta] = np.nan
        
    return sufficient_statistics, deltas

def get_sufficient_statistics_next_val(characteristics_panel, max_delta=None):
    '''
    get the next value and the lag coresponding to it for the characteristics panel passed in
    '''
    T, N, L = characteristics_panel.shape
    last_val = np.copy(characteristics_panel[-1])
    lag_amount = np.zeros_like(last_val)
    sufficient_statistics = np.zeros((T,N,L), dtype=float)
    deltas = np.copy(sufficient_statistics)
    sufficient_statistics[:,:,:] = np.nan
    for t in tqdm(range(T-2, -1, -1)):
        lag_amount += 1
        sufficient_statistics[t] = np.copy(last_val)
        deltas[t] = lag_amount
        
        present_t = ~np.isnan(characteristics_panel[t])
        last_val[present_t] = np.copy(characteristics_panel[t, present_t])
        lag_amount[present_t] = 0
        if max_delta is not None:
            last_val[lag_amount >= max_delta] = np.nan
        
    return sufficient_statistics, deltas

def get_sufficient_statistics_fwbw(characteristics_panel, max_delta=None):
    '''
    get the last and next value and the lags coresponding to them for the characteristics panel passed in
    '''
    bw, deltas_bw = get_sufficient_statistics_last_val(characteristics_panel, max_delta)
    fw, deltas_fw = get_sufficient_statistics_next_val(characteristics_panel, max_delta)
    return np.concatenate([np.expand_dims(bw, axis=3), np.expand_dims(fw, axis=3)], axis=3), deltas_bw, deltas_fw

def impute_beta_regression(characteristics_panel, gamma_ts, sufficient_statistics=None, 
                           window_size=0, beta_weights=None):
    T, N, L = characteristics_panel.shape
    if gamma_ts is not None:
        _, _, K = gamma_ts.shape
    else:
        K = 0
    if sufficient_statistics is not None:
        K += sufficient_statistics.shape[3]
    print(K)
    betas = np.zeros((T, L, K))
    imputed_data = np.copy(characteristics_panel)
    imputed_data[:,:,:]=np.nan
    for l in tqdm(range(L)):
        fit_suff_stats = []
        fit_tgts = []
        inds = []
        curr_ind = 0
        for t in range(T):
            inds.append(curr_ind)
            if gamma_ts is None:
                suff_stat = sufficient_statistics[t,:,l]
            elif sufficient_statistics is not None:
                suff_stat = np.concatenate([gamma_ts[t], sufficient_statistics[t,:,l]], axis=1)
            else:
                suff_stat = gamma_ts[t]
            available_for_imputation = np.all(~np.isnan(suff_stat), axis=1)
            available_for_fit = np.logical_and(~np.isnan(characteristics_panel[t,:,l]),
                                                  available_for_imputation)

            fit_suff_stats.append(suff_stat[available_for_fit, :])
            fit_tgts.append(characteristics_panel[t,available_for_fit,l])
            curr_ind += np.sum(available_for_fit)
        inds.append(curr_ind)
        fit_suff_stats = np.concatenate(fit_suff_stats, axis=0)
        fit_tgts = np.concatenate(fit_tgts, axis=0)
        for t in range(T):
            beta_l_t = np.linalg.lstsq(fit_suff_stats[inds[max(0, t-window_size)]:inds[t+1]],
                                       fit_tgts[inds[max(0, t-window_size)]:inds[t+1]], rcond=None)[0]
            if np.sum(np.isnan(beta_l_t)) > 0:
                print("should be no nans, t=", t,)
            betas[t,l] = beta_l_t
            
            if gamma_ts is None:
                suff_stat = sufficient_statistics[t,:,l]
            elif sufficient_statistics is not None:
                suff_stat = np.concatenate([gamma_ts[t], sufficient_statistics[t,:,l]], axis=1)
            else:
                suff_stat = gamma_ts[t]

            available_for_imputation = np.all(~np.isnan(suff_stat), axis=1)
            if beta_weights is None:
                imputed_data[t,available_for_imputation,l] = suff_stat[available_for_imputation,:] @ beta_l_t
            else:
                for i in np.argwhere(available_for_imputation).squeeze():
                    assert np.all(~np.isnan(beta_weights[(t,i,l)]))
                    imputed_data[t,i,l] = suff_stat[i,:] @ np.diag(beta_weights[(t,i,l)]) @ beta_l_t
            
            
    return imputed_data, betas

def impute_fixed_beta_regression(characteristics_panel, gamma_ts, sufficient_statistics=None, num_months_train=None,
                                fit_mask=None, betas=None, beta_weights=None):
    """
    characteristics_panel: panel of characteristics to fit regression over
    gamma_ts: xs factors
    sufficient_statistics: sufficient statistics
    num_months_train: if specified determines how man months to use to fit the regression
    beta_weights: a parameter to allow a prescribed changing of betas over time
    - should map from (t,i,l) -> np.array(size=sufficient_statistics.shape[-1])
    - for for example for FW we have sufficient_statistics -> ([xs vals], last, next)
    - weight can be ([1] * len(xs vals), (T -t) / T, t / T) where t is the index in the missing gap, T is the length
    - of the missing window
    """
    T, N, L = characteristics_panel.shape
    if num_months_train is None:
        num_months_train = T
    if gamma_ts is None:
        assert sufficient_statistics is not None
        K = sufficient_statistics.shape[3]
    else:
        _, _, K = gamma_ts.shape
        if sufficient_statistics is not None:
            K += sufficient_statistics.shape[3]
    print(K)
    if betas is None:
        betas = np.zeros((L, K))
    imputed_data = np.copy(characteristics_panel)
    imputed_data[:,:,:]=np.nan
    for l in tqdm(range(L)):
        if np.sum(betas[l]) == 0:
            fit_suff_stats = []
            fit_tgts = []
            for t in range(num_months_train):
                if gamma_ts is None:
                    suff_stat = sufficient_statistics[t,:,l]
                elif sufficient_statistics is not None:
                    suff_stat = np.concatenate([gamma_ts[t], sufficient_statistics[t,:,l]], axis=1)
                else:
                    suff_stat = gamma_ts[t]

                available_for_imputation = np.all(~np.isnan(suff_stat),
                                                        axis=1)
                available_for_fit = np.logical_and(~np.isnan(characteristics_panel[t,:,l]),
                                                      available_for_imputation)
                if fit_mask is not None:
                    available_for_fit = np.logical_and(available_for_fit, fit_mask)

                fit_suff_stats.append(suff_stat[available_for_fit, :])
                fit_tgts.append(characteristics_panel[t,available_for_fit,l])

            beta_l = np.linalg.lstsq(np.concatenate(fit_suff_stats, axis=0),
                                        np.concatenate(fit_tgts, axis=0), rcond=None)[0]

            betas[l] = beta_l
        for t in range(T):
            if gamma_ts is None:
                suff_stat = sufficient_statistics[t,:,l]
            elif sufficient_statistics is not None:
                suff_stat = np.concatenate([gamma_ts[t], sufficient_statistics[t,:,l]], axis=1)
            else:
                suff_stat = gamma_ts[t]
            available_for_imputation = np.all(~np.isnan(suff_stat),
                                                    axis=1)
            if beta_weights is None:
                imputed_data[t,available_for_imputation,l] = suff_stat[available_for_imputation,:] @ betas[l]
            else:
                for i in np.argwhere(available_for_imputation).squeeze():
                    # print(beta_weights[(t,i,l)])
                    if (t,i,l) in beta_weights:
                        assert np.all(~np.isnan(beta_weights[(t,i,l)]))
                        imputed_data[t,i,l] = suff_stat[i,:] @ np.diag(beta_weights[(t,i,l)]) @ betas[l]
                    else:
                        imputed_data[t,i,l] = suff_stat[i,:] @ betas[l]
            
    return imputed_data, betas


def prev_impute(char_panel):
    """
    imputes using the last value of the characteristic time series
    """
    imputed_panel = np.copy(char_panel)
    imputed_panel[:,:,:] = np.nan
    imputed_panel[0] = np.copy(char_panel[0])
    for t in tqdm(range(1, imputed_panel.shape[0])):
        present_t_l = ~np.isnan(char_panel[t-1])
        imputed_t_1 = ~np.isnan(imputed_panel[t-1])
        imputed_panel[t, present_t_l] = char_panel[t-1, present_t_l]
        imputed_panel[t, np.logical_and(~present_t_l, 
                                     imputed_t_1)] = imputed_panel[t-1, 
                                                                   np.logical_and(~present_t_l, imputed_t_1)]
        imputed_panel[t, ~np.logical_or(imputed_t_1, present_t_l)] = np.nan
        
    return imputed_panel

def xs_industry_median_impute(char_panel, industry_codes):
    """
    imputes using the last value of the characteristic time series
    """
    imputed_panel = np.copy(char_panel)
    for t in tqdm(range(imputed_panel.shape[0])):
        for c in range(imputed_panel.shape[2]):
            for x in np.unique(industry_codes):
                industry_filter = industry_codes==x
                present_t_l_i = np.logical_and(~np.isnan(char_panel[t,:, c]), industry_filter)
                imputed_panel[t, industry_filter, c] = np.median(char_panel[t,present_t_l_i, c])        
    return imputed_panel


def impute_chars(imputation_method, gamma_ts, char_data,
                          num_months_train=None, window_size=None):
    """
    given XS-factors (gamma) run the time series + XS info regression outlined in the paper
    Parameters
    ----------
        imputation_method: one of 'XS', 'BW-XS', 'BW', 'FW-XS', 'FWBW-XS'
        gamma_ts: XS factors, can be None to run with only tim series info
        char_data: characteristcs
        num_months_train=None: if beta is fixed, number of months over which to fit beta
        window_size: if not none, then the window size over which to fit time varying betas
    """
    if 'XS' in imputation_method:
        assert gamma_ts is not None, "needs xs factors to run any kind of xs imputation"
    
    if imputation_method == 'XS':
        suff_stat_method = 'None'
    elif imputation_method == 'BW-XS':
        suff_stat_method = 'last_val'
    elif imputation_method == 'FW-XS':
        suff_stat_method = 'next_val'
    elif imputation_method == 'BW':
        suff_stat_method = 'last_val'
    elif imputation_method == 'FWBW-XS':
        suff_stat_method = 'fwbw'
    
    if suff_stat_method == 'last_val':
        suff_stats, _ = get_sufficient_statistics_last_val(char_data, max_delta=None)
        suff_stats = np.expand_dims(suff_stats, axis=3)
        beta_weights = None
    elif suff_stat_method == 'next_val':
        suff_stats = np.expand_dims(get_sufficient_statistics_next_val(char_data, max_delta=None)[0], axis=3)
        beta_weights = None
    elif suff_stat_method == 'fwbw':
        next_val_suff_stats, fw_deltas = get_sufficient_statistics_next_val(char_data, max_delta=None)
        prev_val_suff_stats, bw_deltas = get_sufficient_statistics_last_val(char_data, max_delta=None)
        suff_stats = np.concatenate([np.expand_dims(prev_val_suff_stats, axis=3), 
                                              np.expand_dims(next_val_suff_stats, axis=3)], axis=3)
        beta_weight_arr = np.concatenate([np.expand_dims(fw_deltas, axis=3), 
                                              np.expand_dims(bw_deltas, axis=3)], axis=3)
        beta_weight_arr = 2 * beta_weight_arr / np.sum(beta_weight_arr, axis=3, keepdims=True)
        beta_weights = {}
        one_arr = np.ones((gamma_ts.shape[-1], 1))
        for t, i, j in tqdm(np.argwhere(np.logical_and(~np.isnan(fw_deltas), ~np.isnan(bw_deltas)))):
            beta_weights[(t,i,j)] = np.concatenate([one_arr, beta_weight_arr[t,i,j].reshape(-1, 1)], axis=0).squeeze()
        
    elif suff_stat_method == 'None':
        suff_stats = None
        beta_weights = None
    
    
    if window_size is not None:
        imputed_chars, betas = impute_beta_regression(char_data, gamma_ts, suff_stats,
                                                                      window_size=window_size, beta_weights=beta_weights)
    else:
        imputed_chars, betas = impute_fixed_beta_regression(char_data, gamma_ts, suff_stats,
                                                                        num_months_train=num_months_train,
                                                                        beta_weights=beta_weights)
    return imputed_chars
