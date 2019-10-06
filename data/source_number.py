import numpy as np
from data.fr import make_hankel

EPS = 1e-10


def SORTE(s, param=20):
    """
    Based on https://www.ese.wustl.edu/~nehorai/paper/Han_Jackknifing_TSP_2013.pdf
    """
    Y = make_hankel(s, param)
    sig = Y @ np.conj(Y.T)
    eig_values, _ = np.linalg.eigh(sig)
    eig_values = np.sort(eig_values)[::-1]  # \lamda_1, ..., \lambda_N
    delta_lambda = -np.diff(eig_values)
    var = var_delta(delta_lambda)
    sorte = np.divide(var[1:], var[:-1])
    return np.argmin(sorte) + 1


def var_delta(delta_lambda):
    cummean = np.cumsum(delta_lambda[::-1]) / np.arange(1, len(delta_lambda) + 1)  # mean( \delta_K, ..., \delta_{N-1} )
    delta_lambda_norm = (delta_lambda[None] - cummean[:, None]) ** 2
    var = np.sum(np.triu(delta_lambda_norm), axis=1) / np.arange(1, len(delta_lambda) + 1)
    return var


def AIC(s, param=20):
    """
    Based on http://www.dsp-book.narod.ru/DSPMW/67.PDF
    """
    Y = make_hankel(s, param)
    sig = Y @ np.conj(Y.T)
    eig_values, _ = np.linalg.eigh(sig)
    eig_values = np.sort(eig_values)[::-1]
    eig_values = np.clip(eig_values, EPS, np.inf)

    cumprod = np.cumprod(eig_values[::-1])[::-1]  # , 1/np.arange(1, len(eig_values)+1))[::-1]
    cummean = (np.cumsum(eig_values[::-1]) / np.arange(1, len(eig_values) + 1))
    cummean = np.power(cummean, np.arange(1, len(eig_values) + 1))[::-1]
    log_div = np.log(cumprod / cummean)
    n_s = np.arange(1, len(eig_values) + 1)
    m = len(s)
    n = sig.shape[0]
    k = np.argmin(-n * log_div + n_s * (2 * m - m))
    return k


def MDL(s, param=20):
    """
    http://www.dsp-book.narod.ru/DSPMW/67.PDF
    """
    Y = make_hankel(s, param)
    sigma = Y @ np.conj(Y.T)
    eig_values, _ = np.linalg.eigh(sigma)
    eig_values = np.sort(eig_values)[::-1]
    eig_values = np.clip(eig_values, EPS, np.inf)

    cumprod = np.cumprod(eig_values[::-1])[::-1]
    cummean = (np.cumsum(eig_values[::-1]) / np.arange(1, len(eig_values) + 1))
    cummean = np.power(cummean, np.arange(1, len(eig_values) + 1))[::-1]
    log_div = np.log(cumprod / cummean)
    n_s = np.arange(1, len(eig_values) + 1)
    m = len(s)
    n = sigma.shape[0]

    k = np.argmin(-n * log_div + 0.5 * n_s * (2 * m - n_s) * np.log(n))
    return k

def sorte_arr(signals, param=20):
    result = []
    for s in signals:
        result.append(SORTE(s, param))
    return np.array(result)


def mdl_arr(signals, param=20):
    result = []
    for s in signals:
        result.append(MDL(s, param))
    return np.array(result)


def aic_arr(signals, param=20):
    result = []
    for s in signals:
        result.append(AIC(s, param))
    return np.array(result)
