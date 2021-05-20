import numpy as np
import random
from functions import utils


def ar2(value1,value2,coef1,coef2,mu,sigma):
    """
    AR(2) model, cfr. paper
    """
    return coef1*value1+coef2*value2 + np.random.randn()*sigma+mu


def ar1(value1,coef1,mu,sigma):
    """
    AR(1) model, cfr. paper
    """
    return coef1*value1 + np.random.randn()*sigma+mu


def generate_jumpingmean(window_size, stride=1, nr_cp=49, delta_t_cp = 100, delta_t_cp_std = 10, scale_min=-1, scale_max=1):
    """
    Generates one instance of a jumping mean time series, together with the corresponding windows and parameters
    """
    mu = np.zeros((nr_cp,))
    parameters_jumpingmean = []
    for n in range(1,nr_cp):
        mu[n] = mu[n-1] + n/16
    for n in range(nr_cp):
        nr = int(delta_t_cp+np.random.randn()*np.sqrt(delta_t_cp_std))
        parameters_jumpingmean.extend(mu[n]*np.ones((nr,)))
    
    parameters_jumpingmean = np.array([parameters_jumpingmean]).T

    ts_length = len(parameters_jumpingmean)
    timeseries = np.zeros((ts_length))
    for i in range(2,ts_length):
        timeseries[i] = ar2(timeseries[i-1],timeseries[i-2], 0.6,-0.5, parameters_jumpingmean[i], 1.5)

    windows = utils.ts_to_windows(timeseries, 0, window_size, stride)
    windows = utils.minmaxscale(windows, scale_min, scale_max)
    
    return timeseries, windows, parameters_jumpingmean


def generate_scalingvariance(window_size, stride=1, nr_cp=49, delta_t_cp = 100, delta_t_cp_std = 10, scale_min=-1, scale_max=1):
    """
    Generates one instance of a scaling variance time series, together with the corresponding windows and parameters
    """
    sigma = np.ones((nr_cp,))
    parameters_scalingvariance = []
    for n in range(1,nr_cp-1,2):
        sigma[n] = np.log(np.exp(1)+n/4)
    for n in range(nr_cp):
        nr = int(delta_t_cp+np.random.randn()*np.sqrt(delta_t_cp_std))
        parameters_scalingvariance.extend(sigma[n]*np.ones((nr,)))

    parameters_scalingvariance = np.array([parameters_scalingvariance]).T

    
    ts_length = len(parameters_scalingvariance)
    timeseries = np.zeros((ts_length))
    for i in range(2,ts_length):
        timeseries[i] = ar2(timeseries[i-1],timeseries[i-2], 0.6,-0.5, 0, parameters_scalingvariance[i])

    windows = utils.ts_to_windows(timeseries, 0, window_size, stride)
    windows = utils.minmaxscale(windows, scale_min, scale_max)
    
    return timeseries, windows, parameters_scalingvariance


def generate_gaussian(window_size, stride=1, nr_cp=49, delta_t_cp = 100, delta_t_cp_std = 10, scale_min=-1, scale_max=1):
    """
    Generates one instance of a Gaussian mixtures time series, together with the corresponding windows and parameters
    """
    mixturenumber = np.zeros((nr_cp,))
    parameters_gaussian = []
    for n in range(1,nr_cp-1,2):
        mixturenumber[n] = 1
    for n in range(nr_cp):
        nr = int(delta_t_cp+np.random.randn()*np.sqrt(delta_t_cp_std))
        parameters_gaussian.extend(mixturenumber[n]*np.ones((nr,)))

    parameters_gaussian = np.array([parameters_gaussian]).T

    ts_length = len(parameters_gaussian)
    timeseries = np.zeros((ts_length))
    for i in range(2,ts_length):
        if parameters_gaussian[i] == 0:
            timeseries[i] = 0.5*0.5*np.random.randn()+0.5*0.5*np.random.randn()
        else:
            timeseries[i] = -0.6 - 0.8*1*np.random.randn() + 0.2*0.1*np.random.randn()

    windows = utils.ts_to_windows(timeseries, 0, window_size, stride)
    windows = utils.minmaxscale(windows, scale_min, scale_max)
    
    return timeseries, windows, parameters_gaussian


def generate_changingcoefficients(window_size, stride=1, nr_cp=49, delta_t_cp = 100, delta_t_cp_std = 10, scale_min=-1, scale_max=1, p=1):
    """
    Generates one instance of a changing coefficients time series, together with the corresponding windows and parameters
    """
    # generate coefficients
    coeff = np.ones((nr_cp,p))
    if p == 1:
        parameters = []
    elif p == 2:
        parameters1 = []
        parameters2 = []
    for n in range(0,nr_cp,2):
        if p == 1:
            coeff[n] = np.random.rand() * 0.2
            if n + 1 < nr_cp:
                coeff[n + 1] = np.random.rand() * 0.15 + 0.8
        elif p == 2:
            coeff[n][0] = np.random.rand() * -0.7
            coeff[n][1] = np.random.rand() * 0.1
            if n + 1 < nr_cp:
                coeff[n + 1][0] = np.random.rand() * 0.2
                coeff[n + 1][1] = np.random.rand() * 0.6
    for n in range(nr_cp):
        nr = int(delta_t_cp+np.random.randn()*np.sqrt(delta_t_cp_std))
        if p == 1:
            parameters.extend(coeff[n] * np.ones((nr,)))
        elif p == 2:
            parameters1.extend(coeff[n][0] * np.ones((nr,)))
            parameters2.extend(coeff[n][1] * np.ones((nr,)))
    if p == 1:
        timeseries = [0]
        parameters = np.array([parameters]).T
        for idx in range(1, np.shape(parameters)[0]):
            timeseries.append(ar1(timeseries[idx-1], parameters[idx-1][0], 0, 1))
    elif p == 2:
        timeseries = [0, 0]
        parameters = [parameters1, parameters2]
        parameters = np.array(parameters).T
        for idx in range(2, np.shape(parameters)[0]):
            timeseries.append(ar2(timeseries[idx-1], timeseries[idx-2], parameters[idx][0], parameters[idx][1], 0, 1))

    windows = utils.ts_to_windows(timeseries, 0, window_size, stride)
    windows = utils.minmaxscale(windows, scale_min, scale_max)
    
    return timeseries, windows, parameters


def generate_changingcoefficients_5(window_size, stride=1, nr_cp=49, delta_t_cp=1000, delta_t_cp_std=10, scale_min=-1,
                                    scale_max=1):
    """
    Generates one instance of a changing coefficients time series, together with the corresponding windows and parameters
    """
    parameters1 = []
    parameters2 = []
    parameters3 = []
    parameters4 = []
    parameters5 = []

    index_old = 5
    coeff_candidate = [[0.1, 0.1, 0.1, 0.1, 0.1],
                       [0.1, -0.1, 0.1, -0.1, 0.1],
                       [0.1, -0.1, 0.3, -0.1, 0.1],
                       [0.1, -0.3, 0.1, -0.3, 0.1],
                       [0.1, 0.3, 0.1, 0.3, 0.1]]

    for n in range(nr_cp):
        while True:
            index = random.randint(0, 4)
            if index != index_old:
                break
        coeff = coeff_candidate[index]
        nr = int(delta_t_cp+np.random.randn()*np.sqrt(delta_t_cp_std))
        parameters1.extend(coeff[0] * np.ones((nr,)))
        parameters2.extend(coeff[1] * np.ones((nr,)))
        parameters3.extend(coeff[2] * np.ones((nr,)))
        parameters4.extend(coeff[3] * np.ones((nr,)))
        parameters5.extend(coeff[4] * np.ones((nr,)))

    timeseries = [0, 0, 0, 0, 0]
    parameters = [parameters1, parameters2, parameters3, parameters4, parameters5]
    parameters = np.array(parameters).T

    def ar5(value1, value2, value3, value4, value5, coef, mu, sigma):
        return coef[0] * value1 + coef[1] * value2 + coef[2] * value3 + coef[3] * value4 + coef[
            4] * value5 + np.random.randn() * sigma + mu

    for idx in range(5, np.shape(parameters)[0]):
        timeseries.append(
            ar5(timeseries[idx-1], timeseries[idx-2], timeseries[idx-3], timeseries[idx-4], timeseries[idx-4],
                parameters[idx], 0, 1))

    windows = utils.ts_to_windows(timeseries, 0, window_size, stride)
    windows = utils.minmaxscale(windows, scale_min, scale_max)

    return timeseries, windows, parameters