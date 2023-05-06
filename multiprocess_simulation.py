#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 12:10:13 2023

@author: huangzhiyi
"""

import multiprocessing
from Artemis_command_10 import multi_run_simulation



def worker(params):
    exponent_, sigma_value_, num_planets_ = params
    multi_run_simulation(exponent_, sigma_value_, num_planets_)


def parallel_simulations():
    exponents = [0.925, 0.95, 0.975, 1]
    sigma_values = [0.1, 0.3]
    num_planets_list = [3]

    params_list = [(exponent, sigma_value, num_planets)
                   for exponent in exponents
                   for sigma_value in sigma_values
                   for num_planets in num_planets_list]

    pool = multiprocessing.Pool(8)  # use 16 threads
    pool.map(worker, params_list)
    pool.close()
    pool.join()


if __name__ == "__main__":
    parallel_simulations()


