#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 19:27:48 2021

@author: feynmanliu
"""

import pandas as pd
import numpy as np
import re
from numpy import log
from  itertools import combinations
import matplotlib.pyplot as plt
from scipy.stats import rayleigh
import random
import rebound
from cmath import sqrt
import seaborn as sns
#from Petiti.survivaltime import Tsurv

import os

current_directory = os.getcwd()
print("Current directory:", current_directory)


def scale (array_a, array_b):
    array_new = []
    for i in array_b:
        frac = log(array_a[-1])/ log(array_b[-1])
        i = np.exp((log(i))*frac)
        array_new.append(i)
    return array_new

def a_dist_new(a_array, factor=1):
    if len(a_array) > 3:
        a_array[3] = a_array[2] + (a_array[3] - a_array[2]) * factor
    a_array_new = []
    for i in range(len(a_array) - 1):
        a_array_new.append((a_array[i] + a_array[i + 1]) / 2)
    a_array_new.append(a_array[-1] * 1.15)
    return a_array_new



def uni():#define the function to draw sample from uniform distribution
    value = np.random.uniform(low = 0, high = 2*np.pi) #range from 0 to 2 pi
    return value

def rl(scales): #define the function to draw sample from rayleigh distribution
    value = rayleigh.rvs(scale=abs(scales))
    return value

def lognorm(m_planet, sigma_value): #generate the random mass for the planets follow the lognorm distribution
    mass = np.random.lognormal(mean= 0, sigma = sigma_value)*(m_planet)
    return mass

#define the formula to calculate the semi-major axis of the next planets based on its previous semi-major axis, mass of the planets, and \beta
def semi_hill(m_i, m_j, k, a_i, exp, power):
    m_s = 1  # mass of the star
    beta = k
    delta = beta / 2 * ((m_i + m_j) / (3 * m_s))**(1 / power)  # solar mass in the denominator is assumed to be 1
    frac = (1 + delta) / (1 - delta)
    
    a_j = a_i**exp * frac  # change the "exp" in the power of a_i for different graph
    return a_j


def extrac(planet):
    m, x, y, z, vx, vy, vz = planet.m, planet.x, planet.y, planet.z, planet.vx, planet.vy, planet.vz
    e, a, inc = planet.e, planet.a, planet.inc

    parameter_array = [float(param) for param in [m, x, y, z, vx, vy, vz, e, a, inc]]

    return np.asarray(parameter_array)

 
def m_final(mass, dlnmass, num_planets):
    m_array = []#array for the mass
    for i in range(num_planets):
      m_array.append(lognorm(mass, dlnmass))
    return m_array


def a_final(m_array, k, exponent, num_planets=None):
    if num_planets is None:
        num_planets = 4

    if num_planets == 3:
        num_a_arrays = 2
    else:
        num_a_arrays = 4

    a_arrays = [[1] for _ in range(num_a_arrays)]
    funcs = [semi_hill] * num_a_arrays
    exp_values = [1, exponent] * (num_a_arrays // 2)
    powers = [3, 3, 4, 4][:num_a_arrays]

    for i, (func, exp, power) in enumerate(zip(funcs, exp_values, powers)):
        for j in range(len(m_array) - 1):
            a_arrays[i].append(func(m_array[j], m_array[j + 1], k, a_arrays[i][j], exp=exp, power=power))

    if num_planets == 3:
        a_array_final = scale(a_arrays[0], a_arrays[1])
    else:
        a_array_3_new = scale(a_arrays[0], a_arrays[1])
        a_array_4_new = scale(a_arrays[2], a_arrays[3])
        a_array_final = scale(a_array_3_new, a_array_4_new)

    return a_dist_new(a_array_final, factor=1)


def e_final(a_array):
    e_array = []

    for i in range(len(a_array)):
        if i == 0:  # first planet
            planet = (a_array[i + 1] - a_array[i]) / a_array[i] / 20
        elif i == len(a_array) - 1:  # last planet
            planet = (a_array[i] - a_array[i - 1]) / a_array[i - 1] / 20
        else:  # middle planets
            planet = ((a_array[i + 1] - a_array[i]) / a_array[i] + (a_array[i] - a_array[i - 1]) / a_array[i]) / 40

        e_array.append(planet)

    # change the standard deviation into an actual function
    e_array = rl(abs(np.array(e_array)))

    return e_array


#given the period of the two adjcant planet, found the close Mimport numpy as np
def MMR_func(planet_1, planet_2):
    # Calculate the period ratio of the two planets
    r = (planet_2.a / planet_1.a) ** (3 / 2)

    # Define n and k values
    n_values = np.arange(1, 11)
    k_values = np.arange(1, 5)

    # Calculate all MMRs and differences
    MMRs, diffs, i_values = [], [], []
    for n in n_values:
        for k in k_values:
            MMR = (k + n) / n
            diff = abs(r - MMR)

            MMRs.append(MMR)
            diffs.append(diff)
            i_values.append(n)

    # Find the index of the minimum difference
    min_index = np.argmin(diffs)

    return MMRs[min_index], i_values[min_index]  # Return the value of MMR and k separately


def e_vec(planet):
    w_i = planet.pomega #extract the value of longitude of pericenter
    e = planet.e #extract the eccentricity of the planet
    return (e*np.exp(sqrt(-1)* w_i))


def delta(planet_1, planet_2):
    def e_ij_hat(planet_1, planet_2):
        e_i_vec = e_vec(planet_1)
        e_j_vec = e_vec(planet_2)

        e_ij_vec = abs(e_j_vec - e_i_vec)  # taking the abs to change the complex into real
        e_cross_ij = (planet_2.a - planet_1.a) / planet_2.a
        return e_ij_vec / e_cross_ij

    def sigma_p_over_p(planet_1, planet_2):
        k = MMR_func(planet_1, planet_2)[1]
        m_ij = (planet_1.m + planet_2.m) / 1  # 1 means the mass of the sun

        e_ij_hat_value = e_ij_hat(planet_1, planet_2)

        h_k = 1
        sigma_a_over_a = 4 * np.sqrt(h_k / 3) * np.sqrt(m_ij) * (e_ij_hat_value) ** (k / 2)
        return 3 / 2 * sigma_a_over_a

    sigma_p_over_p_value = sigma_p_over_p(planet_1, planet_2)

    # calculate the period ratio of the two planets
    r = (planet_2.a / planet_1.a) ** (3 / 2)

    MMR = MMR_func(planet_1, planet_2)[0]

    return (r / MMR - 1) / (sigma_p_over_p_value)


def MMR_system(sim):
    """
    Calculate the MMR of the whole system.

    :param sim: REBOUND simulation object
    :return: Mean MMR, minimum absolute MMR, index of the minimum MMR, and the second smallest MMR
    """
    ptcl = sim.particles  # easy for getting the parameters of planets

    delta_array = []
    num_planets = sim.N - 1

    for p in range(num_planets - 1):
        planet_1 = ptcl[p + 1]
        planet_2 = ptcl[p + 2]

        delta_value = delta(planet_1, planet_2)
        delta_array.append(delta_value)

    array_mean = np.mean(delta_array)
    min_index = np.argmin(np.absolute(delta_array))
    sorted_deltas = sorted(delta_array)

    return array_mean, delta_array[min_index], min_index, sorted_deltas[1]


def period_ratio(sim):
    """
    Calculate the mean period ratio of the planets in the simulation.

    :param sim: Simulation object
    :return: Mean period ratio of the planets
    """
    sim_particles = sim.particles
    num_planets = sim.N - 1  # Subtract 1 for the central star

    # Get semi-major axes for all the planets
    planet_semi_major_axes = [sim_particles[i + 1].a for i in range(num_planets)]

    # Calculate period ratios for the planets
    period_ratios = [(planet_semi_major_axes[i + 1] / planet_semi_major_axes[i])**(3 / 2) for i in range(num_planets - 1)]

    # Return the mean period ratio
    return np.mean(period_ratios)

    
def simulation(k, exponent, m_planet , sigma_value):
    sim = rebound.Simulation()
    sim.integrator = "WHFast"
    sim.units = ('yr', 'AU', 'Msun')
    sim.add(m=1)

    m_array = m_final(m_planet, sigma_value)
    a_array_final = a_final(m_array, k, exponent)
    e_array_final = e_final(a_array_final)

    for i in range(5): #5 means the number of planets
        sim.add(m = m_array[i], inc = rl(0.01) ,
                Omega = uni(), omega = uni(), f = uni(),
                a = a_array_final[i], e = e_array_final[i])
    
    ptcl = sim.particles#easy for getting the parameters of planets
    #change the radius of all the planets to its rhill
    for i in range(len(ptcl)-1):
        ptcl[i+1].r = ptcl[i+1].rhill
    
    return sim

#random select the \beta value in a given range follows uniform distribution
def rand(lower, upper):
    lower = lower *10
    upper = upper *10
    return random.randint(lower, upper) *0.1

def count_frequency(array):
    """
    Calculate the relative frequency (in percentage) of each element in an array.

    :param array: Input array
    :return: Relative frequency in percentage for each unique element in the array
    """
    unique_values, counts = np.unique(array, return_counts=True)

    # Calculate relative frequency in percentage
    relative_frequencies = counts / len(array) * 100

    # Combine unique values and their relative frequencies
    result = np.vstack((unique_values, relative_frequencies))

    return result


#generate num of system to calculate the distrbution of MMR
def MMR_sim(k_lower, k_upper, expoent, m_planet, sigma_value, num):
    MMR_result_array = []
    MMR_result_collide = []
    for i in range(int(num)):
        sim = simulation(k = rand(k_lower, k_upper),
                        exponent = expoent,
                        m_planet = m_planet ,
                        sigma_value = sigma_value)
        MMR_result_array.append(MMR_system(sim)[1])
        MMR_result_collide.append(MMR_system(sim)[2])
    
    result = count_frequency(MMR_result_collide)
    result = np.array(result)
    std = np.std(result)
    #result.append(np.std(result))
    print("Planet mass is ", m_planet, "sigma_m", sigma_value, "exponent", expoent, 
          "Relative Collide Frequency", result )
    print("Planet mass is ", m_planet, "sigma_m", sigma_value, "exponent", expoent, 
          "Collide Frequency std", std )

    mean = np.mean(MMR_result_array)
    std = np.std(MMR_result_array)
    
    #logrithms x-axis
    MMR_result_array = np.abs(MMR_result_array)
    MMR_result_array = np.log10(MMR_result_array)
    
    print("Planet mass is ", m_planet, "sigma_m", sigma_value, "exponent", expoent, "MMR mean", mean )
    print("Planet mass is ", m_planet, "sigma_m", sigma_value, "exponent", expoent, "MMR std", std )
    return MMR_result_array


#calcualte the realtive frequency of collide planets from MMR.
def MMR_sim_collide(k_lower, k_upper, expoent, m_planet, sigma_value, num):
    MMR_result_array = []
    for i in range(int(num)):
        sim = simulation(k = rand(k_lower, k_upper),
                        exponent = expoent,
                        m_planet = m_planet ,
                        sigma_value = sigma_value)
        MMR_result_array.append(MMR_system(sim)[3])
    
    result = count_frequency(MMR_result_array)
    result = np.array(result)
    std = np.std(result)
    #result.append(np.std(result))
    print("Planet mass is ", m_planet, "sigma_m", sigma_value, "exponent", expoent, 
          "Relative Collide Frequency", result )
    print("Planet mass is ", m_planet, "sigma_m", sigma_value, "exponent", expoent, 
          "std", std )

    return result #the last elemnt in the array are the std of the previous four element


def plot_MMR_dist(planet_mass, sigma_m, num_trail, name):
    # Define k ranges based on planet_mass
    k_l, k_u = {
        3e-5: (6, 9),
        3e-4: (5, 7),
        3e-3: (4, 6)
    }.get(planet_mass, (None, None))

    exponents = [1, 0.975, 0.95, 0.925, 0.9]
    MMR_arrays = []

    for expoent in exponents:
        MMR_array = MMR_sim(k_lower=k_l,
                            k_upper=k_u,
                            expoent=expoent,
                            m_planet=planet_mass,
                            sigma_value=sigma_m,
                            num=num_trail)
        MMR_arrays.append(MMR_array)

    labels = [r'$k = {}$'.format(exp) for exp in exponents]

    for MMR_array, label in zip(MMR_arrays, labels):
        sns.kdeplot(MMR_array, label=label, fill=False)

    plt.xlabel("Logarithm Distance to Mean Motion Resonance")
    plt.ylabel("Relative Frequency")
    plt.title(name)
    plt.legend()

    return plt.savefig(name + ".pdf")


def run_simulation(m_planet, sigma_value, time_length, k, exponent):
    sim = simulation(k, exponent, m_planet, sigma_value)
    ptcl = sim.particles
    sim.collision = "direct"

    pl1, pl2, pl3, pl4, pl5 = (extrac(ptcl[i]) for i in range(1, 6))

    sim.dt = 1/100

    time_result = 0
    collide_planet_1 = 6
    collide_planet_2 = 6

    try:
        for time in np.linspace(0, time_length, 301):
            sim.integrate(time)

        time_result = sim.t
        pl1_new, pl2_new, pl3_new, pl4_new, pl5_new = (extrac(ptcl[i]) for i in range(1, 6))

    except rebound.Collision:
        time_result = sim.t
        print("Collision at", time_result, "year")

        pl1_new, pl2_new, pl3_new, pl4_new, pl5_new = (extrac(ptcl[i]) for i in range(1, 6))

        collided = [p.index for p in ptcl if p.lastcollision == sim.t]

        if len(collided) >= 2:
            collide_planet_1, collide_planet_2 = collided[:2]

    data = {'exponent': [exponent], 'k': [k], 't_max': [time_length],
            'planet_1': [pl1], 'planet_2': [pl2], 'planet_3': [pl3], 'planet_4': [pl4], 'planet_5': [pl5],
            'Encounter_time': [time_result],
            'planet_1_new': [pl1_new], 'planet_2_new': [pl2_new], 'planet_3_new': [pl3_new], 'planet_4_new': [pl4_new], 'planet_5_new': [pl5_new],
            'collide_planet_1': [collide_planet_1], 'collide_planet_2': [collide_planet_2],
            'mass': [m_planet],
            'dlnmass': [sigma_value]}

    file_name = "Exp_{}_k_{}_M_{}_dlnM_{}.csv".format(exponent, k, m_planet, sigma_value)

    result = pd.DataFrame(data)

    return result

def get_particle_attribute(ptcl_, num_planets, attribute):
    attribute_array = []

    for i in range(num_planets):
        # Check if the index is within the range of available particles
        if i + 1 < len(ptcl_):
            attribute_value = getattr(ptcl_[i + 1], attribute)
            attribute_array.append(attribute_value)

    return attribute_array

# Function to get the closest value within array
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()

    if array[idx] > value:
      idx = idx - 1
    return idx

# The eccentricty checking time series 
# Estimate_encounter_time in log10
# Come from the average of encounter time from prevouis simulation
def get_ecc_time_series(Estimate_encounter_time): 

    #logspace time initially
    time_series_log = 1000*np.logspace(0, (Estimate_encounter_time) - 3, 2001) 
    
    #linspace time after the average enclutner time
    time_series_linear = np.arange(10**Estimate_encounter_time, 1e8, 1e3)
    
    #reverse the logspace to get more data point at beginning
    #less data point within the end
    final_time = np.repeat(10**Estimate_encounter_time, 2001)
    time_series_log = final_time - time_series_log
    time_series_log = np.flip(time_series_log)
    
    #combine the logspace time and linspace time
    time_series = np.concatenate((time_series_log, time_series_linear),  axis=None)
    
    return time_series


# The eccentricty checking time series 
# Assuming we already known the encounter time 
def get_ecc_time_series_new(Estimate_encounter_time):
    time_series_log = 100_000* np.logspace(0, Estimate_encounter_time -5, 10) 
    
    #reverse the logspace to get more data point at beginning
    #less data point within the end
    final_time = np.repeat(10**Estimate_encounter_time, 10)
    time_series_log = final_time - time_series_log
    time_series_log = np.flip(time_series_log)
    
    time_series_log = np.append(time_series_log, 10**Estimate_encounter_time)
    
    
    return time_series_log#np.linspace(0, 10**(Estimate_encounter_time), 11)

def get_all_parameter(ptcl, num_planets):
    eccentricity_n = []
    semi_major_n = []
    inclin_n = []
    Omega_n = []
    omega_n = []
    f_n = []
    remaining_indices = []

    for i in range(1, num_planets + 1):
        try:
            if ptcl[i].e <= 1:
                eccentricity_n.append(ptcl[i].e)
                semi_major_n.append(ptcl[i].a)
                inclin_n.append(ptcl[i].inc)
                Omega_n.append(ptcl[i].Omega)
                omega_n.append(ptcl[i].omega)
                f_n.append(ptcl[i].f)
                remaining_indices.append(i)
        except AttributeError:
            continue

    return eccentricity_n, semi_major_n, inclin_n, f_n, Omega_n, omega_n, remaining_indices


def data_to_system(data, row_num):
    data = data.iloc[row_num]
    
    sim = rebound.Simulation()
    sim.integrator = "WHFast"
    sim.units = ('yr', 'AU', 'Msun')
    sim.add(m=1)

    planet_columns = [col for col in data.index if re.match(r"planet_\d+$", col)]
    for col in planet_columns:
        planet = data[col]

        sim.add(m = planet[0],
                x = planet[1], y = planet[2], z = planet[3],
                vx = planet[4], vy = planet[5], vz = planet[6])

    return sim


def transform_data(data_path):
    data = pd.read_csv(data_path)
    
    # Automatically detect the number of planet columns
    orig_labels = [col for col in data.columns if re.match(r"planet_\d+$", col)]
    final_labels = [col for col in data.columns if re.match(r"planet_\d+_new$", col)]

    for label in orig_labels + final_labels:
        data[label] = [data[label].values[j].replace('\n', '').replace('[', '').replace(']', '').replace('  ', ',').replace(' ', ',').split(',') for j in range(len(data))]
        data[label] = [np.array(i)[np.array(i) != ''].astype(float) for i in data[label].values]

    data['mass'] = data['mass'].values.astype(float)
    data['k'] = data['k'].values.astype(float)
    data['logt'] = np.log10(data.Encounter_time.values.astype(float))
    data['exponent'] = data['exponent'].values.astype(float)
    data['S'] = data['k'] * data.mass**(1. / 12.) * 3**(-1. / 3.)

    return data

def process_data(data, row_num_=0):
    #collide_planet_1 = int(data['collide_planet_1'].values[row_num_])
    #collide_planet_2 = collide_planet_1 + 1

    #data_pivot = data.groupby(by=['k', 'exponent'], as_index=False).mean()

    Encounter_time = data["Encounter_time"]
    Encounter_time = np.array(Encounter_time)
    Encounter_time = Encounter_time[row_num_]
    Encounter_time = np.log10(float(Encounter_time))

    sim = data_to_system(data, row_num=row_num_)

    return sim, Encounter_time


def reciprocal_sum(arr):
    arr = np.array(arr)
    if np.any(arr == 0):
        raise ValueError("Input array contains zero(s), and division by zero is not allowed.")
    
    result = np.sum(1 / arr)
    return 1 / result

def analyze_system(sim):
    # Make sure to import rebound before calling this function
    ptcl = sim.particles
    num_planets = sim.N - 1  # subtract 1 to exclude the star

    if num_planets < 4:
        raise ValueError("The simulation must have at least four planets.")

    results = []

    for i in range(2, num_planets - 1):
        period_ratio_1 = ptcl[i].P / ptcl[i + 1].P
        period_ratio_2 = ptcl[i + 1].P / ptcl[i + 2].P

        planet_masses = np.array([ptcl[j].m for j in range(i, i + 3)])

        results.append((period_ratio_1, period_ratio_2, planet_masses))

    return results






