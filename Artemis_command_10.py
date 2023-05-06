#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 14:03:42 2021

@author: feynmanliu
"""
import rebound
import numpy as np
import pandas as pd
from function import uni, rl, extrac, m_final, a_final, e_final


def simulation(k, exponent, m_planet, sigma_value, num_planets):
    sim = rebound.Simulation()
    sim.integrator = "WHFast"
    sim.units = ('yr', 'AU', 'Msun')
    sim.add(m=1)

    m_array = m_final(m_planet, sigma_value, num_planets)
    if num_planets == 3:
        a_array_final = a_final(m_array, k, exponent, num_planets)
    else:
        a_array_final = a_final(m_array, k, exponent)
    e_array_final = e_final(a_array_final)

    for i in range(num_planets):
        sim.add(m=m_array[i], inc=rl(0.01),
                Omega=uni(), omega=uni(), f=uni(),
                a=a_array_final[i], e=e_array_final[i])

    ptcl = sim.particles  # easy for getting the parameters of planets
    # change the radius of all the planets to its rhill
    for i in range(len(ptcl) - 1):
        ptcl[i + 1].r = ptcl[i + 1].rhill

    return sim


def run_simulation(m_planet, sigma_value, time_length, k, exponent, num_planets):  # length means the length of the simulation

    sim = simulation(k, exponent, m_planet, sigma_value, num_planets)
    ptcl = sim.particles  # easy for getting the parameters of planets
    sim.collision = "direct"

    # save the data of the planets before the simulation start
    planets_initial = [extrac(ptcl[i]) for i in range(1, num_planets+1)]

    sim.dt = 1 / 100  # t-step of the simulation in years

    try:
        for time in np.linspace(0, time_length, 101):
            sim.integrate(time)  # start the simulation in the given time

        time_result = sim.t
        planets_final = [extrac(ptcl[i]) for i in range(1, num_planets+1)]

        collide_planet_1 = 100
        collide_planet_2 = 100

    except rebound.Collision:
        planets_final = [extrac(ptcl[i]) for i in range(1, num_planets+1)]

        time_result = sim.t
        print("Collision at", time_result, "year")

        pass

        collided = []
        for p in ptcl:
            if p.lastcollision == sim.t:
                collided.append(p.index)

        collide_planet_1 = collided[0]
        collide_planet_2 = collided[1]

    data = {'exponent': [exponent], 'k': [k], 't_max': [time_length]}

    for i in range(num_planets):
        data[f"planet_{i+1}"] = [planets_initial[i]]
        data[f"planet_{i+1}_new"] = [planets_final[i]]

    data.update({
        'Encounter_time': [time_result],
        'collide_planet_1': [collide_planet_1], 'collide_planet_2': [collide_planet_2],
        'mass': [m_planet],
        'dlnmass': [sigma_value],
        'num_planets': [num_planets]
    })

    # output all the data as csv file
    file_name = "Exp_{}_k_{}_M_{}_dlnM_{}_N_{}.csv".format(exponent, k, m_planet, sigma_value, num_planets)

    result = pd.DataFrame(data)
    result_csv = result.to_csv(file_name, index=False, mode="a")

    return result
    
def multi_run_simulation(exponent_, sigma_value_, num_planets_):
    
    scenarios = {
        3e-5: [6, 7, 8],
        3e-4: [5.5, 6.0, 6.5],
        3e-3: [4.5, 5.0, 5.5]
    }
    
    for m_planet, k_values in scenarios.items():
        for k in k_values:
            for _ in range(100):
                print(run_simulation(m_planet=m_planet, sigma_value=sigma_value_, time_length=1e8, k=k, exponent=exponent_, num_planets=num_planets_))

    return
"""
def run_simulation_N3(m_planet, sigma_value, time_length, k, exponent):
    # Call run_simulation with num_planets = 3
    result = run_simulation(m_planet=m_planet, sigma_value=sigma_value, time_length=time_length, k=k, exponent=exponent, num_planets=3)
    return result

m_planet = 3e-5
sigma_value = 0.1
time_length = 1e8
k = 6
exponent = 0.925

result = run_simulation_N3(m_planet, sigma_value, time_length, k, exponent)
print(result)
"""
