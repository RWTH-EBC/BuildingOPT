#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 08:11:18 2016

@author: tsz
"""
from __future__ import division
import design_computation_5R1C_multi_obj as opti_model

def run_multi_obj(number_simulations, enev_restrictions=True, pv_scenario=False, folder="results"):
    # Filename definitions
    if pv_scenario:
        tag = "pv"
        filename_start_values = "start_values_pv.csv"
    else:
        if enev_restrictions:
            tag = "enev_restrictions"
            filename_start_values = "start_values_enev.csv"
        else:
            tag = "no_restrictions"
            filename_start_values = "start_values_without_enev.csv"
    
    # Compute limits (min costs, min emissions)
    emissions_max = 1000 # ton CO2 per year
    # Minimize costs
    filename_min_costs = folder + "/" + tag + str(0) + ".pkl"
    options={"filename_results" : filename_min_costs,
             "enev_restrictions": enev_restrictions,
             "pv_scenario": pv_scenario,
             "opt_costs": True,
             "store_start_vals": False,
             "load_start_vals": False,
             "filename_start_vals": filename_start_values}
    (min_costs, max_emissions) = opti_model.optimize(emissions_max, options)

    # Minimize emissions (lexicographic optimization)
    filename_min_emissions = folder + "/" + tag + str(number_simulations+1) + ".pkl"
    options["opt_costs"] = False
    options["store_start_vals"] = True
    options["filename_results"] = filename_min_emissions
    (max_costs, min_emissions) = opti_model.optimize(emissions_max, options)
    # Second optimization to minimize the costs at minimal emissions
    options["opt_costs"] = True
    options["store_start_vals"] = True
    options["load_start_vals"] = True
    options["filename_results"] = filename_min_emissions
    (max_costs, min_emissions) = opti_model.optimize(min_emissions, options)
    
    # Run multiple simulations
    options["opt_costs"] = True
    options["store_start_vals"] = False
    options["load_start_vals"] = True
    prev_emissions = max_emissions
    for i in range(1, 1+number_simulations):
        # Emissions limit is the minimum of:
        # 1. linear interpolation between max_emissions and min_emissions
        # 2. previous iteration's emissions * (1-eps)
        limit_emissions = min(max_emissions - (max_emissions-min_emissions) * i / (number_simulations+1),
                              prev_emissions * 0.999)
    
        options["filename_results"] = folder + "/" + tag + str(i) + ".pkl"
        (costs, prev_emissions) = opti_model.optimize(limit_emissions, options)

if __name__ == "__main__":
    # Possible scenarios:
    # pv    enev
    # True  irrelevant
    # False True
    # False False
    
    # PV scenario
    print "Running PV scenario"
    run_multi_obj(number_simulations=2,
                  enev_restrictions=True,
                  pv_scenario=True)

    # EnEV scenario
    print "Running EnEV scenario"
    run_multi_obj(number_simulations=2,
                  enev_restrictions=True,
                  pv_scenario=False)

    # Without restrictions
    print "Running free scenario"
    run_multi_obj(number_simulations=2,
                  enev_restrictions=False,
                  pv_scenario=False)
