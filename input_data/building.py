# -*- coding: utf-8 -*-
"""
Created on Wed Oct 07 17:25:45 2015

@author: rsa-lsc
"""
import numpy as np
directions = ("south", "west", "north", "east")

#%% Geometry (symmetric)
A = {} # in m2
A["opaque", "south"] = 42.25
A["opaque", "west"] = 42.25
A["opaque", "north"] = 42.25
A["opaque", "east"] = 42.25

A["opaque", "roof"] = 99.75
A["opaque", "floor"] = 99.75

A["window", "south"] = 7.5
A["window", "west"] = 7.5
A["window", "north"] = 7.5
A["window", "east"] = 7.5
A["window", "roof"] = 0.
A["window", "floor"] = 0.

A["opaque", "wall"] = sum([A["opaque", direction] for direction in directions])
A["opaque", "intWall"] = 375.
A["opaque", "ceiling"] = 75.
A["opaque", "intFloor"] = 75.
A["window"] = sum([A["window", direction] for direction in directions])
A["f"] = 150.

V = 450 # in m3 (A_f * heightWalls)

# S, W, N, E, Roof, PV/STC
beta = [90, 90, 90, 90, 0, 35]  # beta: slope angle 
gamma = [0, 90, 180, 270, 0, 0] # gamma: surface azimuth angle

# Form factor for radiation between the element and the sky
# (DIN EN ISO 13790, section 11.4.6, page 73)
# No direct interaction between sun and floor, therefore the 
# corresponding F_r entry is zero.
F_r   = {"south" : 0.5,
         "west"  : 0.5,
         "north" : 0.5,
         "east"  : 0.5,
         "roof"  : 1.0,
         "floor" : 0.0}

#%% Internal gains: DIN EN ISO 13790, Table G.8, page 140
no_living_kitchen = 1
no_other_rooms = 11
no_total = no_living_kitchen + no_other_rooms
phi_int_relative = np.zeros(24)
phi_int_relative[0:7] = (8 * no_living_kitchen + 1 * no_other_rooms) / no_total
phi_int_relative[7:17] = (20 * no_living_kitchen + 1 * no_other_rooms) / no_total
phi_int_relative[17:24] = (2 * no_living_kitchen + 6 * no_other_rooms) / no_total
phi_int = phi_int_relative * A["f"] # in W

#%% Power generation / consumption in kW (positive, device dependent)
power = {}  
# Electricity demand in kW
power["House"] = np.loadtxt("input_data/demand_electricity.txt")

#%% Temperature boundaries and ventilation rate (constant)
T_set_min = 20 # °C
T_set_max = 27 # °C
ventilationRate = 0.5  # Air exchanges per hour

#%% Approximated room temperature
T_i_appr = np.zeros(12)
T_i_appr[0:4] = 20  # °C
T_i_appr[4:9] = 27  # °C
T_i_appr[9:12] = 20 # °C
