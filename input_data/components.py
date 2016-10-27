# -*- coding: utf-8 -*-
"""
Created on Tue Oct 06 11:23:38 2015

@author: rsa-lsc
"""

import numpy as np

def specificHeatCapacity(d, d_iso, density, cp):
    """ 
    Computation of (specific) heat capacity of each wall-type-surface
    method from ISO 13786:2007 A.2.4
    Result is in J/m2K
    """
    d_t = min(0.5 * np.sum(d), d_iso , 0.1)
    sum_d_i = d[0]
    i = 0 
    kappa = 0       
    while sum_d_i <= d_t:
        kappa += d[i] * density[i] * cp[i]
        i += 1
        sum_d_i += d[i]
    else:
        sum_d_i -=  d[i]
        d_part = d_t - sum_d_i          
        kappa += d_part * density[i] * cp[i]

    return kappa

# Define how many types of components are considered for each different 
# component. 'numberDevices["opaque", "wall"] = 4' implies that 4 different
# types of walls are available and the optimizer has to choose one of these.
numberDevices = {} 
numberDevices["opaque", "wall"] = 4
numberDevices["opaque", "floor"] = 1
numberDevices["opaque", "roof"] = 4
numberDevices["window"] = 4
# intWall and ceiling must have the same amount of components as wall. There is
# only one decision variable for these 3 components to prevent a coupling of 
# massive internal construction and light weight outdoor components
numberDevices["opaque", "intWall"] = numberDevices["opaque", "wall"]
numberDevices["opaque", "ceiling"] = numberDevices["opaque", "wall"]
numberDevices["opaque", "intFloor"] = numberDevices["opaque", "wall"]

# Define properties of each type of component
U = {}
U_win = {}
d = {}
d_iso = {}
rho = {}
cp = {}
Lambda = {}
inv = {}
kappa = {}
g_gl = {}
R_se = {}
R_si = {}
epsilon = {}
alpha_Sc = {}
t_life = {}

attributes_op = [d, d_iso, rho, cp, Lambda, U, kappa, inv, R_se, R_si, epsilon, alpha_Sc, t_life]
attributes_win = [U, inv, g_gl, U_win, R_se, R_si, epsilon, t_life]
opaque_ext = ["wall", "roof", "floor"]
opaque = ["wall", "roof", "floor", "intWall", "ceiling", "intFloor"]
for x in attributes_op:
    for y in opaque:
        x["opaque", y] = [0] * numberDevices["opaque", y]

for x in attributes_win:
    x["window"] = [0] * numberDevices["window"]

# ISO 6946 Table 1, Heat transfer resistances for opaque components
for x in opaque[0:2]:
    for i in range(numberDevices["opaque", x]):
        R_se["opaque", x][i] = 0.04 # m²K/W

for i in range(numberDevices["opaque", "wall"]):
    R_si["opaque", "wall"][i] = 0.13 # m²K/W
for i in range(numberDevices["opaque", "roof"]):
    R_si["opaque", "roof"][i] = 0.13 # m²K/W
for i in range(numberDevices["opaque", "floor"]):
    R_si["opaque", "floor"][i] = 0.13 # m²K/W

# ASHRAE 140 : 2011, Table 5.6 p.19
for i in range(numberDevices["window"]):
    R_se["window"][i] = 0.04 # m²K/W
    R_si["window"][i] = 0.1299 # m²K/W

# ASHRAE 140 : 2011, Table 5.3, page 18 (infrared emittance)
for x in opaque_ext:
    for i in range(numberDevices["opaque", x]):
        epsilon["opaque", x][i] = 0.9
for i in range(numberDevices["window"]):
    epsilon["window"][i] = 0.9

# ASHRAE 140 : 2011, Table 5.3, page 18 (Absorptionskoeffizient opake Fläche)
for x in opaque_ext:
    for i in range(numberDevices["opaque", x]):
        alpha_Sc["opaque", x][i] = 0.6 

# Outer walls
i = 0 # 1969-1978 heavy (existing)
# [ferroconcrete, polystyrene, concrete outside layer]
d["opaque", "wall"][i] = np.array([0.175, 0.03, 0.08]) #[m]
d_iso["opaque", "wall"][i] = sum(d["opaque", "wall"][i][0:1]) # [m]
rho["opaque", "wall"][i] = np.array([2400.0, 20.0, 2200.0]) # [kg/m³]
cp["opaque", "wall"][i] = np.array([1000, 1450, 1000]) # [J/(kgK)]
Lambda["opaque", "wall"][i] = np.array([2.5, 0.04, 1.65]) # [W/(mK)]
inv["opaque", "wall"][i] = sum([0, 0, 0]) # [€/m²] # Existing wall --> no costs
t_life["opaque", "wall"][i] = 40. #years

# The other walls utilize the existing material and extend on the outer side
i = 1
# [ferroconcrete, polystyrene, concrete layer, gypsum plasterboard, sandwich panel]
d["opaque", "wall"][i] = np.array([0.175, 0.03, 0.08, 0.0125, 0.04]) #[m]
d_iso["opaque", "wall"][i] = sum(d["opaque", "wall"][i][0:4]) # [m]
rho["opaque", "wall"][i] = np.array([2400.0, 20.0, 2200.0, 800, 30]) # [kg/m³]
cp["opaque", "wall"][i] = np.array([1000, 1450, 1000, 1000, 1000]) # [J/(kgK)]
Lambda["opaque", "wall"][i] = np.array([2.5, 0.04, 1.65, 0.25, 0.04]) # [W/(mK)]
inv["opaque", "wall"][i] = sum([0, 0, 0, 1.5, 28.5]) # [€/m²]
t_life["opaque", "wall"][i] = 40. #years

i = 2 # EnEV 2016 conform
# [Plastering, ferroconcrete, polystyrene, concrete layer, core insulation, air gap, lime stone]
d["opaque", "wall"][i] = np.array([0.008, 0.175, 0.03, 0.08, 0.15, 0.01, 0.115]) #[m]
d_iso["opaque", "wall"][i] = sum(d["opaque", "wall"][i][0:4]) # [m]
rho["opaque", "wall"][i] = np.array([1600., 2400., 20, 2200, 30., 1.2, 2000.]) # [kg/m³]
cp["opaque", "wall"][i] = np.array([960., 1000., 1450., 1000., 1450, 1000, 880.]) # [J/(kgK)]
Lambda["opaque", "wall"][i] = np.array([0.6, 2.5, 0.04, 1.65, 0.032, 0.067, 0.99]) # [W/(mK)]
inv["opaque", "wall"][i] = sum([2.35, 0, 0, 0, 29.25, 0, 18.39]) # [€/m²]
t_life["opaque", "wall"][i] = 40. #years

i = 3 # Future oriented
# [Plastering, ferroconcrete, polystyrene, concrete layer, core insulation, air gap, lime stone]
d["opaque", "wall"][i] = np.array([0.008, 0.175, 0.03, 0.08, 0.22, 0.01, 0.115]) #[m]
d_iso["opaque", "wall"][i] = sum(d["opaque", "wall"][i][0:4]) # [m]
rho["opaque", "wall"][i] = np.array([1600., 2400., 20, 2200, 30., 1.2, 2000.]) # [kg/m³]
cp["opaque", "wall"][i] = np.array([960., 1000., 1450., 1000., 1450, 1000, 880.]) # [J/(kgK)]
Lambda["opaque", "wall"][i] = np.array([0.6, 2.5, 0.04, 1.65, 0.032, 0.067, 0.99]) # [W/(mK)]
inv["opaque", "wall"][i] = sum([2.35, 0, 0, 0, 42.9, 0, 18.39]) # [€/m²]
t_life["opaque", "wall"][i] = 40. #years


# Internal wall
# [Lime plaster, concrete, lime plaster]
for i in range(numberDevices["opaque", "intWall"]):
    d["opaque", "intWall"][i] = np.array([0.01, 0.10, 0.01]) #[m]
    d_iso["opaque", "intWall"][i] = sum(d["opaque", "intWall"][i]) # [m]
    rho["opaque", "intWall"][i] = np.array([1600., 1300., 1600.]) # [kg/m³]
    cp["opaque", "intWall"][i] = np.array([1000., 1000., 1000.]) # [J/(kgK)]
    inv["opaque", "intWall"][i] = sum([0, 0, 0]) # [€/m²]
    t_life["opaque", "intWall"][i] = 40. #years


# Ceiling
# [Concrete, polystyrene, cement screed]
for i in range(numberDevices["opaque", "ceiling"]):
    d["opaque", "ceiling"][i] = np.array([0.16, 0.06, 0.04]) #[m]
    d_iso["opaque", "ceiling"][i] = d["opaque", "ceiling"][i][0] # [m]
    rho["opaque", "ceiling"][i] = np.array([2000., 30., 2000.]) # [kg/m³]
    cp["opaque", "ceiling"][i] = np.array([1000., 1450., 1000.]) # [J/(kgK)]
    inv["opaque", "ceiling"][i] = sum([0, 0, 0]) # [€/m²]
    t_life["opaque", "ceiling"][i] = 40. # years


# Floor
# [Cement screed, polystyrene, concrete]
for i in range(numberDevices["opaque", "intFloor"]):
    d["opaque", "intFloor"][i] = np.array([0.04, 0.06, 0.16]) #[m]
    cp["opaque", "intFloor"][i] = np.array([1000., 1450., 1000.]) # [J/(kgK)]
    Lambda["opaque", "intFloor"][i] = np.array([1.4, 0.40, 2]) # [W/(mK)]
    inv["opaque", "intFloor"][i] = sum([0, 0, 0]) # [€/m²]
    t_life["opaque", "intFloor"][i] = 40. # years
    d_iso["opaque", "intFloor"][i] = d["opaque", "intFloor"][i][0]
    rho["opaque", "intFloor"][i] = np.array([2000., 30., 2000.]) # [kg/m³]


# Ground floor
# [Cement screed, polystyrene, concrete]
i = 0
d["opaque", "floor"][i] = np.array([0.04, 0.03, 0.15]) #[m]
d_iso["opaque", "floor"][i] = d["opaque", "floor"][i][0] # [m]
rho["opaque", "floor"][i] = np.array([2000., 30., 2400.]) # [kg/m³]
cp["opaque", "floor"][i] = np.array([1000., 1000., 1000.]) # [J/(kgK)]
Lambda["opaque", "floor"][i] = np.array([1.4, 0.04, 2.5]) # [W/(mK)]
inv["opaque", "floor"][i] = sum([0, 0, 0]) # [€/m²] 
t_life["opaque", "floor"][i] = 40. # years


# Roofs:
# 1969-1978 heavy (ferroconcrete, foam glas, gravel fill)
i = 0
d["opaque", "roof"][i]= np.array([0.15, 0.07, 0.03]) #[m]
d_iso["opaque", "roof"][i] = d["opaque", "roof"][i][0] # [m]
rho["opaque", "roof"][i] = np.array([2400., 120., 1800.]) # [kg/m³]
cp["opaque", "roof"][i] = np.array([1000., 1000., 1000.]) # [J/(kgK)]
Lambda["opaque", "roof"][i] = np.array([2.5, 0.04, 0.7]) # [W/(mK)]
inv["opaque", "roof"][i] = sum([0, 0, 0]) # [€/m²]
t_life["opaque", "roof"][i] = 40. # years

i = 1 # (ferroconcrete, foam glas, gravel fill)
d["opaque", "roof"][i]= np.array([0.15, 0.09, 0.03]) #[m]
d_iso["opaque", "roof"][i] = d["opaque", "roof"][i][0] # [m]
rho["opaque", "roof"][i] = np.array([2400., 120., 1800.]) # [kg/m³]
cp["opaque", "roof"][i] = np.array([1000., 1000., 1000.]) # [J/(kgK)]
Lambda["opaque", "roof"][i] = np.array([2.5, 0.04, 0.7]) # [W/(mK)]
inv["opaque", "roof"][i] = sum([0, 50, 0]) # [€/m²]
t_life["opaque", "roof"][i] = 40. # years

i = 2 # EnEV 2016 conform - (ferroconcrete, insulation board, gravel fill)
d["opaque", "roof"][i]= np.array([0.15, 0.24, 0.03]) #[m]
d_iso["opaque", "roof"][i] = d["opaque", "roof"][i][0] # [m]
rho["opaque", "roof"][i] = np.array([2400., 100., 1800.]) # [kg/m³]
cp["opaque", "roof"][i] = np.array([1000., 1030., 1000.]) # [J/(kgK)]
Lambda["opaque", "roof"][i] = np.array([2.5, 0.037, 0.7]) # [W/(mK)]
inv["opaque", "roof"][i] = sum([0, 67.2, 0]) # [€/m²]
t_life["opaque", "roof"][i] = 40. # years

i = 3 # Future oriented - (ferroconcrete, insulation board, gravel fill)
d["opaque", "roof"][i]= np.array([0.15, 0.36, 0.03]) #[m]
d_iso["opaque", "roof"][i] = d["opaque", "roof"][i][0] # [m]
rho["opaque", "roof"][i] = np.array([2400., 100., 1800.]) # [kg/m³]
cp["opaque", "roof"][i] = np.array([1000., 1030., 1000.]) # [J/(kgK)]
Lambda["opaque", "roof"][i] = np.array([2.5, 0.037, 0.7]) # [W/(mK)]
inv["opaque", "roof"][i] = sum([0, 100, 0]) # [€/m²]
t_life["opaque", "roof"][i] = 40. # years


# Windows:
# Wooden frame, double glazing (existing)
i = 0
U_win["window"][i] = 2.80 # [W/K]
g_gl["window"][i] = 0.78
inv["window"][i] = 0. # [€/m²]
t_life["window"][i] = 40. # years

# Plastics frame, insulation glazing
i = 1
U_win["window"][i] = 1.90 # [W/K] # http://www.energiesparaktion.de/wai3/showcontent.asp?ThemaID=4803
g_gl["window"][i] = 0.67
inv["window"][i] = 60. # [€/m²] # (approximation)
t_life["window"][i] = 40. # years

# EnEV 2016 conform
# Double insulation glazing 
# http://www.fensterdepot24.de/thermomax-5classic-kunststofffenster.html
i = 2
U_win["window"][i] = 1.10 # [W/K]
g_gl["window"][i] = 0.575
inv["window"][i] = 140. # [€/m²]
t_life["window"][i] = 40. # years

# Future oriented
# Triple insulation glazing
# http://www.fensterdepot24.de/thermomax-5classic-kunststofffenster.html
i = 3
U_win["window"][i] = 0.7 # [W/K]
g_gl["window"][i] = 0.5
inv["window"][i] = 155. # [€/m²]
t_life["window"][i] = 40. # years


# Compute U and kappa for each component
for x in opaque_ext:
    for i in range(numberDevices["opaque", x]):
        kappa["opaque", x][i] = specificHeatCapacity(d["opaque", x][i], 
                                                     d_iso["opaque", x][i], 
                                                     rho["opaque", x][i], 
                                                     cp["opaque", x][i])
        U["opaque", x][i] = 1.0 / (R_si["opaque", x][i] + 
                                   sum(d["opaque", x][i] / Lambda["opaque", x][i]) + 
                                   R_se["opaque", x][i])

for x in ["intWall", "ceiling", "intFloor"]:
    for i in range(numberDevices["opaque", "wall"]):
        kappa["opaque", x][i] = specificHeatCapacity(d["opaque", x][i], 
                                                     d_iso["opaque", x][i], 
                                                     rho["opaque", x][i], 
                                                     cp["opaque", x][i])

for i in range(numberDevices["window"]):
    U["window"][i] = 1.0 / (R_si["window"][i] + 1.0 / U_win["window"][i] + R_se["window"][i])
