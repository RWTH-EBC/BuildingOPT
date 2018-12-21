# BuildingOPT
Building envelope and energy system optimization package

This package presents the optimization model used in our Applied Energy paper:
Optimal design of energy conversion units and envelopes for residential building retrofits using a comprehensive MILP model

This paper can be downloaded free of charge until December 15th, 2016 with this link:
<http://authors.elsevier.com/a/1Txo815eiekkqr>

DOI: 10.1016/j.apenergy.2016.10.049


## Brief documentation 

### File structure
- input_data is a folder that holds external inputs for the optimization module. It has a separate documentation file that describes each required input.

- design_computation_5R1C_multi_obj.py is the main contribution of this work. Here we have the optimization model described in the referenced paper. All considered energy conversion technologies are hard-coded in this file.

- design_computation_5R1C_statusquo.py provides a benchmark for the developed optimization results. Similar to design_computation_5R1C_multi_obj.py, all considered energy conversion technologies are hard-coded in this file.

- run_multi_obj.py executes design_computation_5R1C_multi_obj.py repeatedly to perform a multi-objective optimization in order to determine tradeoffs between costs and emissions.

- sun.py is an implementation of common equations to describe the sun's position and to transform solar irradiation (direct and diffuse) onto a horizontal surface to the total solar irradiation onto a tilted surface area.

### Results
The results produced by these scripts can be found in the folder results. Each generated .pkl-file holds information regarding the optimal energy conversion system, building envelope and system operation as well as KPIs such as total annualized costs and annual emissions.
