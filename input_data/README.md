# BuildingOPT - inputs

This repository uses two types of inputs: text files and Python files.

## Inputs from text files

* **demand_domestic_hot_water.txt**
  List of hourly hot water demands. All values are specified in kWh for each time step.
* **demand_electricity.txt**
  Array of averaged electricity demands for one year. The specified values are given in kWh. Rows represent months (firts row represents January, twelfth row stands for December) and columns symbolize hours of each day.
* **remuneration_chp.txt**
  Array of averaged feed in remuneration for CHPs. This file is structured similar as demand_elecricity.txt
* **weather.txt**:
  Hourly weather inputs based on the German Test Reference Year for region 05 (city of Essen).

## Inputs from Python files

* **building.py:** Building inputs
  * Geometry
  * Internal gains 
  * Set points for indoor temperature
  * Ventilation rates
* **components.py:** List of all available types of each component from which the solver can choose.
	This implies several types of walls, roofs, etc. and the solver can choose to install exactly one of each component.
  * External walls
  * Ground floor plate
  * Roofs
  * Windows
  * Internal walls
  * Internal ceilings