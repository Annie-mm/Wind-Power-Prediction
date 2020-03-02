# Wind-Power-Prediction
ENE418 Assignment #2

Data processing and machine learning:
- Data pre-processing
- Implementation of machine learning algorithms
- Discussion of results
- Report

Data: Stochastic wind data
- Weather data (European Centre for Medium-range Weather Forecasts)
	- Wind forecasts at 2 heights (10m, 100m)
	- Zonal and meridional components
		- Correspond to the projection of the wind vector on West-East 
		  and South-North axes, respectively
- Power observations that were collected at wind farm
- Real wind power data (Normalized by the wind farm nominal capacity)

Objective:
- Generate wind power forecasts for a given period based on 2 years of data

Training Model:
- Learn relationship between weather variables and power produced
- Weather forecasts -> Model -> Power forecast over the evaluation period

Data description:
"Assignment2.csv" 
Observed power (normalized) associated with its time stamp which gives the date and time of hourly wind power.

	Variables:
	- U10: zonal component of the wind forecast (West-East projection) at 10 m above
	ground level [m/s]
	- V10: meridional component of the wind forecast (South-North projection) at 10 m
	above ground level [m/s]
	- These two variables (U10, V10) can be used to calculate the wind direction at  10 m above the ground (WD10)
	- WS10: wind speed at 10 m above ground level [m/s]
	- U100: zonal component of the wind forecast (West-East projection) at 100 m above ground level [m/s]
	- V100: meridional component of the wind forecast (South-North projection) at 100 m above ground level [m/s]
	- WS100: wind speed at 100 m above ground level [m/s]
	- TEMP2: temperature at 2 m above the ground level [degrees C]
	- PRESS: Air pressure [kPa]

Tasks:

1. Preprocessing of missing values and outliers
2. Apply techniques to understand relationships between variables and their time lag
   Visualize results
3. Focus is on relationship between wind power and wind speed.
   Apply ML techniques to find the relationship between power generation and
   wind variables.
   Free to choose the inputs (WS10, WS100, WD10) to predict the wind power based on results.
   ML techniques: Linreg, multiple linreg, polynomial linreg, kNN, SVR
   Use error metric RMSE, NRMSE and R^2 to evaluate and compare the predicted accuracy among the machine learning approaches. 
4. Convert power data to categorical data (0,1) based on having high power
   treshold of 0.7, then apply logistic regression technique fitting the categorical data and the associated weather parameters and evaluate the model using the confusion matrix statistics.
5. Develop a wind power generation forecasting based on only the historical wind
   power data to predict power at (t+6h)
