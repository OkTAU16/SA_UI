<p align="center">
  <img src="SLM_logo_short.png" >
</p>


******About:******

This repository contains an implementation of the SLM Algorithm described in the paper **"Non equilibrium Self-Assembly Time Forecasting by the Stochastic Landscape Method"**, written by Michael Faran, and Gili Bisker.
The paper can be found here: <https://pubs.acs.org/doi/full/10.1021/acs.jpcb.3c01376>.

The Stochastic Landscape Method (SLM) has been developed to analyze time series data in non-equilibrium self-assembly processes and to predict the time of the first assembly of the target.
This code is a generalized implementation of the Stochastic Landscape Method That can be used on any time series data fitting the requirements.
Key adaptations include developing a user-friendly graphical interface with handling of irregularly spaced time series.


This code was written by Omri Kovarsky and Idan Frenkel, 28/3/2024. For any questions or inquiries, email: <Kovarsk@gmail.com>

******Requirements:******

The program can be used via python or the executable file in the repository. 
python kernel is not needed for running the executable file.

**If the time series is irregularly spaced, an added time vector is required. Your data will be interpolated over a regularly spaced time vector before analysis**

******Input:******

The inputs to the algorithm is time series data, distance data, and sample time in one file in the following order:

**Sample Time:**
(N,1) Numeric containing the elapsed time of each sample.

**Measured Parameter:**
(N,1) Numeric containing the value of the Measured parameter in your system

**Distance:**
(N,M) Numeric containing the distance from the target (by whatever metric), distance at target should be 0.
M represents the Number of Targets.

If your distance vector doesn't reach 0:
1. set distance vector toggle to True
2. Enter an energy threshold: the value of the measured parameter which represent that the target has been reached. 

**If you have no distance vector**:
1. Set distance vector toggle to false
2. Enter an energy threshold: the value of the measured parameter which represent that the target has been reached. 

**The permitted file formats are: .csv,.mat,.xlsx**

******Outputs:******

The model,run log, feature matrix and output graphs will be saved to the selected Output Directory

**Graph 1**
<p align="center">
  <img src="example_output\Predictions Scatter and Histogram.png" >
</p>
The upper plot is a scatter plot of the model’s predictions in all the cross-validation iterations.

The lower plot shows the separation of the prediction into histogram bins. The x-axis is the center of each bin, and the y-axis is the mean of the predictions per bin in every cross-validation iteration. The dashed line is the perfect predictor, and the blue line is the mean of the model’s predictions.


**Graph 2**
<p align="center">
  <img src="example_output\Stochastic Landscape 2D.png" >
</p>


**Graph 3**
<p align="center">
  <img src="example_output\Predictor Evaluation.png" >
</p>
The upper plot, shows the predictions before (round grey) and after (black square) the bias correction. 

The middle plot is a box plot of the CV-corrected predictor separated into histogram bins. The dashed line represents the perfect predictor. As shown, the mean of each bin converges with the ideal predictor. 

The lower plot shows the mean error of the CV-corrected predictor when compared to a naïve predictor in each bin. The color bar represents the relative weight of the data in each bin. The dashed line is the perfect predictor with zero error.

**Running Example data**
set Added Time Series To True
Set Down Sample to False
set CV to True With 10 Itterations
set Number of particle clusters to 3
set Number Of Targets To 2
set distance vector to true and leave the energy threshold unfilled
you will be asked a variable name fo the .mat file
variable name for example data: energy_distance