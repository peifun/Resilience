## Core rationale
Resilience is quantified using the 1-lag temporal autocorrelation (TAC) of kNDVI. Lower TAC values indicate faster recovery from disturbances and therefore higher resilience.
The central hypothesis of this study is that increases in fractional vegetation cover (FVC) enhance forest resilience, whereas elevated temperature variability (AT_CV) weakens this beneficial effect. Specifically, we hypothesize that temperature variability negatively modulates the relationship between FVC and resilience, resulting in a negative FVC × AT_CV interaction.
## Scripts
### 1. Calculate TAC.py
Calculates pixel-wise 1-lag temporal autocorrelation (TAC) from kNDVI time series using the `statsmodels.tsa.acf` function. The script outputs a resilience raster named `kNDVI_TAC.tif`.
### 2. Kendall's tau.py
Performs pixel-wise Mann–Kendall trend analysis on climate variables and fractional vegetation cover time series using the `pymannkendall` package. The script generates raster outputs of Kendall's τ statistics and their corresponding p-values.
### 3. XGBoost.py
Trains an XGBoost regression model to predict forest TAC using fractional vegetation cover and climate-related predictors, including the variability, mean state, and temporal autocorrelation metrics of air temperature (AT), precipitation (PRE), surface solar radiation downward (SSRD), and vapor pressure deficit (VPD). In total, 13 predictors are included. SHAP analysis is used to quantify feature importance, and model hyperparameters are optimized through five-fold grid-search cross-validation.
### 4. interaction.py
Evaluates the interaction between fractional vegetation cover and temperature variability using ordinary least squares (OLS) regression combined with 100 bootstrap iterations. The script estimates the effects of FVC on resilience under low (−1 SD), mean, and high (+1 SD) levels of temperature variability through simple slope analysis. Bootstrapped 95% confidence intervals are used to assess the robustness and significance of the estimated coefficients.
