### Kaggle Competition | [San Francisco Crime Classification](https://www.kaggle.com/c/sf-crime)


### 1. My Conclusion Analysis Report - Jupyter Notebook
* [San Francisco Crime](https://UnimportantCoolProfessional.saeyoonyoon.repl.co)


### 2. About Data :
* This dataset contains incidents derived from SFPD Crime Incident Reporting system. The data ranges from 1/1/2003 to 5/13/2015. The training set and test set rotate every week, meaning week 1,3,5,7... belong to test set, week 2,4,6,8 belong to training set. 


### 3. Process Introduction :
It is a competition that can be said to be Kaggle's introductory period and conducts a Python-based analysis. 

**[My focusing was on]** 
1. EDA - Focusing on dependent variable
2. Correlation coefficient Analysis
3. One - Hot - Encoding
4. Feature engineering(Address, Datetime)
5. Mapping process(folium, plotly) for searching locations 
6. Hold-out Validation 
7. sparse matrix(csr_matrix)
8. Ensemble Model Selection(RandomForestClassifier vs LGBMClassifier)
9. Optimization process
10. 'log_loss' metrics
11. hyperparameter Tuning 
12. predict_proba prediction

**[Dependencies & Tech]:**
* [IPython](http://ipython.org/)
* [NumPy](http://www.numpy.org/)
* [Pandas](http://pandas.pydata.org/)
* [SciKit-Learn](http://scikit-learn.org/stable/)
* [SciPy](http://www.scipy.org/)
* [Seaborn](https://seaborn.pydata.org/)
* [Matplotlib](http://matplotlib.org/)
* [Plotly](https://plotly.com/python/)
* [Folium](https://pypi.org/project/folium/)
* [StatsModels](http://statsmodels.sourceforge.net/)
* [LightGBM](https://lightgbm.readthedocs.io/en/latest/)


### 4. San Francisco Crime Classification
From 1934 to 1963, San Francisco was infamous for housing some of the world's most notorious criminals on the inescapable island of Alcatraz.

Today, the city is known more for its tech scene than its criminal past. But, with rising wealth inequality, housing shortages, and a proliferation of expensive digital toys riding BART to work, there is no scarcity of crime in the city by the bay.

From Sunset to SOMA, and Marina to Excelsior, this competition's dataset provides nearly 12 years of crime reports from across all of San Francisco's neighborhoods. Given time and location, you must predict the category of crime that occurred.

We're also encouraging you to explore the dataset visually. What can we learn about the city through visualizations like this Top Crimes Map? The top most up-voted scripts from this competition will receive official Kaggle swag as prizes. 
