# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 17:44:54 2021

@author: Nishan Kapoor
"""

#########   Question 1    #########

import pandas as pd
import scipy
from scipy import stats


cutlets = pd.read_csv(r"E:\Hypothesis testing\Cutlets.csv")
cutlets

cutlets.columns = "UnitA", "UnitB"

# Normality Test
stats.shapiro(cutlets.UnitA) # Shapiro Test

print(stats.shapiro(cutlets.UnitB))
help(stats.shapiro)

# Variance test
scipy.stats.levene(cutlets.UnitA, cutlets.UnitB)
help(scipy.stats.levene)

# p-value = 0.417 > 0.05 so p high null fly ,we are fail to reject Ho => equal variance

# Performing 2 Sample T test
scipy.stats.ttest_ind(cutlets.UnitA, cutlets.UnitB)
help(scipy.stats.ttest_ind)


###################################################################################

########    Question 2     ##########

############# One - Way Anova ################
import pandas as pd
import scipy
from scipy import stats

Lab= pd.read_csv(r"E:\Hypothesis testing\lab_tat_updated.csv")
Lab
Lab.columns = "Laboratory_1", "Laboratory_2", "Laboratory_3","Laboratory_4"

# Normality Test
stats.shapiro(Lab.Laboratory_1) # Shapiro Test p> 0.05- data normal
stats.shapiro(Lab.Laboratory_2) # Shapiro Test p> 0.05- data normal
stats.shapiro(Lab.Laboratory_3) # Shapiro Test p> 0.05- data normal
stats.shapiro(Lab.Laboratory_4) # Shapiro Test p> 0.05- data normal

# Variance test
help(scipy.stats.levene)
# All 4 labs are being checked for variances
scipy.stats.levene(Lab.Laboratory_1, Lab.Laboratory_2, Lab.Laboratory_3,Lab.Laboratory_4)

# One - Way Anova
F, p = stats.f_oneway(Lab.Laboratory_1, Lab.Laboratory_2, Lab.Laboratory_3, Lab.Laboratory_4)
p

# p value is 2.143 >0.05
# P high null fly
# All the 4 laboratories have equal mean TAT (Turn around time)


###########################################################################################

#######    Question 5     ##########

######### 2-proportion test ###########
import pandas as pd
import numpy as np

fantaloons = pd.read_csv(r"E:\Hypothesis testing\Fantaloons.csv")

from statsmodels.stats.proportion import proportions_ztest

tab1 = fantaloons.Weekdays.value_counts()
tab1
tab2 = fantaloons.Weekend.value_counts()
tab2

# crosstable table
pd.crosstab(fantaloons.Weekdays, fantaloons.Weekend)

count = np.array([120, 47])
nobs = np.array([287, 113])

stats, pval = proportions_ztest(count, nobs, alternative = 'two-sided') 
print(pval) # Pvalue 0.9681

stats, pval = proportions_ztest(count, nobs, alternative = 'larger')
print(pval) # Pvalue = 0.4840

############################################################################################



########      Question 4     ##########
################ Chi-Square Test ################

import pandas as pd
import scipy
from scipy import stats

Telecall = pd.read_csv(r"E:\Hypothesis testing\CustomerOrderform.csv")
Telecall

Telecall_1 = pd.melt(Telecall,value_vars=['Phillippines', 'Indonesia', 'Malta', 'India'])
Telecall_1.columns = [ 'Country', 'Condition']

count = pd.crosstab(Telecall_1["Country"],Telecall_1["Condition"])
count
Chisquares_results = scipy.stats.chi2_contingency(count)

Chi_square = [['Test Statistic', 'p-value'], [Chisquares_results[0], Chisquares_results[1]]]
Chi_square

# pvalue is 0.27>0.05, p high null fly, fail to reject null hypothesis
#########################################################################################



##########      Question 3       #########

################ Chi-Square Test ################

import pandas as pd
import scipy
from scipy import stats

buyerratio = pd.read_csv(r"E:\Hypothesis testing\BuyerRatio.csv")
buyerratio

buyerratio_1 = pd.melt(buyerratio,value_vars=['East', 'West', 'North', 'South'])
buyerratio_1.columns = [ 'observedvalues', 'gender']

count = pd.crosstab(buyerratio_1["observedvalues"], buyerratio_1["gender"])
count
Chisquares_results = scipy.stats.chi2_contingency(count)

Chi_square = [['Test Statistic', 'p-value'], [Chisquares_results[0], Chisquares_results[1]]]
Chi_square

# p>0.05, p high null fly, hence we fail to reject null hypothesis

###############         completed        #####################




