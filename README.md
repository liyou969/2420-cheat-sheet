# 2420-cheat-sheet

# import
import numpy as np

import pandas as pd

from scipy import stats

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt


import seaborn as sns

# boxplot
df = pd.read_excel('a.xlsx')

df['a'] = df['Letters'].where(df['Letters']=='a','notA')

grouped = df.groupby('a')['Numbers'].apply(list).to_dict()

fig = plt.figure(figsize=(22,14))

plt.boxplot([grouped['a'], grouped['notA']])

plt.xticks([1,2],["a", "Not A"])

plt.title("Boxplot of a and not_a")

plt.ylabel("Number of day stay at hospital")

plt.xlabel("If they use something")

plt.show()

from scipy.stats import ttest_1samp


t,p = stats.ttest_1samp(SeaIce["Sum_Arctic"], 10)

print("The result of one sample test's t_value is",t)

print("The result of one sample test's p_value is",p)

t1,p1 = stats.ttest_ind(SeaIce["Sum_Antarctic"],SeaIce["Sum_Arctic"])

print("The result of two sample test's is", t1)

print("The result of two sample test's p value is", p1)

# 统计测试

#correlation

from scipy.stats import pearsonr

corr, p_value = pearsonr(data[""],data[""])

print("The correlation is ",corr, " units" )

print("The p_value is ",p_value)

print("R square: ", corr * corr)



#T-test
'''
If the populations are normally distributed or nearly so, and want to compare the mean of one population with the mean of another population,
then a t-test can be used (cf. nonparametric Wilcoxon test). 

Null Hypothesis: The means of both populations are equal.

Alternate Hypothesis: The means of both populations are not equal.
 
A large t-score tells you that the groups are different.

A small t-score tells you that the groups are similar.

用p value 判断是否reject
'''

T test performs a hypothesis test for the mean between two independent groups of scores 
eg. claiming the average marks between two similar courses are the same

t, p = stats.ttest_ind(p24_300mg, p24_600mg)


performs a hypothesis test for the mean between two related groups of scores
eg. claiming that a particular student's average marks in two different courses are the same

t, p = stats.ttest_rel(p24_300mg, p24_600mg)

stats.ttest_1samp(data[''],76)

p value

The p-value is below 0.05, so reject the null hypothesis: the means of both ... are not equal

p-value is larger than 0.05 so it is out of the rejection region, 

thus we can not reject the null hypothesis and decide that the  is equivalent between  and .
