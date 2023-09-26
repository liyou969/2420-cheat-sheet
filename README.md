<h1 align='center'> COMP2420/COMP6420 - Introduction to Data Management, Analysis and Security</h1>

<h2 align='center'> Assignment 1 - 2023</h2>


### Discussion

Use the *assignment_1* folder in Piazza discussions.  Check to see if your question has already been answered before starting a new topic.

# import
import numpy as np
<h2 align='center'> Assignment 1 - 2023</h2>

import pandas as pd

from scipy import stats

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt


import seaborn as sns

# output
df.to_excel('output.xlsx', index=False, engine='openpyxl')

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


# 画图sns

Line Plot: trends and relationships of continuous variables, such as time series data or variables changing with a parameter.

Scatter Plot: relationship between two continuous variables, helping to observe correlations or distributions between variables.
如果你想展示变量之间的关系(强)和趋势，散点图可能更合适；如果你想展示一个变量随着另一个变量的变化而变化的趋势，折线图可能更适合。

！Bar Plot: plt.bar()适用于比较不同类别或组之间的离散数据。
plt.hist()适用于展示连续变量的分布情况。

Histogram: display the distribution of numerical data, helping to understand the central tendency and dispersion of the data.

Box Plot:display the distribution of numerical data, including median, quartiles, and outliers, allowing the observation of outliers and distribution shapes.

Heatmap: Used to show the relationship between two categorical variables, often using colors to represent the degree of association or frequency.

Violin Plot: Combines the features of a box plot and a kernel density plot, used to display the distribution and density of numerical data.

Categorical Plot: Includes bar plots, count plots, box plots, etc., used to display data distribution and relationships between different categories.
'''
#hue for different lines
sns.lineplot(x = '', y='', hue = '' , data=q3q4_df)
sns.lineplot(x=x, y=y)
plt.title('')
plt.show()

sns.scatterplot(x=x, y=y)

sns.barplot(x=x, y=y)

sns.histplot(data)

sns.boxplot(data=data)

sns.heatmap(data, cmap='YlGnBu', annot=True, fmt='.2f')

sns.violinplot(x=x, y=y,hue='')


# date
#start of week2
start='2023-02-27'

#end of week2
end='2023-03-03'

#create a new column in dataframe to record the date of each row
SeaIce["Date"] = pd.to_datetime(SeaIce[["Year","Month","Day"]])


is_date = (SeaIce['Date'] >= start) & (SeaIce["Date"] <= end)

# plot more than one graph in one pic

x = SeaIce["Date"]
y = SeaIce["Extent(Antarctic)"]
z = SeaIce["Extent(Arctic)"]

plt.plot(x,y,label="Antarctic")
plt.plot(x,z,label="Arctic")

plt.xlabel("Date")
plt.ylabel("Sea Ice extents(10^6 sq km)")
plt.title("Daily trend of the Antarctic and Arctic sea ice extents")

plt.legend()
plt.show()
