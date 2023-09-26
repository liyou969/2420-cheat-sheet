# 2420-cheat-sheet

# import
import numpy as np

import pandas as pd

from scipy import stats

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

from matplotlib import cm

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


