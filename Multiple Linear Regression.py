# import required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sn


df = pd.read_csv("House_price_post_Preprocessing.csv")

x_multi = df.drop("price",axis=1)
y_multi = df['price']
x_multi_cons = sn.add_constant(x_multi)
lm_multi = sn.OLS(y_multi,x_multi_cons).fit()
print(lm_multi.summary())

# Method 2
from sklearn.linear_model import LinearRegression
lm3 = LinearRegression()
lm3.fit(x_multi,y_multi)

