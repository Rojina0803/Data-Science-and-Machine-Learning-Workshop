#import data
#EDA
#Model building
#Model Evaluation
#Pickle

#importing libraries
# pip3 install pandas
# pip3 install numpy
# pip3 install seaborn
# pip3 install matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as lib
import seaborn as sns

#Quick check for null values
# us_housing.isnull().sum()

us_housing = pd.read_csv("USA_Housing.csv")
us_housing.head()
us_housing.columns
list(us_housing.columns)
us_housing.shape
us_housing.info()
us_housing.describe()

sns.pairplot(us_housing)
plt.show()


#Model Building and Training
us_housing.columns
X = us_housing[['Avg. Area Income', 
                'Avg. Area House Age', 
                'Avg. Area Number of Rooms',
                'Avg. Area Number of Bedrooms', 
                'Area Population']]

y = us_housing['Price']


# Train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3
)

from sklearn.linear_model import LinearRegression
ln = LinearRegression()
ln.fit(X_train, y_train)

#Model Eval
coff_df=pd.DataFrame(ln.coef_,X.columns,columns=["Coefficient"])
coff_df

