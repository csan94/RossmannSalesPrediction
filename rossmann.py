# Import packages
import numpy as np # library for working with arrays, matrices (linear algebra)
import pandas as pd # library for data manipulation and analysis
import seaborn as sns # library for statistical data visualization
import matplotlib.pyplot as plt # library for plots

# Description of the main task:
# prediction of Rossmann sales data using a machine learning model

# Loading the data
stores = pd.read_csv("store.csv")
sales = pd.read_csv("rossmann_train.csv")

# Printing the number of stores
print('\nThere are ' + str(stores.shape[0]) + ' stores.\n')

# Printing the  first 10 rows of store data to check
print('The data about the stores is the following:\n',stores.head(10))

# Printing the first 10 rows of sales data to check
print('\nThe data about the sales is the following:\n',sales.head(10))

# Merging the data
rossmann = pd.merge(sales, stores, left_on = 'Store', right_on = 'Store', how = 'inner')

# Printing the new data shape
print('\nNumber of all rows and columns: ' + str(rossmann.shape[0]) + ', ' + str(rossmann.shape[1]) + '\n')

# Printing the start and end date
print('The start date is: ', rossmann['Date'].sort_values(ascending = True).iloc[0], '\n')
print('The end date is: ', rossmann['Date'].sort_values(ascending = True).iloc[-1],'\n')

# Split the dates
rossmann['Year'] = [int(i.split('-')[0]) for i in rossmann['Date']]
rossmann['Month'] =  [int(i.split('-')[1]) for i in rossmann['Date']]
rossmann['Day'] =  [int(i.split('-')[2]) for i in rossmann['Date']]

# Posibble values of column StateHoliday
print('Possible values of StateHoliday: ', rossmann['StateHoliday'].unique(), '-> the column gets deleted because of lack of information\n')

# Deleting column StateHoliday because of lack of information
rossmann = rossmann.drop('StateHoliday', axis = 'columns')

# Basic statistics
print(rossmann.describe())

# Sales by stores
fig = plt.figure()
plt.title('Sales vs Stores')
plt.ylabel('Store')
sns.barplot( x = rossmann['Store'], y = rossmann['Sales'])
plt.savefig('SalesVsStores.png')

# Sales/Customer distribution
fig = plt.figure()
plt.hist(rossmann['Sales'])
plt.ylabel('Count')
plt.xlabel('Sales')
plt.title('Sales distribution')
plt.savefig('SalesDistr.png')

fig = plt.figure()
plt.hist(rossmann['Customers'])
plt.ylabel('Count')
plt.xlabel('Customers')
plt.title('Customers distribution')
plt.savefig('CustomerDistr.png')

# Sales vs Customers scatterplot
fig = plt.figure()
plt.scatter(rossmann['Customers'],rossmann['Sales'], color = 'green')
plt.ylabel('Sales')
plt.xlabel('Customers')
plt.title('Sales vs Customers')
plt.savefig('SalesVsCust.png')

# The subplot grid and figure sizes

x_values=('Promo','Promo2','DayOfWeek','SchoolHoliday','StoreType','Assortment')
y_values=('Sales','Customers')

fig, axs = plt.subplots(len(y_values), len(x_values))

for r in range(len(y_values)):
    for c in range(len(x_values)):
         i = r*len(x_values) + c # index to go through the number of columns
         ax = axs[r][c] # show where to position each plots
         ax.set_title(x_values[c] + ' vs ' + y_values[r])
         sns.barplot( x = rossmann[x_values[c]], y = rossmann[y_values[r]], ax = ax)
plt.tight_layout()
plt.show()


# Sales vs Month
fig = plt.figure()
sns.barplot( x = rossmann['Month'], y = rossmann['Sales'])
plt.ylabel('Sales')
plt.xlabel('Month')
plt.title('Sales vs Month')
plt.savefig('SalesVsMonth.png')

# Min and max value of competition distance
print('\nThe maximum distance of competition is:', rossmann['CompetitionDistance'].max(), 'meters.')
print('\nThe minimum distance of competition is:', rossmann['CompetitionDistance'].min(), 'meters.\n')

# Sales/customers in case of store with competition closer than 200, 1000 meters and otherwise
fig = plt.figure()
y_sales = [ sum(rossmann['Sales'][i] for i in range(len(rossmann['Sales'])) if rossmann['CompetitionDistance'][i] < 200.0  ),
sum(rossmann['Sales'][i] for i in range(len(rossmann['Sales'])) if 200.0 <= rossmann['CompetitionDistance'][i] < 1000.0  ),
sum(rossmann['Sales'][i] for i in range(len(rossmann['Sales'])) if 1000.0 < rossmann['CompetitionDistance'][i] ) ]
x_sales = [ ' < 200 m', '> 200, but < 1000 m', '> 1000 m']

fig.suptitle('Sales vs comp. distance')
plt.xlabel('Competition distance')
plt.ylabel('Sales')
sns.barplot( x = x_sales, y = y_sales)
plt.savefig('SalesCompDist.png')

fig = plt.figure()
y_sales = [ sum(rossmann['Customers'][i] for i in range(len(rossmann['Sales'])) if rossmann['CompetitionDistance'][i] < 200.0  ),
sum(rossmann['Customers'][i] for i in range(len(rossmann['Sales'])) if 200.0 <= rossmann['CompetitionDistance'][i] < 1000.0  ),
sum(rossmann['Customers'][i] for i in range(len(rossmann['Sales'])) if 1000.0 < rossmann['CompetitionDistance'][i] ) ]
x_sales = [ ' < 200 m', '> 200, but < 1000 m', '> 1000 m']

fig.suptitle('Customers vs comp. distance')
plt.xlabel('Competition distance')
plt.ylabel('Customers')
sns.barplot( x = x_sales, y = y_sales)
plt.savefig('CustCompDist.png')

# Machine learning part
# Count the empty values in each column
print(rossmann.isna().sum())

# Drop the columns
rossmann = rossmann.drop(['Date','Open','CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear','PromoInterval'], axis=1)

# Remove the rows with missing value_counts
rossmann = rossmann.dropna( subset = ['CompetitionDistance'] )

# Count the new number of rows and columns in the data set
print('\nThe shape of the dataset is: ',rossmann.shape)

# Look at the data types
print('\nThe used data types are:\n ',rossmann.dtypes)

# Print the unique values in the columns
print('\nThe unique values of StoreType: ',rossmann['StoreType'].unique())
print('\nThe unique values of Assortment: ',rossmann['Assortment'].unique())

# LabelEncoder
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

# Encode the StoreType column
rossmann.iloc[:,6] = labelencoder.fit_transform(rossmann.iloc[:,6].values)

# Encode the Assortment column
rossmann.iloc[:,7] = labelencoder.fit_transform(rossmann.iloc[:,7].values)

# Print the unique values in the columns
print('\nThe encoded values of StoreType: ',rossmann['StoreType'].unique())
print('\nThe encoded values of Assortment: ',rossmann['Assortment'].unique())

# Look at the data types
print('\nAfter encoding the used data types are:\n ',rossmann.dtypes)

# Split the data into independent 'X' and dependent 'Y' variables
X_train = rossmann.drop(['Sales'], axis = 1).values
Y_train = rossmann.iloc[:,2].values

# Scale the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
Y_train = sc.fit_transform(Y_train.reshape(-1,1)).ravel()

# Create a function with many machine learning models
def models(X_train, Y_train):

    from sklearn.ensemble import RandomForestRegressor
    regr = RandomForestRegressor(max_depth = 10, random_state = 0)
    regr.fit(X_train,Y_train)

    # Print the training accuracy
    print('\nRandom Forest Regressor: ', regr.score(X_train,Y_train))

    return regr

# Get and train all of the models
model = models(X_train,Y_train)

# Get feature importance
regr = model
importances = pd.DataFrame({'importance': np.round(regr.feature_importances_,3), 'feature': rossmann.drop(['Sales'], axis = 1).columns})
importances = importances.sort_values('importance',ascending = False).set_index('feature')
print('\nThe impoartances are: \n',importances)

# Visualize the importances
fig = plt.figure()
importances.plot.bar()
plt.savefig('importance.png')
plt.show()

# Test set
# Loading the data
test_set = pd.read_csv("rossmann_test.csv")
rossmann_test = pd.merge(test_set, stores, left_on = 'Store', right_on = 'Store', how = 'inner')

# Split the dates
rossmann_test['Year'] = [int(i.split('-')[0]) for i in rossmann_test['Date']]
rossmann_test['Month'] =  [int(i.split('-')[1]) for i in rossmann_test['Date']]
rossmann_test['Day'] =  [int(i.split('-')[2]) for i in rossmann_test['Date']]

# Drop the columns
rossmann_test = rossmann_test.drop(['Date','Open','StateHoliday','CompetitionOpenSinceMonth','CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear','PromoInterval'], axis=1)

# Count the empty values in each column
print('\nNumber of NaN:\n',rossmann_test.isna().sum())

# Remove the rows with missing value_counts
rossmann_test = rossmann_test.dropna( subset = ['CompetitionDistance'] )

# Encode the StoreType column
rossmann_test.iloc[:,5] = labelencoder.fit_transform(rossmann_test.iloc[:,5].values)

# Encode the Assortment column
rossmann_test.iloc[:,6] = labelencoder.fit_transform(rossmann_test.iloc[:,6].values)

# Print predection of my survival using Random Forest Classifier method
pred = model.predict(rossmann_test)
rossmann_test['Prediction'] = sc.inverse_transform(pred)

# Writing out the prediction into a file
rossmann_test.to_csv('rossmann_pred.csv')
print('\nPrediction data is written out into: rossmann_pred.csv')
