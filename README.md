```python
from platform import python_version
print("Python version:", python_version())
```

    Python version: 3.11.7



```python
# Importing libraries

import numpy as np
import pandas as pd

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

from math import sqrt
from sqlalchemy import create_engine
from sklearn.impute import SimpleImputer
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
```


```python
connection_string = f'postgresql://b_madhura:Qazxsw@123@localhost:5432/drivers_db'
engine = create_engine(connection_string)
query = f'SELECT * FROM driver_historical_data'

# Read the data 
df = pd.read_sql(query, engine)

# Create a copy
historical_df = df.copy()
```


```python
# structure of data
print("The dataset has {} records with {} fields.".format(*df.shape))
```

    The dataset has 197428 records with 16 fields.



```python
# Columns in the dataset
df.columns
```




    Index(['market_id', 'created_at', 'actual_delivery_time', 'store_id',
           'store_primary_category', 'order_protocol', 'total_items', 'subtotal',
           'num_distinct_items', 'min_item_price', 'max_item_price',
           'total_onshift_dashers', 'total_busy_dashers',
           'total_outstanding_orders', 'estimated_order_place_duration',
           'estimated_store_to_consumer_driving_duration'],
          dtype='object')




```python
# Index, Datatype and Memory information
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 197428 entries, 0 to 197427
    Data columns (total 16 columns):
    market_id                                       196441 non-null float64
    created_at                                      197428 non-null object
    actual_delivery_time                            197421 non-null object
    store_id                                        197428 non-null int64
    store_primary_category                          192668 non-null object
    order_protocol                                  196433 non-null float64
    total_items                                     197428 non-null int64
    subtotal                                        197428 non-null int64
    num_distinct_items                              197428 non-null int64
    min_item_price                                  197428 non-null int64
    max_item_price                                  197428 non-null int64
    total_onshift_dashers                           181166 non-null float64
    total_busy_dashers                              181166 non-null float64
    total_outstanding_orders                        181166 non-null float64
    estimated_order_place_duration                  197428 non-null int64
    estimated_store_to_consumer_driving_duration    196902 non-null float64
    dtypes: float64(6), int64(7), object(3)
    memory usage: 24.1+ MB



```python
# Eyeballing the data
with pd.option_context('display.max_rows', 10, 'display.max_columns', df.shape[1]):
    display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>market_id</th>
      <th>created_at</th>
      <th>actual_delivery_time</th>
      <th>store_id</th>
      <th>store_primary_category</th>
      <th>order_protocol</th>
      <th>total_items</th>
      <th>subtotal</th>
      <th>num_distinct_items</th>
      <th>min_item_price</th>
      <th>max_item_price</th>
      <th>total_onshift_dashers</th>
      <th>total_busy_dashers</th>
      <th>total_outstanding_orders</th>
      <th>estimated_order_place_duration</th>
      <th>estimated_store_to_consumer_driving_duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.0</td>
      <td>2015-02-06 22:24:17</td>
      <td>2015-02-06 23:27:16</td>
      <td>1845</td>
      <td>american</td>
      <td>1.0</td>
      <td>4</td>
      <td>3441</td>
      <td>4</td>
      <td>557</td>
      <td>1239</td>
      <td>33.0</td>
      <td>14.0</td>
      <td>21.0</td>
      <td>446</td>
      <td>861.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2.0</td>
      <td>2015-02-10 21:49:25</td>
      <td>2015-02-10 22:56:29</td>
      <td>5477</td>
      <td>mexican</td>
      <td>2.0</td>
      <td>1</td>
      <td>1900</td>
      <td>1</td>
      <td>1400</td>
      <td>1400</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>446</td>
      <td>690.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>3.0</td>
      <td>2015-01-22 20:39:28</td>
      <td>2015-01-22 21:09:09</td>
      <td>5477</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1</td>
      <td>1900</td>
      <td>1</td>
      <td>1900</td>
      <td>1900</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>446</td>
      <td>690.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3.0</td>
      <td>2015-02-03 21:21:45</td>
      <td>2015-02-03 22:13:00</td>
      <td>5477</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>6</td>
      <td>6900</td>
      <td>5</td>
      <td>600</td>
      <td>1800</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>446</td>
      <td>289.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>3.0</td>
      <td>2015-02-15 02:40:36</td>
      <td>2015-02-15 03:20:26</td>
      <td>5477</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>3</td>
      <td>3900</td>
      <td>3</td>
      <td>1100</td>
      <td>1600</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>9.0</td>
      <td>446</td>
      <td>650.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>197423</td>
      <td>1.0</td>
      <td>2015-02-17 00:19:41</td>
      <td>2015-02-17 01:24:48</td>
      <td>2956</td>
      <td>fast</td>
      <td>4.0</td>
      <td>3</td>
      <td>1389</td>
      <td>3</td>
      <td>345</td>
      <td>649</td>
      <td>17.0</td>
      <td>17.0</td>
      <td>23.0</td>
      <td>251</td>
      <td>331.0</td>
    </tr>
    <tr>
      <td>197424</td>
      <td>1.0</td>
      <td>2015-02-13 00:01:59</td>
      <td>2015-02-13 00:58:22</td>
      <td>2956</td>
      <td>fast</td>
      <td>4.0</td>
      <td>6</td>
      <td>3010</td>
      <td>4</td>
      <td>405</td>
      <td>825</td>
      <td>12.0</td>
      <td>11.0</td>
      <td>14.0</td>
      <td>251</td>
      <td>915.0</td>
    </tr>
    <tr>
      <td>197425</td>
      <td>1.0</td>
      <td>2015-01-24 04:46:08</td>
      <td>2015-01-24 05:36:16</td>
      <td>2956</td>
      <td>fast</td>
      <td>4.0</td>
      <td>5</td>
      <td>1836</td>
      <td>3</td>
      <td>300</td>
      <td>399</td>
      <td>39.0</td>
      <td>41.0</td>
      <td>40.0</td>
      <td>251</td>
      <td>795.0</td>
    </tr>
    <tr>
      <td>197426</td>
      <td>1.0</td>
      <td>2015-02-01 18:18:15</td>
      <td>2015-02-01 19:23:22</td>
      <td>3630</td>
      <td>sandwich</td>
      <td>1.0</td>
      <td>1</td>
      <td>1175</td>
      <td>1</td>
      <td>535</td>
      <td>535</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>12.0</td>
      <td>446</td>
      <td>384.0</td>
    </tr>
    <tr>
      <td>197427</td>
      <td>1.0</td>
      <td>2015-02-08 19:24:33</td>
      <td>2015-02-08 20:01:41</td>
      <td>3630</td>
      <td>sandwich</td>
      <td>1.0</td>
      <td>4</td>
      <td>2605</td>
      <td>4</td>
      <td>425</td>
      <td>750</td>
      <td>20.0</td>
      <td>20.0</td>
      <td>23.0</td>
      <td>446</td>
      <td>134.0</td>
    </tr>
  </tbody>
</table>
<p>197428 rows × 16 columns</p>
</div>



```python
# Analyzing summary statistics of variables
df.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>market_id</td>
      <td>196441.0</td>
      <td>2.978706</td>
      <td>1.524867</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <td>store_id</td>
      <td>197428.0</td>
      <td>3530.510272</td>
      <td>2053.496711</td>
      <td>1.0</td>
      <td>1686.0</td>
      <td>3592.0</td>
      <td>5299.0</td>
      <td>6987.0</td>
    </tr>
    <tr>
      <td>order_protocol</td>
      <td>196433.0</td>
      <td>2.882352</td>
      <td>1.503771</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <td>total_items</td>
      <td>197428.0</td>
      <td>3.196391</td>
      <td>2.666546</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>411.0</td>
    </tr>
    <tr>
      <td>subtotal</td>
      <td>197428.0</td>
      <td>2682.331402</td>
      <td>1823.093688</td>
      <td>0.0</td>
      <td>1400.0</td>
      <td>2200.0</td>
      <td>3395.0</td>
      <td>27100.0</td>
    </tr>
    <tr>
      <td>num_distinct_items</td>
      <td>197428.0</td>
      <td>2.670791</td>
      <td>1.630255</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <td>min_item_price</td>
      <td>197428.0</td>
      <td>686.218470</td>
      <td>522.038648</td>
      <td>-86.0</td>
      <td>299.0</td>
      <td>595.0</td>
      <td>949.0</td>
      <td>14700.0</td>
    </tr>
    <tr>
      <td>max_item_price</td>
      <td>197428.0</td>
      <td>1159.588630</td>
      <td>558.411377</td>
      <td>0.0</td>
      <td>800.0</td>
      <td>1095.0</td>
      <td>1395.0</td>
      <td>14700.0</td>
    </tr>
    <tr>
      <td>total_onshift_dashers</td>
      <td>181166.0</td>
      <td>44.808093</td>
      <td>34.526783</td>
      <td>-4.0</td>
      <td>17.0</td>
      <td>37.0</td>
      <td>65.0</td>
      <td>171.0</td>
    </tr>
    <tr>
      <td>total_busy_dashers</td>
      <td>181166.0</td>
      <td>41.739747</td>
      <td>32.145733</td>
      <td>-5.0</td>
      <td>15.0</td>
      <td>34.0</td>
      <td>62.0</td>
      <td>154.0</td>
    </tr>
    <tr>
      <td>total_outstanding_orders</td>
      <td>181166.0</td>
      <td>58.050065</td>
      <td>52.661830</td>
      <td>-6.0</td>
      <td>17.0</td>
      <td>41.0</td>
      <td>85.0</td>
      <td>285.0</td>
    </tr>
    <tr>
      <td>estimated_order_place_duration</td>
      <td>197428.0</td>
      <td>308.560179</td>
      <td>90.139653</td>
      <td>0.0</td>
      <td>251.0</td>
      <td>251.0</td>
      <td>446.0</td>
      <td>2715.0</td>
    </tr>
    <tr>
      <td>estimated_store_to_consumer_driving_duration</td>
      <td>196902.0</td>
      <td>545.358935</td>
      <td>219.352902</td>
      <td>0.0</td>
      <td>382.0</td>
      <td>544.0</td>
      <td>702.0</td>
      <td>2088.0</td>
    </tr>
  </tbody>
</table>
</div>



From the above analysis, we can see that like 'min_item_price', 'total_onshift_dashers', 'total_busy_dashers' and 'total_outstanding_orders' have negative values which we need to address. Looking at the quantile, we can observe that the max value of few columns (eg total_items, subtotal) are extreme compared to the 75%.


```python
# Eliminating negative values from the above mentioned columns
def process_neg(df):
    df = df[~(df['min_item_price'] < 0)]
    df = df[~(df['total_onshift_dashers'] < 0)]
    df = df[~(df['total_busy_dashers'] < 0)]
    df = df[~(df['total_outstanding_orders'] < 0)]
    return df
df = process_neg(df)
```


```python
#Column wise count of null values
df.isnull().sum()
```




    market_id                                         987
    created_at                                          0
    actual_delivery_time                                7
    store_id                                            0
    store_primary_category                           4758
    order_protocol                                    995
    total_items                                         0
    subtotal                                            0
    num_distinct_items                                  0
    min_item_price                                      0
    max_item_price                                      0
    total_onshift_dashers                           16261
    total_busy_dashers                              16261
    total_outstanding_orders                        16261
    estimated_order_place_duration                      0
    estimated_store_to_consumer_driving_duration      525
    dtype: int64



The above analysis tells us that columns like 'market_id', 'actual_delivery_time', 'order_protocol', 'total_onshift_dashers', 'total_busy_dashers', 'total_outstanding_orders' and 'estimated_store_to_consumer_driving_duration' consists null values which we will address later. 

### Creating target variable

The aim is to predict the total delivery duration for an order, we need to create the target variable. This target variable will be the total time taken after submiting the order by the customer to the order being delivered.\
   **Total delivery duration = Time the order delivered to the customer - Time customer submits the order**


```python
def create_features(df):    
    # Converting time to date-time object
    df['created_at'] = df['created_at'].apply(pd.to_datetime)
    df['actual_delivery_time'] = df['actual_delivery_time'].apply(pd.to_datetime)

    # Converting timestamp UTC to PST 
    df['created_at'] = df['created_at'].dt.tz_localize('utc').dt.tz_convert('US/Pacific')
    df['actual_delivery_time'] = df['actual_delivery_time'].dt.tz_localize('utc').dt.tz_convert('US/Pacific')

    # Creating the target variable 
    df['delivery_time'] = df['actual_delivery_time'] - df['created_at']
    df['delivery_time'] = df['delivery_time'].dt.total_seconds()

    # Parse created_at variable to obtain more granular details like date, time, hour and day of the week for further analysis
    df['created_at_date'] = df['created_at'].dt.date
    df['created_at_time'] = df['created_at'].dt.time
    df['created_at_hour'] = df['created_at'].dt.hour
    df['created_at_day'] = df['created_at'].dt.dayofweek
    return df

df = create_features(df)

# Glimse of new columns added
df[['created_at', 'actual_delivery_time', 'delivery_time', 'created_at_date', 'created_at_time', 'created_at_hour', 'created_at_day']].head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>created_at</th>
      <th>actual_delivery_time</th>
      <th>delivery_time</th>
      <th>created_at_date</th>
      <th>created_at_time</th>
      <th>created_at_hour</th>
      <th>created_at_day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2015-02-06 14:24:17-08:00</td>
      <td>2015-02-06 15:27:16-08:00</td>
      <td>3779.0</td>
      <td>2015-02-06</td>
      <td>14:24:17</td>
      <td>14</td>
      <td>4</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2015-02-10 13:49:25-08:00</td>
      <td>2015-02-10 14:56:29-08:00</td>
      <td>4024.0</td>
      <td>2015-02-10</td>
      <td>13:49:25</td>
      <td>13</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2015-01-22 12:39:28-08:00</td>
      <td>2015-01-22 13:09:09-08:00</td>
      <td>1781.0</td>
      <td>2015-01-22</td>
      <td>12:39:28</td>
      <td>12</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("The dateset contains order information from {} to {}".format(df['created_at_date'].min(), df['created_at_date'].max()))
```

    The dateset contains order information from 2014-10-18 to 2015-02-17


### EDA
Using colorblind friendly palette


```python
# Analyzing total orders for different market Ids
market_id_df = df[['market_id', 'delivery_time']]
market_id_df = market_id_df.groupby('market_id').size().reset_index(name = 'total_orders')
market_id_df = market_id_df.set_index('market_id')

# plotting 
plt.figure(figsize=(12,4))
market_id_df['total_orders'].plot(kind='bar',alpha=0.75)
plt.title('Total orders per Market Id')
plt.ylabel('Count')
plt.xlabel('Market Id')
plt.xticks(rotation='horizontal')
plt.show()
```


    
![png](output_16_0.png)
    


From the above graph, we can see that **Market Id 2** and **Market Id 4** have the highest number of orders. 


```python
# Analyzing average delivery time taken for different market Ids
market_id_df = df[['market_id', 'delivery_time']]
market_id_df = market_id_df.groupby('market_id')['delivery_time'].mean().reset_index()
market_id_df = market_id_df.set_index('market_id')

# plotting 
plt.figure(figsize=(12,4))
market_id_df['delivery_time'].plot(kind='bar',alpha=0.75)
plt.title('Average delivery time per Market Id')
plt.xticks(rotation=90)
plt.ylabel('Average delivery time')
plt.xlabel('Market Id')
plt.xticks(rotation='horizontal')
plt.show()
```


    
![png](output_18_0.png)
    


From the above graph, we can see that **Market Id 1** has the highest average delivery time.


```python
# Dividing the time according to meals
def bins(x):
    if x>10 and x<=16:
        return 'Lunch'
    if x>16 and x<=23:
        return 'Dinner'
    else:
        return 'Breakfast'
df['meal'] = df['created_at_hour'].apply(lambda x: bins(x))
```


```python
# Analyzing total orders by Meals (breakfast, lunch, dinner) for different market Ids
market_id_df = df[['market_id', 'delivery_time', 'meal']]
market_id_df = market_id_df.groupby(['market_id', 'meal']).size().reset_index(name = 'total_orders')

breakfast_time = market_id_df[market_id_df['meal'] == 'Breakfast']['total_orders']
lunch_time = market_id_df[market_id_df['meal'] == 'Lunch']['total_orders']
dinner_time = market_id_df[market_id_df['meal'] == 'Dinner']['total_orders']

X = np.arange(len(breakfast_time))
plt.figure(figsize=(18,6))
plt.bar(X, breakfast_time, color = '#bae4bc', width = 0.25)
plt.bar(X + 0.25, lunch_time, color = '#7bccc4', width = 0.25)
plt.bar(X + 0.5, dinner_time, color = '#0868ac', width = 0.25)
plt.legend(['Breakfast', 'Lunch', 'Dinner'])
plt.xticks([i + 0.25 for i in range(6)], ['1', '2', '3', '4', '5', '6'])
plt.title('Bar plot representing total orders by meals market id wise')
plt.xlabel('Market Id')
plt.ylabel('Count')
plt.show()
```


    
![png](output_21_0.png)
    


People tend to order more dinner meals than breakfast or lunch. **Market id 2** and **market id 4** have the highest number of orders during dinner time. **Market id 5** and **market id 6** have the lowest number of breakfast orders. Overall **market id 3, 5 and 6** have a smaller number of orders from while **market id 1,2 and 4** have the most number of orders.


```python
# Analyzing average delivery time by Meals (breakfast, lunch, dinner) for different market Ids
market_id_df = df[['market_id', 'delivery_time', 'meal']]
market_id_df = market_id_df.groupby(['market_id', 'meal'])['delivery_time'].mean().reset_index()

breakfast_time = market_id_df[market_id_df['meal'] == 'Breakfast']['delivery_time']
lunch_time = market_id_df[market_id_df['meal'] == 'Lunch']['delivery_time']
dinner_time = market_id_df[market_id_df['meal'] == 'Dinner']['delivery_time']

X = np.arange(len(breakfast_time))
plt.figure(figsize=(18,6))
plt.bar(X, breakfast_time, color = '#bae4bc', width = 0.25)
plt.bar(X + 0.25, lunch_time, color = '#7bccc4', width = 0.25)
plt.bar(X + 0.5, dinner_time, color = '#0868ac', width = 0.25)
plt.legend(['Breakfast', 'Lunch', 'Dinner'])
plt.xticks([i + 0.25 for i in range(6)], ['1', '2', '3', '4', '5', '6'])
plt.title("Bar plot representing average delivery time by meals market id wise")
plt.xlabel('Market Id')
plt.ylabel('Average delivery time')
plt.show()
```


    
![png](output_23_0.png)
    


From the above graph, we can see that **market id 1** has the highest average delivery time for all kind of meals time. Delivery location being far away from store location or traffic can be possible reasons for market id 1 to have longer delivery time. Looking at market id 3, we can say that breakfast orders take longer than other orders. Morning traffic can be a possibility for long delivery times. 


```python
# Analyzing relationship between average delivery time and total orders
total_items_df = df[['total_items', 'delivery_time']]
total_items_df = total_items_df.groupby('total_items')['delivery_time'].mean().reset_index()
total_items_df = total_items_df.set_index('total_items')

# plotting 
plt.figure(figsize=(12,6))
total_items_df['delivery_time'].plot(kind='bar',alpha=0.75)
plt.title('Average delivery time vs total items per order')
plt.ylabel('Average delivery time')
plt.xlabel('Total items per order')
plt.show()
```


    
![png](output_25_0.png)
    


We cannot determine accurate relationship between delivery time and total items per order from the above graph. We can see an upward trend in the initial stage. Large item orders have less average delivery time; same type of item for entire order requires less prep time than different items for the same order. Overall average delivery time for large items per order are more than time taken for fewer items per order except a handful of orders. We need more data for larger items per order for accurately determine their relationship. 


```python
hour_df = df[['market_id', 'delivery_time', 'created_at_hour']]
hour_df = hour_df.groupby(['market_id', 'created_at_hour'])['delivery_time'].mean().reset_index()

# Plotting line graphs of variables
plt.figure(figsize=(15,10))

plt.subplot(2,3,1)
sns.lineplot(data= hour_df[hour_df['market_id'] == 1.0], x='created_at_hour', y = 'delivery_time')
plt.title('Market Id 1')

plt.subplot(2,3,2)
sns.lineplot(data= hour_df[hour_df['market_id'] == 2.0], x='created_at_hour', y = 'delivery_time')
plt.title('Market Id 2')

plt.subplot(2,3,3)
sns.lineplot(data= hour_df[hour_df['market_id'] == 3.0], x='created_at_hour', y = 'delivery_time')
plt.title('Market Id 3')

plt.subplot(2,3,4)
sns.lineplot(data= hour_df[hour_df['market_id'] == 4.0], x='created_at_hour', y = 'delivery_time')
plt.title('Market Id 4')

plt.subplot(2,3,5)
sns.lineplot(data= hour_df[hour_df['market_id'] == 5.0], x='created_at_hour', y = 'delivery_time')
plt.title('Market Id 5')

plt.subplot(2,3,6)
sns.lineplot(data= hour_df[hour_df['market_id'] == 6.0], x='created_at_hour', y = 'delivery_time')
plt.title('Market Id 6')

plt.tight_layout()
plt.show()
```


    
![png](output_27_0.png)
    


Hour of the day has great impact on the delivery time. Time required to deliver morning orders takes much more time in **market id 2**, **market id 3** and **market id 4**. **Market id 1** and **market id 5** have longer delivery time during evening meals. 


```python
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
sns.boxplot(df['min_item_price'])
plt.xticks(rotation=45)

plt.subplot(1,2,2)
sns.boxplot(df['max_item_price'])
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```


    
![png](output_29_0.png)
    


We can see that there are some outliers in each variable which we will address below.


```python
def treat_null(df):
    df['store_primary_category'] = df['store_primary_category'].replace(np.nan, 'other')
    #df['market_id'] = df['market_id'].replace(np.nan, np.float64(-1))
    df['order_protocol'] = df['order_protocol'].replace(np.nan, np.float64(-1))

    median = SimpleImputer(missing_values=np.nan, strategy='median')
    columns = ['total_onshift_dashers', 'total_busy_dashers', 'total_outstanding_orders', 'estimated_store_to_consumer_driving_duration']
    median.fit(df[columns])
    df[columns] = median.transform(df[columns])
    return df

df = treat_null(df)

# Since the target variable will be difference of created_at and actual_delivery_time, we should make sure that they do not contain null values
df = df.dropna(subset = ['actual_delivery_time'])
df = df.dropna(subset = ['market_id'])
```

Null value treatment
- Discarding rows which have null values in actual delivery time variable (7 values)
- Replacing null values in store primary category with 'other'
- Replacing null values in market id and order protocol with -1 type float64
- Imputing median values for the columns total_onshift_dashers, total_busy_dashers, total_outstanding_orders and estimated_store_to_consumer_driving_duration


```python
# Plotting histograms of variables
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
sns.distplot(df['total_items'], kde=False, bins=400)
plt.xticks(rotation=45)
plt.xlim([0,85])

plt.subplot(2,2,2)
sns.distplot(df['subtotal'], kde=False)
plt.xticks(rotation=45)

plt.subplot(2,2,3)
sns.distplot(df['total_busy_dashers'], kde=False)
plt.xticks(rotation=45)

plt.subplot(2,2,4)
sns.distplot(df['total_outstanding_orders'], kde=False)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```


    
![png](output_33_0.png)
    


From the above graphs, we can infer following:
- The variable total items has exponential distribuiton. Majority orders have total_items less than 5. 
- The variable sub_total has a skewed distribution. Majority orders are below 3000 cents (30 dollars). There are few orders whos subtotal is more than 5000 cents (50 dollars) seen in the tail of the distribution.
- The variable total_busy_dashers and total_outstanding_orders follows downward linesr trens with few exceptions. At a given point of time. We will need more data in order to analyze these variables in detail.


```python
# removing outliers manually
def remove_outliers(df):
    df = df[df['max_item_price'] <= 10000]
    df = df[df['min_item_price'] <= 10000]
    df = df[df['total_items'] <= 100]
    df = df[df['subtotal'] <= 20000]
    return df

df = remove_outliers(df)
df = df[df['delivery_time'] <= 35000]
```


```python
# Dropping redundant columns
#extra = ['actual_delivery_time', 'created_at', 'created_at_date', 'created_at_time', 'created_at_hour', 'created_at_day', 'store_primary_category']
extra = ['actual_delivery_time', 'created_at', 'created_at_date', 'created_at_time', 'store_primary_category']

df.drop(extra, axis=1, inplace=True)
```


```python
# Standardize continuous variables by removing the mean and scaling to unit variance
def standardize(df, cols):
    ss = StandardScaler()
    
    for col in cols:
        df[col] = ss.fit_transform(df[col].values.reshape(-1, 1))
    return df

continuous =  ['estimated_order_place_duration', 'estimated_store_to_consumer_driving_duration']
df = standardize(df, continuous)
```


```python
# One hot encoding of categorical variables
def one_hot(df, cols):
    
    for col in cols:
        one_hot = pd.get_dummies(df[col], prefix=col)
        df = df.drop(col,axis = 1)
        df = pd.concat([df, one_hot], axis=1)
    return df

categorical = ['meal']
df = one_hot(df, categorical)
```


```python
X = df.loc[:, df.columns != 'delivery_time']
y = df.loc[:, df.columns == 'delivery_time']
```

**Feature selection** \
Select from Model is a technique which uses a machine learning algorithm (usually a random forest) of choice to select the features that are most important to the said model. This technique is computationally expensive for a large dataset with multiple features.
For feature selection, I used SelectFromModel from sklearn. The target variable here is continous. The feature variables are a combination of categorical and continuous.
From a statistical purpose, we need to calculate Pearson Correlation Coefficient to accurately determine relationship between features and target variable when they have a linear relationship.


```python
# Feature selection using SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

sel = SelectFromModel(RandomForestRegressor(n_estimators = 100))
sel.fit(X, y.to_numpy().ravel())
sel.get_support()

selected_features= X.columns[(sel.get_support())]
print('Number of selected features are: ',len(selected_features))
print("Selected features are as follows: ")
print(selected_features)

# Taking selected features
X = X[selected_features]
```

    Number of selected features are:  8
    Selected features are as follows: 
    Index(['store_id', 'subtotal', 'min_item_price', 'max_item_price',
           'total_onshift_dashers', 'total_outstanding_orders',
           'estimated_store_to_consumer_driving_duration', 'created_at_hour'],
          dtype='object')


**Train test split and cross validation** \
Ideally, we don't need another test set since Cross Validation will take care to test the model on unseen data. I have choosen 5 splits for cross validation. It means that at each time, my model would take 5 chunks of data sets, wherein it will use 4 sets for training and the remaining set to test. This will be carried out 5 times. This will also prevent our model to overfit from training data. 


```python
# Splitting our data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)
```


```python
# Traning the model with CV
def train_cv( X_train, y_train, model_name, model, cv=5):
    kf = KFold(n_splits=cv,random_state=42, shuffle=True)
    RMSE_list = []
    MAE_list = []
    i=0
    print(model_name)
    print("Performing training with cross validation for ", model_name)
    for train_index, test_index in kf.split(X_train):
        X_cv_train, X_cv_test = X_train.iloc[train_index], X_train.iloc[test_index]
        y_cv_train, y_cv_test = y_train.iloc[train_index], y_train.iloc[test_index]
        y_cv_train = y_cv_train.to_numpy().ravel()
        y_cv_test = y_cv_test.to_numpy().ravel()
        model = model.fit(X_cv_train,y_cv_train)
        y_cv_pred = model.predict(X_cv_test)
        RMSE = sqrt(mean_squared_error(y_cv_test, y_cv_pred))
        MAE = mean_absolute_error(y_cv_test, y_cv_pred)
        print("Cross validation:{} | MAE: {:.4f}, RMSE: {:.4f}".format(i+1, MAE, RMSE))
        RMSE_list.append(RMSE)
        MAE_list.append(MAE)
        i+=1
    print('Mean CV MAE :', (sum(MAE_list)/cv))
    print('Mean CV RMSE :', (sum(RMSE_list)/cv))
    return model

# Predicting on test set
def predict(model_name, model, X_test, y_test):
    y_test = y_test.to_numpy().ravel()
    print("Predicting on test set for ", model_name)
    y_pred = model.predict(X_test)
    print("RMSE test:",sqrt(mean_squared_error(y_test, y_pred)))
    print('MAE test:' ,mean_absolute_error(y_test, y_pred))
    return
```


```python
# Initializing Random Forest Regressor
model_name = 'Random Forest Regressor'
rfr = RandomForestRegressor(n_estimators=100, random_state=42,  n_jobs = -1)
model_rfr = train_cv(X_train, y_train, model_name, rfr, cv=5)
predict(model_name, model_rfr, X_test, y_test)
```

    Random Forest Regressor
    Performing training with cross validation for  Random Forest Regressor
    Cross validation:1 | MAE: 699.1140, RMSE: 996.2278
    Cross validation:2 | MAE: 693.3670, RMSE: 975.7811
    Cross validation:3 | MAE: 701.6954, RMSE: 984.9967
    Cross validation:4 | MAE: 689.9597, RMSE: 960.6535
    Cross validation:5 | MAE: 698.3133, RMSE: 1029.0484
    Mean CV MAE : 696.4898852643189
    Mean CV RMSE : 989.3414944254131
    Predicting on test set for  Random Forest Regressor
    RMSE test: 975.2965624558371
    MAE test: 696.2387136162671



```python
# Feature importance score for Random Forest
sorted_idx = model_rfr.feature_importances_.argsort()
names = [X.columns[i] for i in sorted_idx]
plt.barh(names, model_rfr.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")
plt.show()
```


    
![png](output_46_0.png)
    



```python
# Initializing XGBoost Regressor
model_name = 'XGB Regressor'
xgbr = XGBRegressor(random_state=40,n_jobs=-1)
model_xgbr = train_cv(X_train, y_train, model_name, xgbr, cv=5)
predict(model_name, model_xgbr, X_test, y_test)
```

    XGB Regressor
    Performing training with cross validation for  XGB Regressor
    Cross validation:1 | MAE: 666.4369, RMSE: 966.0612
    Cross validation:2 | MAE: 660.2455, RMSE: 948.7416
    Cross validation:3 | MAE: 663.8988, RMSE: 947.4235
    Cross validation:4 | MAE: 655.6210, RMSE: 926.6337
    Cross validation:5 | MAE: 667.1950, RMSE: 1003.2846
    Mean CV MAE : 662.679432737064
    Mean CV RMSE : 958.4289242427556
    Predicting on test set for  XGB Regressor
    RMSE test: 946.5646074613579
    MAE test: 663.7489426197952



```python
# Feature importance score for XG Boost
sorted_idx = model_xgbr.feature_importances_.argsort()
names = [X.columns[i] for i in sorted_idx]
plt.barh(names, model_xgbr.feature_importances_[sorted_idx])
plt.xlabel("XGBoost Feature Importance")
plt.show()
```


    
![png](output_48_0.png)
    


By comparing results of both the models, XGBoost Regressor performs better than Random Forest Regressor. Therefore, we will go ahead with XGBoost for our predictions.

## Predict on test data


```python
# Read the test data
predict_df = pd.read_csv('predict_data.csv')

# Making a raw copy
predict_df_raw = predict_df.copy()

# Structure of test data
print("The test dataset has {} records with {} fields.".format(*predict_df.shape))
```

    The test dataset has 54778 records with 17 fields.



```python
# Columns in dataset
predict_df.columns
```




    Index(['market_id', 'created_at', 'store_id', 'store_primary_category',
           'order_protocol', 'total_items', 'subtotal', 'num_distinct_items',
           'min_item_price', 'max_item_price', 'total_onshift_dashers',
           'total_busy_dashers', 'total_outstanding_orders',
           'estimated_order_place_duration',
           'estimated_store_to_consumer_driving_duration', 'delivery_id',
           'platform'],
          dtype='object')




```python
def create_features_test(df):
    df['created_at'] = df['created_at'].apply(pd.to_datetime)
    
    # Parse created_at variable to obtain more granular details like date, time, hour and day of the week for further analysis
    df['created_at_date'] = df['created_at'].dt.date
    df['created_at_time'] = df['created_at'].dt.time
    df['created_at_hour'] = df['created_at'].dt.hour
    df['created_at_day'] = df['created_at'].dt.dayofweek
    return df
```


```python
# Function for treating null values
def treat_null(df):
    df['store_primary_category'] = df['store_primary_category'].replace(np.nan, 'other')
    df['market_id'] = df['market_id'].replace(np.nan, np.float64(-1))
    df['order_protocol'] = df['order_protocol'].replace(np.nan, np.float64(-1))

    median = SimpleImputer(missing_values=np.nan, strategy='median')
    columns = ['total_onshift_dashers', 'total_busy_dashers', 'total_outstanding_orders', 'estimated_store_to_consumer_driving_duration']
    median.fit(df[columns])
    df[columns] = median.transform(df[columns])
    return df
```


```python
# Eliminating negative columns from columns
predict_df = process_neg(predict_df)
predict_df_raw = process_neg(predict_df_raw)

# Adding new features
predict_df = create_features_test(predict_df)
#predict_df_raw = create_features_test(predict_df_raw)

# Remove outliers
predict_df = remove_outliers(predict_df)
predict_df_raw = remove_outliers(predict_df_raw)

# Treating null values
predict_df = treat_null(predict_df)

# Dropping redundant columns
extra = ['created_at', 'created_at_date', 'created_at_time', 'store_primary_category', 'platform']
predict_df.drop(extra, axis=1, inplace=True)

# Standardize continuous variables by removing the mean and scaling to unit variance
continuous =  ['estimated_order_place_duration', 'estimated_store_to_consumer_driving_duration']
predict_df = standardize(predict_df, continuous)

# One hot encoding of categorical variable
predict_df['meal'] = predict_df['created_at_hour'].apply(lambda x: bins(x))
categorical = ['meal']
predict_df = one_hot(predict_df, categorical)
```


```python
# Taking same features as the training set
predict_df = predict_df[selected_features]

# Generating predictions for XG Boost
#pred_rfr = model_rfr.predict(predict_df)
pred_xgbr = model_xgbr.predict(predict_df)

# Adding predicted delivery time in raw dataframe
predict_df_raw['predicted_duration'] = pred_xgbr

# Choosing only delivery_id and predicted_duration for results
predict_df_raw = predict_df_raw[['delivery_id', 'predicted_duration']]

# Checking the dataset
predict_df_raw.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>delivery_id</th>
      <th>predicted_duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>194096</td>
      <td>3203.282471</td>
    </tr>
    <tr>
      <td>1</td>
      <td>236895</td>
      <td>3177.063477</td>
    </tr>
    <tr>
      <td>2</td>
      <td>190868</td>
      <td>2689.479980</td>
    </tr>
    <tr>
      <td>3</td>
      <td>183076</td>
      <td>3076.231445</td>
    </tr>
    <tr>
      <td>4</td>
      <td>186200</td>
      <td>3760.927490</td>
    </tr>
  </tbody>
</table>
</div>



We have predicted_duration (in seconds) for each delivery id.


```python
# Converting into CSV
predict_df_raw.to_csv('predicted_data_madhura.csv')
```
