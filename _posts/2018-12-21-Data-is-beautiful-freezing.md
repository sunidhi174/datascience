---
title: Freezing Cycle of Lake Mendota
tags: [Visualization]
header:
excerpt: "A visualization of freezing cycle of Lake Mendota"
---
---
## BattleViz December 2018
[r/dataisbeautiful](https://old.reddit.com/r/dataisbeautiful/comments/a2p5f0/battle_dataviz_battle_for_the_month_of_december/?st=jr88xxut&sh=39c37bae)


```python
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```

## Data preparation:
---

For small datasets I prefer to work with MS-Excel as it gives a visual cue. I've taken the following steps to prepare the [data]({{ site.baseurl }}/datasets/freezing_2.csv):

1. Remove closing and opening dates
2. Winter year was in the format "1885-86", I considered it to be the winter of 1885
3. Since we still are in the winter of 2018 I've excluded that row from the analysis
4. "DAYS" columns contains the total number of days the Lake was frozen in the season
5. I've calculated a standard moving averages for 5 and 10 years
6. The Weighted moving average is calculated as follows: $5*year+4*(year-1)+3*(year-2)+2*(year-3)+(year-4)/15$


```python
data = pd.read_csv("./freezing_2.csv")
```


```python
data = data.iloc[:,0:7]
data.head()
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
      <th>WINTER</th>
      <th>DAYS</th>
      <th>Moving_Average_5</th>
      <th>Moving_Average_10</th>
      <th>weighted_moving_average_5</th>
      <th>Ratio</th>
      <th>Trend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1855</td>
      <td>118</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1856</td>
      <td>151</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1857</td>
      <td>121</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1858</td>
      <td>96</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1859</td>
      <td>110</td>
      <td>119.2</td>
      <td>NaN</td>
      <td>114.466667</td>
      <td>1.040606</td>
      <td>116.824659</td>
    </tr>
  </tbody>
</table>
</div>



## Visualizing the data:
---

I've used standard python library matplotlib but the same visualization can be replicated using any available tool.


```python
fig, ax = plt.subplots(figsize = (20,18), nrows=4)
i = 0
ys = ['DAYS', 'Moving_Average_10', 'weighted_moving_average_5','Trend']
for row in ax:
    row.plot(data.WINTER, data[ys[i]])
    row.set_title(ys[i])
    i += 1
plt.show()
```


![png]({{ site.baseurl }}/images/output_6_0.png)


## Insights
---
In the plots above i've tried to visualize the total duration of freezing period in each year. Since the year over year fluctations are quite high I've smoothed it using 3 different smoothing techniques.

1. Moving average of last 10 years
2. Weighted Moving average of 5 years, where the latest year gets the highest weight
3. Linear smoothing of 5 years weighted moving average using regression

The apparent trend in data is duration of freezing period is decreasing at an alarming rate. In just ~150 years we've lost a month's worth of winter.

Don't believe "The President". Global Warming is real. 
