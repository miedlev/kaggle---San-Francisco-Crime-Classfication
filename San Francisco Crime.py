#!/usr/bin/env python
# coding: utf-8

# ## San Francisco Crime Classification
# 
# 
# ![sf crime](http://drive.google.com/uc?export=view&id=1WxNezzSSmypLqQfPd-bLajr09qvmjAHF)

# In[1]:


import pandas as pd
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
plt.style.use('ggplot')

pd.options.display.max_rows = 1000 
pd.options.display.max_columns = 100 

import matplotlib
matplotlib.rc("font", family = "AppleGothic")
matplotlib.rc("axes", unicode_minus = False)

from IPython.display import set_matplotlib_formats
set_matplotlib_formats("retina")


# In[2]:


train = pd.read_csv("Desktop/phthon/Kaggle/sanfransico/train.csv")
print(train.shape)
train.head(2)


# In[3]:


test = pd.read_csv("Desktop/phthon/Kaggle/sanfransico/test.csv", index_col = "Id")
print(test.shape)
test.head(2)


# ## Preprocessing

# ### 1. Find not useful columns / np.nan

# train과 test 데이터 확인결과 구해야하는 "Category" 이외에 
# 
# train의 "Descript", "Resolution" 데이터가 test에는 없다. 이는 이 두개는 사용하지 않을 가능성이 높다는 것. test에 없는 데이터로 test의 y를 맞출 수는 없기 때문이다 

# In[4]:


# np.nan은 없다 그대로 진행하면 됨

train.isnull().sum ()


# ### 2. Dates

# In[5]:


# date를 세부적으로 분석하기 위해 나눈다

train["Dates"] = pd.to_datetime(train["Dates"])

train["Dates-year"] = train["Dates"].dt.year
train["Dates-month"] = train["Dates"].dt.month
train["Dates-day"] = train["Dates"].dt.day
train["Dates-hour"] = train["Dates"].dt.hour
train["Dates-minute"] = train["Dates"].dt.minute
train["Dates-second"] = train["Dates"].dt.second

print(train.shape)
train.head(2)


# In[6]:


test["Dates"] = pd.to_datetime(test["Dates"])

test["Dates-year"] = test["Dates"].dt.year
test["Dates-month"] = test["Dates"].dt.month
test["Dates-day"] = test["Dates"].dt.day
test["Dates-hour"] = test["Dates"].dt.hour
test["Dates-minute"] = test["Dates"].dt.minute
test["Dates-second"] = test["Dates"].dt.second

print(test.shape)
test.head(2)


# ### 3. X, Y

# In[7]:


# x, y는 아마 위도, 경도일 가능성이 높고 이것이 맞는지 map을 통해 확인절차를 거친다
# 0번 row의 숫자를 대입해본다

x = 37.735051
y = -122.399588

import folium
folium.Map([x, y], zoom_start = 12, height = 300)


# In[8]:


# 이번에는 X, Y가 어떻게 분포가 되어있는지를 체크해본다

sns.scatterplot(x = "Y", y = "X", data = train)


# 샌프란시스코 모습이 나와야함에도 불구하고 다른 부분이 나왔다면 이것은 아웃라이어(outlier)가 있다는 것임. 즉 이 아웃라이어(outlier)를 제거해야 더 좋은 train이 나올 것임

# In[9]:


train_over40 = train["Y"] > 40
train_over122 = train["X"] > -122.25

# 총 67개가 outliner로서 
# 이미 충분한 데이터가 있고 이것이 잘못된 데이터라는건 확인을 했으니 제거한다
print(train[train_over40 & train_over122].shape)

train = train[~(train_over40 & train_over122)]

print(train.shape)


# In[10]:


# 다시 확인해본다

sns.scatterplot(x = "X", y = "Y", data = train)


# ## Exploratory Data Analysis(EDA)

# ### 1. Analysis of the column ["Dates"], ["DayOfWeek"]

# In[11]:


# 각각을 그래프로 시각화시켜 본다 

fig = plt.figure(figsize = [15,10])

# Dates-year
ax1 = fig.add_subplot(2,3,1)
ax1 = sns.countplot(x = "Dates-year", data = train)

# Dates-month
ax2 = fig.add_subplot(2,3,2)
ax2 = sns.countplot(x = "Dates-month", data = train)

# Dates-day
ax3 = fig.add_subplot(2,3,3)
ax3 = sns.countplot(x = "Dates-day", data = train)

# Dates-hour
ax4 = fig.add_subplot(2,3,4)
ax4 = sns.countplot(x = "Dates-hour", data = train)

# Dates-minute
ax5 = fig.add_subplot(2,3,5)
ax5 = sns.countplot(x = "Dates-minute", data = train)

# Dates-second
ax6 = fig.add_subplot(2,3,6)
ax6 = sns.countplot(x = "Dates-second", data = train)


# [다음과 같은 insight를 발견할 수 있다]
# 
# 1) "Dates-year"의 경우 주목해야 할 부분은 2015년의 그래프 막대 하락이다. 이것은 2015년 어떠한 사건으로 인해 전체 범죄의 수가 적어졌을 가능성이나 / 데이터 자료 자체가 부족한, 즉 12월 31일까지가 아니라 중간에 짤렸을 가능성 두가지 중에 하나이다. 따라서 그 부분을 반드시 체크하고 넘어가야 하고 이에 따라 feature_names에 적용을 할지 말지 결정해야 한다고 본다. 또한 test 데이터를 분석해 년도를 반드시 체크해봐야 한다고 보인다.
# 
# 2) "Dates-month"는 크게 차이가 없기때문에 일단 넘어간다 추후에 범죄를 월에 넣어봐서 특정 범죄가 특정 달에 많이 일어난다면 월 컬럼을 feature_name에 가져가야 할 가능성이 높음. 상식적으로 외부범죄의 경우 빈도수가 여름 > 겨울일 가능성이 높고 특히 바다에 둘러쌓인 샌프란시스코라면 더욱 더 그럴것임.
# 
# 3) "Dates-day"는 1일과 31일 제외하고는 큰 차이가 없다. 일단 31일이 적은 이유는 월별로 31일이 없는 달도 있기 때문이다. 따라서 이것이 중요한 키포인트가 될 것 같지는 않다. 이 day는 사실 DayOfWeek와 연관성이 있다고 보인다. 즉 이 컬럼에서 DayOfWeek가 있는건 그만큼 중요하기 때문이라고 보인다. 즉 그래서 일단은 DayOfWeek를 먼저 분석한 뒤에 체크를 해본다.
# 
# 4) "Dates-hour"은 범죄 발생 빈도에 큰 영향이 있을 것 같다.새벽에는 범죄가 덜 발생할 것이고, 오후 시간에는 범죄가 많이 발생할 것으로 보이고 그래프 낙폭의 차이가 있다
# 
# 5) "Dates-minute"의 경우 0분, 30분에 데이터가 몰려 있는 경향을 보이고 / 그 뒤에 5분 단위로 움직이고 있다. 일단 범죄자가 분단위까지 신경써가며 범죄를 할 가능성은 매우 낮고 결과가 0, 30분이라고 되어있는건 범죄시간을 분단위로 파악하기가 어려워 5~ 10분단위로 끈어서 한 것으로 보인다 따라서 이 컬럼은 "수정해서쓰던가", "활용하지 않는다"
# 
# 6) "Dates-second"는 모든 데이터가 0이다. 즉 이건 데이터가 0으로 되어있는 것이니까 활용해도 크게 의미가 없다고 보인다. 
# 
# ***RESULT***
# 
# Taking = "Dates-hour"
# 
# Holding = "Dates-month" , "Dates-year", "Dates-day", "Dates-minute" -> 단 사용할 가능성 70% 이상
# 
# Delete = "Dates-second"

# ### 1) Dates-year

# In[12]:


# 2014~2015년 차이를 조사한다

sns.countplot(data=train, x="Dates-year")


# In[13]:


# check 결과 2015년의 숫자 자체가 매우 적다 
# 하지만 이것만 봐서는 위의 그래프와 별반 다를게 없고 파악할 수가 없기 때문에 월을 봐야한다고 보인다
train["Dates-year"].value_counts()


# In[14]:


# 2015년만 빼와서 월을 체크해보았다
# 체크결과 1~5월까지는 있는데 6월이 없다. 6월부터는 업데이트가 안되었다는 이야기임
print("Months of 2015 is :" ,train[train["Dates-year"] 
                                   == 2015]["Dates-month"].unique())

# 혹시 확인차 2014년까지 데이터들중에 월이 빠진게 있나 체크해본다


# 그래프 결과 없음
# 즉 이건 2015년 중간까지만(5월) 있는 자료임으로 년도를 후에 반드시 체크해준다
train_except2015 = train[train["Dates-year"] != 2015]

train_except2015.groupby(["Dates-month", "Dates-year"])["Dates-second"].count().unstack().plot(figsize = [10,5])


# In[15]:


Category_unit = train_except2015["Category"].value_counts()
Category_unit = Category_unit[Category_unit > 10000]


train_except2015_10 = train_except2015[train_except2015["Category"]                                                        .isin(Category_unit.                                                              index)]
   
    
# area 차트를 사용한 결과 어느정도 2014년까지는 데이터가 있다는 것을 다시한번 파악해볼수 있다    

train_except2015_10.groupby(["Dates-year", "Category"])["Dates-second"].count().unstack().plot(figsize = [10,5], 
                        kind = "area", alpha=0.5,
                        stacked = True)

plt.title('Top 10 Category')
plt.ylabel('Number of Category')
plt.xlabel('Years')
plt.show()


# In[16]:


# 이번에는 categoty별로 분석을 진행해보고 
# 2015년은 제거한다 데이터 수가 절대적으로 부족하기 때문이다 

train_except2015 = train[train["Dates-year"] != 2015]

train_except2015.groupby(["Dates-year", "Category"])["Dates-second"].count().unstack().plot(figsize = [15,10])

# 변동지역 표시
plt.axvline(x = 2013, ymax = 1, linestyle='--', color = "r")
plt.axvline(x = 2011, ymax = 1, linestyle='--', color = "r")
plt.axvline(x = 2006, ymax = 1, linestyle='--', color = "r")
plt.axvline(x = 2008, ymax = 1, linestyle='--', color = "r")


# 역시 시각화하면 어느정도 느낄수는 있지만
# 만약에 가장 범죄가 많은 카테고리별로 꾸준하게 10년간 순위가 변동이 되지 않는다면 
# 굳이 year를 집어 넣을 이유는 없다고 생각한다

# 하지만 밑에 그래프를 보면 알 수 있듯이 변동이 발생하는 부분이 간혹 발생을 한다 
# 그것때문에 년도를 좀 살펴봐야함을 느낀다 


# In[17]:


category_list = train["Category"].value_counts()
year_list = train["Dates-year"].unique()



fig = plt.figure(figsize = [15, 40])

for i, value in zip(range(1, len(train["Dates-year"].unique()) + 1), year_list):
    # print(value)
    X = train[train["Dates-year"] == value]
    
    ax = fig.add_subplot(13,1,i)
    ax = sns.countplot(x = "Category", data = X, order = category_list.index)
    plt.xlabel(value)


# 데이터를 보면 가장 눈에 띄는건 종합에서 6번째 category인 VEHICLE THEFT의 경우 
# 2003 / 2005년까지는 순위가 3순위였으나 그 뒤로 감소가 되어 총합으로 6순위가 되었다. 즉 2003 / 2005년에는 이와 관련된 사건이 엄청 많았는데 그 뒤에 어떤 해결책을 내놓든 아니면 자동차 관련한 법이 생겼든 여러가지 특수한 이유로 인해 점점 감소하고 심지어 2010년대에는 다른 하위권보다도 적은 경우가 발생한 경우도 많다. 이 경우만 보더라도 이 년도수는 머신러닝에 어느정도 넣어줘어야 할 것으로 보인다. 이것을 보면 통해 2003~2005년에서 일어난 VEHICLE THEFT을 좀더 집중적으로 학습할 가능성이 있다 

# ### 2) Dates-month

# In[18]:


# 일단 이렇게만 하면 month는 그렇게 큰 차이를 느끼지 못하는것 같다
# 하지만 일단 분석 전에 상황을 보더라도 month에 따라 범죄가 달라질 것이다라는건 누구나 생각해볼 수 있는 문제이다
# 날씨의 영향에 따라 분명 큰 차이가 있을 것이기 때문이다

sns.countplot("Dates-month", data = train)


# In[19]:


train_except2015 = train[train["Dates-year"] != 2015]

train_except2015.groupby(["Dates-month", "Category"])["Dates-second"].count().unstack().plot(figsize = [15,10])


# 위 시도 결과 가장 영향력이 큰 상위 4개는 비슷한 그래프의 모습을 보여주고 있다 즉 "LARCENY/THEFT", "OTHER OFFENSES", "NON-CRIMINAL", "ASSAULT"은 전체 month를 기준으로 하는 데이터와 양의 상관관계로 움직이고 월이나, 계절 크게 상관없이 움직이는 것을 알 수 있다(계절로 굳이 나눈다면 아무래도 야외에서 벌어지는 경우가 상당수이기 때문에 봄(3,4,5)에 발생률이 높고, 겨울(12,1,2)과 여름(7,8)에 조금 줄어드는 경향을 보여주긴 한다)
# 
# 
# 다만 이후에 5번째 순위의 그래프부터 조금 다른 양상으로 움직이는 모습이 포착되어 이들을 별도로 분리해서 분석해본다

# In[20]:


# Category의 top4 선정
Category_value = train["Category"].value_counts()
Category_value_top4 = Category_value.head(4)
# top4 제외
train_except4 = train[~train["Category"].isin(Category_value_top4.index)]

# graph
train_except4.groupby(["Dates-month", "Category"])["Dates-second"].count().unstack().plot(figsize = [15, 10], legend = False)

plt.axvline(x = 4, color = "r", linestyle='--')
plt.axvline(x = 3, color = "r", linestyle='--')
plt.axvline(x = 11, color = "r", linestyle='--')


# 5번째 순위부터 조금씩 변동되는 모습을 보여준다. 보통 weekday는 day를 어느정도 대체를 할 수 있는 요소가 되나 month를 대체할 수는 없다. 만약 month의 변동성이 적거나, 별다른 차이가 없다면 모델링에 활용하지 않는 것이 좋지만 지금처럼 변동성을 만들어내는 요소가 있다면 modeling할 경우 좋은 결과를 낼 수 있다. 따라서 month를 활용하는 것이 바람직하다

# In[21]:


Category_value_top12 = Category_value.head(12)
Category_value_top12_8 = Category_value_top12.tail(8)

print(Category_value_top12_8)

fig = plt.figure(figsize = [25, 12])

for i, value in zip(range(1, 9), Category_value_top12_8.index):
    
    X = train[train["Category"] == value]

    ax = fig.add_subplot(2,4,i)
    ax = sns.countplot(x = "Dates-month", data = X, color = "orange")
    plt.xlabel(value)


# 위의 lineplot 그래프에서 Category 5 ~ 12순위 범죄(위에 DRUG/NARCOTIC ~ ROBBERY)의 변동성이 보여 별도로 분석해보았다. 일반적으로 2 ~ 5월까지 범죄 증가 / 여름 기간 범죄 감소 / 9 ~ 10월 재증가 / 겨울기간 감소가 전체 Category 범죄 수에 나타난 특징이였다. 그리고 1 ~ 4순위의 범죄들은 모두 이와 비슷한 흐름을 보였고 이들의 흐름으로 인해 전체 데이터가 그러한 모습을 띄었다고 봐도 무방하다(데이터 수가 많음으로). 하지만 그 뒤의 순위들에서는 조금의 차이가 발생했다. 이를 통해 month를 활용한다면 이들의 변동성을 파악할 수 있는 독립변수로서의 가치를 가진다고 볼 수 있다. 
# 
# 
# 각각 조금의 차이는 발생하지만, 주목해야할 것이 몇 가지 있다.
# 
# **< 위 그래프에서 주목할 점>**
# 
# 1. DRUG/NARCOTIC은 1,2월에 다른 곳보다 많이 발생하는 경향을 보인다. 아무래도 실외보다 실내에서 주로 하는 범죄라는 점과 계절 상 겨울에 많이 할 수도 있다라는 점에 주목할 필요가 있다
# 
# 2. WARRANTS 역시 1,2월달이 이상하리 만치 높다.(3,4월도 마찬가지). 보다 자세히 알아봐야겠지만 보증금 관련해서 샌프란시스코에서는 월말보다 월초에 더 많이 계약이 이루어진다는점, 그리고 이 기간내 범죄가 많이 일어난다고 봐야할 둣 하다. 2월달의 경우 사실상 달 수가 적기 때문에 그래프 절대값 상 줄어드는 것이지 날짜를 고려하면 1~5월은 비슷하다고 봐도 무방하다
# 
# 3. BURGALRY(주거 침입 강도)와 VEHICLE THEFT는 보통 8월달 이외에 최저점을 보이는 경향인 7월달에 오히려 증가하는 경향이 있다. 7 ~ 8월달 주로 집과 차량을 놔두고 여행을 가는 가족이 많아서 그럴 수 있다고 본다. 또한 구글링 결과 한가지를 발견했는데 'With the hot summer weather, vehicle owners are more likely to leave their windows rolled down — an open invitation for thieves, among other reasons.' 이를 통해 여름기간 동안 집이나 자동차에 문을 열어놔서 그럴 가능성도 배제할 수 없다
# 
# 결과적으로 이러한 부수적인 요소 때문에 month는 필요하다(다만 week나 hour처럼 modeling 자체에 영향력이 엄청나게 크지는 않을 듯 하다)

# ### 3) Dayofweek 

# 보통 주, 주말 개념과 월 개념은 겹치는 부분이 있기 때문에 한꺼번에 묶어서 진행한다
# Dayofweek를 준 그 이유가 분명 있다고 생각한다
# 
# 

# In[22]:


# povit_table

pivogt = pd.pivot_table(index = "DayOfWeek", 
                        values = "Dates-second", 
                        aggfunc = len, 
                        data = train).sort_values(by = "Dates-second", 
                                                  ascending = False)

print(pivogt.head(7))


plt.figure(figsize = [10,4])
columns = ['Monday', 'Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

sns.pointplot(x = pivogt.index, y = "Dates-second", markers=' ',
              data= pivogt, order = columns, ci = 0 , orient = True)


# 위의 그래프들을 통해 알수 있는 것은 다음과 같다
# 
# 1. 범죄가 가장 많이 일어나는 것은 금요일 / 수요일 / 토요일 / 목요일 / 화요일 / 월요일 / 일요일 순
# 2. 일요일 같은 경우는 범죄 빈도수가 다른 날에 비해 월등히 적다는점.
# 3. 월화수 까지는 올라가는데 목요일날 범죄 빈도수가 줄어드는데 무언가 이유가 있을 것 같다. 그것은 특정 범죄(가장 높은 범죄율을 가진)가 목요일에 대폭 감소하면서 자연스럽게 전체 숫자가 줄어들 수도 있고, 기타 공휴일이 많다던가 하는 특정 변수에 의해 움직일 가능성이 높다. 어쨋든 이 부분은 유념해서 봐야할 것 같음.
# 
# 
# 단순하게 week 수량만 봐서는 제대로된 정보를 얻을 수 없다는 판단하에 "Category"로 체크해보기로 한다

# #### * category check

# In[23]:


train["Dates_weekday(num)"] = train["Dates"].dt.dayofweek

train.groupby(["Dates_weekday(num)","Category"])["Dates-second"].count().unstack().plot(figsize=(20, 10))

plt.show()


# 
# 위의 시각화 결과 Category의 변수가 너무 많기 때문에 hue로 접근하기에는 한계가 있다 또한 weekday가 어느정도 범죄에 영향을 미친다는 것은 이미 위의 그래프로도 알 수가 있으나 좀더 세부적인 인사이트를 도출하기에는 무리가 있다 따라서 좀더 세부적으로 나누어서 분석을 진행해본다

# In[24]:


# 위 그래프를 보면 밑에 주어진 4개가 가장 경우의 수가 많고 다른 범죄보다 더 숫자가 많다 
# 따라서 이 들만 따로 떼내어서 분석을 진행해본다 

train["Category"].value_counts().head(4)


# In[25]:


# 체크결과 
# 범죄율이 가장 높은 4개를 시각화해서 보았을 때 전체적으로 total을 따라가는 모습을 보이지는 않는다
# 물론 가장 높은 범죄율인 LARCENY/THEFT같은 경우는 좀 비슷하게 가는 경향을 보이지만
# 나머지 3개는 그렇지 않은 모습을 보여준다


category_find = ["LARCENY/THEFT", "OTHER OFFENSES", "NON-CRIMINAL", "ASSAULT"]

# 전체 범죄수
train.groupby(["Dates_weekday(num)"]).agg({'Dates-second':'count'}).plot(figsize = [10, 4])
plt.title("[Total]")

# 4개 category 데이터 선별
train[train["Category"].isin(category_find)].groupby(["Dates_weekday(num)", "Category"])["Dates-second"].count().unstack().plot(figsize = [10, 5])
plt.title("[Category]")


# In[26]:


# 이번에는 범죄 수가 3만 이상인 카테고리를 추출해서 시각화를 해본다 
# 시각화 한담에 범죄순위가 제일 높은 4개는 제외시킨다(위에 이미 했음으로)

Category_over30000 = train["Category"].value_counts() 
Category_over30000 = Category_over30000[Category_over30000 > 30000]


train_fix = train[train["Category"].isin(Category_over30000.index)]
train_fix["Category"].unique()

# 전체 범죄수
train.groupby(["Dates_weekday(num)"]).agg({'Dates-second':'count'}).plot(figsize = [10, 4])
plt.title("[Total]")

# 4개 category 데이터 제외한 나머지
train_fix[~train_fix["Category"].isin(category_find)].groupby(["Dates_weekday(num)", "Category"])["Dates-second"].count().unstack().plot(figsize = [10, 5])
plt.title("[Category2]")


# 이번에는 DRUG/NARCOTIC / WARRANTS 관련범죄가 목요일에 줄었음을 파악할 수 있다

# 1. 목요일 건은 위에 두 범죄같이 직접적으로 목요일이 되었을 때 줄어드는 경향을 보이는 범죄도 있지만 대략적으로 전반적으로 월화수목 까지는 비슷하다가 금요일날 범죄가 급증하는 것으로 분석해도 무방하다
# 2. 즉 목요일이 줄어들었다기 보다는 범죄들 중 특정 몇가지(DRUG/NARCOTIC, WARRANTS,SUSPICIOUS OCC)가 수요일(2)의 범죄가 높고 오히려 목, 금이 갈수록 떨어지는 경향을 보인다. 전체적인 그래프에서 수요일이 목요일 보다 높은 것은 이들이 영향이 컸다는 것을 의미한다.
# 3. 이들은 특정 지역에 있을 가능성도 있다. 즉 그렇기에 이 weekday 컬럼은 필요하며 나중에 지역 등을 분석할때 함께 활용해야할 지표임

# #### * 전 범죄 측정

# In[27]:


category_unique = train["Category"].value_counts().index
order = [0, 1, 2, 3, 4, 5, 6]

fig = plt.figure(figsize = [30,40])

for i, category in zip(range(1, len(category_unique)+1), 
                       category_unique):
    
    X = train[train["Category"] == category]
    ax = fig.add_subplot(8, 5, i)
    ax = sns.countplot(x = "Dates_weekday(num)", data = X, order = order)
    
    plt.xlabel(category)


# **대체적으로 월화수목 보단 금토 범죄가 더 많지만 다른 경우도 존재한다**
# 
# -> 특정 범죄가 수요일에 오버슈팅하는 경향을 보이는데 이것은 분명 지역 혹은 다른 부분과 관련이 있다고 봐야한다. 또한 대체적으로 월화수목은 큰 변동이 없다가 금요일에 범죄수가 증가하는 카테고리가 훨씬 많다는 것도 유념해야한다. 단순하게 수요일이 목요일보다 더 높기 때문에 "수요일 범죄가 목요일보다 더 높다" 라는 일반적인 주장을 하면 안된다는 이야기임
#    
# 
# **관련 그래프를 통해 특정요일에 특정 범죄가 높은 성향을 보이는 것을 알수 있고 이를 통해 weekday컬럼이 중요한지를 판별할 수 있다**
# 
# -> weekday 컬럼은 그 자체만으로 범죄 카테고리별 성향을 알게 해줌과 동시에 다른 부가적 요소들을 분석할때도 유용하게 활용될 가능성이 높아보인다. 중요한 컬럼으로 보이고 이에 활용해야 한다고 생각한다
# -> 가령 과음(DRUNKENNESS), 음주운전(DRIVING UNDER THE INFLUENCE) 같은 경우가 주말에 많이 발생하고 마약(DRUG/NARCOTIC), 절도(BURGLARY) 등은 평일에 더 발생빈도가 높다
# 
# 이를 통해 요일(DayOfWeek)이 범죄를 판가름하는데 중요한 영향을 끼친다는 것을 알 수 있음으로 예측에 활용하도록 한다

# #### One Hot Encoding

# In[28]:


DayOfWeek_unique = train["DayOfWeek"].unique()

for value in DayOfWeek_unique:  
    # print(value)
    train[f"DayOfWeek_{value}"] = train["DayOfWeek"] == value
    
train.columns = ['Dates', 'Category', 'Descript', 'DayOfWeek', 'PdDistrict','Resolution', 'Address', 'X', 'Y', 'Dates-year', 'Dates-month',
                 'Dates-day', 'Dates-hour', 'Dates-minute', 'Dates-second', 'Dates_weekday(num)', 'DayOfWeek_Monday','DayOfWeek_Tuesday','DayOfWeek_Wednesday', 
                 'DayOfWeek_Thursday', 'DayOfWeek_Friday','DayOfWeek_Saturday','DayOfWeek_Sunday']


train.iloc[:, 16:].head()


# In[29]:


DayOfWeek_unique = test["DayOfWeek"].unique()

for value in DayOfWeek_unique:  
    # print(value)
    test[f"DayOfWeek_{value}"] = test["DayOfWeek"] == value
    
test.columns = ['Dates', 'DayOfWeek', 'PdDistrict', 'Address', 'X', 'Y', 'Dates-year', 'Dates-month', 
                'Dates-day', 'Dates-hour', 'Dates-minute','Dates-second', 'DayOfWeek_Monday','DayOfWeek_Tuesday',
                'DayOfWeek_Wednesday', 'DayOfWeek_Thursday', 'DayOfWeek_Friday','DayOfWeek_Saturday','DayOfWeek_Sunday']


test.iloc[:, 12:].head()


# ### 4)  Dates-minute

# In[30]:


fig = plt.figure(figsize = [15,5])

# train
ax = fig.add_subplot(1,2,1)
ax = sns.distplot(train["Dates-minute"], hist = False, color = "g")

# test
ax = fig.add_subplot(1,2,2)
ax = sns.distplot(test["Dates-minute"], hist = False, color = "r")


# train과 test의 dates-minute을 분석해본 결과histogram이 동일한 그래프를 보인다. 전혀 다르지 않고 동일하게 가는데 
# 
# 일단 범죄시간을 작성할때 정확하게 몇분에 일어난지 exact하게 적은게 아니라는 생각이 든다. 그 증거는 시간단위를 0, 15, 30, 45 이런식으로 순서가 되있다는 것이다. 반올림을 했든 아니면 대략적으로 맞췄든 간에
# 범죄를 15분 단위로 어느정도 적었다는 말이 된다. 이는 test에서도 지속적으로 유지되는 모습을 보인다 
# 
# 따라서 이 minute는 쓰지 않거나 / 바꿔줘야 한다. 그리고 그 중에서 점수 결과가 좋은 것으로 가는 것이 옳다
# 만약 test 컬럼의 방향이 완전 다르거나, 아예 없거나 한다면 아예 안쓰는게 좋지만 이렇게 비슷한 방향으로 움직인다면 
# 나름의 영향을 줄수도 있기에 활용하는 것이 좋다.

# In[31]:


train["Dates-minute(fix)"] = np.abs(train["Dates-minute"] - 30)
test["Dates-minute(fix)"] = np.abs(test["Dates-minute"] - 30)


# In[32]:


fig = plt.figure(figsize = [15,5])

# train
ax = fig.add_subplot(1,2,1)
ax = sns.distplot(train["Dates-minute(fix)"], hist = False, color = "g")

# test
ax = fig.add_subplot(1,2,2)
ax = sns.distplot(test["Dates-minute(fix)"], hist = False, color = "r")


# ### 2. Analysis of the column ["PdDistrict"]

# In[33]:


# pivottavle을 통해 데이터를 추출한 뒤에 plot를 이용해 시각화해본다

PdDistricts = train.pivot_table(index = "PdDistrict", 
                                fill_value = True,
                               values = "Dates-year", 
                              # margins = True, margins_name = "Total", 
                               aggfunc = len).sort_values(by = "Dates-year", 
                                                          ascending = False)

plt.figure(figsize = [8,5])
sns.countplot(y = "PdDistrict", data = train)

# 비율을 알고싶어서 따로 구해본다
# margin은 위 그래프에 영향을 주는 관계로 별도로 함
calculation = sum(PdDistricts["Dates-year"])
PdDistricts["Proportion"] = round(PdDistricts["Dates-year"] / 
                                  calculation, 2)


# 상위 3개가 44%를 차지하고 있음을 알 수 있다.
PdDistricts


# In[34]:


# 이번에는 각 지역별로 어떤 범죄가 많이 일어났는지 살펴본다
# 분명 대부분 강도같은 소범죄가 많이 일어나지만 / 특정 지역에서 중범죄가 비정상적으로 높을 수 있음을 가정하고 들어가본다
PdDistrict = pd.pivot_table(index = "PdDistrict", 
                            columns = "Category",
                            values = "Dates-year", 
                            data = train, 
                            aggfunc = len).T


# 범죄가 많이 일어난 순으로 나열한다 
Category_order_index = train["Category"].value_counts().index
PdDistrict = PdDistrict.loc[Category_order_index]


# 범죄가 많이 일어난 지역별로 나열한다
columns = ['SOUTHERN', 'MISSION', 'NORTHERN', 'BAYVIEW', 'CENTRAL',           'TENDERLOIN','INGLESIDE', 'TARAVAL', 'PARK', 'RICHMOND']
PdDistrict = PdDistrict[columns]
PdDistrict.head(10)


# In[35]:


# 위를 봐서는 한눈에 들어오지 않아서 시각화가 필요하다고 판단된다 
# 10개만 추출해서 해본다

# 여기서 내가 체크하려는 것은 다음과 같다 

# 범죄 순서 순으로 나열한 이상 
# 보통의 경우 지역도 비슷하게 순서대로 범죄율이 높을 수 있다고 생각을 한다
# 단 분명 특정지역은 다른데와 상반된 모습을 보일 가능성이 있다고 판단.
PdDistrict_top10 = PdDistrict.head(10)
PdDistrict_top10.plot(figsize = [15,8])

# 위 그래프에서 LARCENY/THEFT 때문에 밑 부분이 잘안보여 이부분만 제거하고 다시 추출해본다
PdDistrict_top10_except_LT = PdDistrict_top10.iloc[ 1:, : ]
PdDistrict_top10_except_LT.plot(figsize = [15,8])


# **< 위 그래프에서 주목할 점>**
# 
# 시각적으로 가장 눈여겨봐야 할 곳은 TENDERLOIN , INGLESIDE
# 
# TENDERLOIN
# 
# 앞서서 요일을 체크할때 수요일 기간 범죄량이 상승하는것. 그리고 그 연관성에 drug이나 warrant가 있다는 것을 파악했는데 그 원인이 TENDERLOIN에 있음을 알 수 있다.
# 이 지역은 다른곳과 다른 움직임을 보인다는 차별점이 있기에 주목해야할 것으로 본다
# 
# INGLESIDE
# 
# 이곳은 범죄율 하위권이다. 그럼에도 자동차 사고가 많다는 건 주목해야함
# 대체적으로 TENDERLOIN, SOUTHERN 제외하고는 자동차 사고는 지역에 많지만 이 지역에서 특히 1위가 됬다는 점에 주목해 후에 맵을 찾아볼 때 자동차와 관련된 무언 장소가 있는지 혹은 해변근처인지 체크해보면 좋을 듯 하다

# ####  전 범죄 측정

# In[36]:


category_unique = train["Category"].value_counts().index
order = train["PdDistrict"].value_counts().index


fig = plt.figure(figsize = [30,50])

for i, category in zip(range(1, 40), category_unique):
    
    X = train[train["Category"] == category]
    ax = fig.add_subplot(10, 4, i)
    ax = sns.countplot(x = "PdDistrict", 
                      data = X, order = order)
    plt.xlabel(category)   


# Weekday와 비교해서 각 PdDistrict 별로 어디가 더 많은 범죄가 일어나는지 그 차이점이 좀더 명확하다. 이는 이 PdDistrict라는 컬럼이 중요한 변수가 될 수 있음을 의미한다. 따라서 이를 모델링에 활용해야한다고 본다

# #### One Hot Encoding

# In[37]:


PdDistrict_code = pd.get_dummies(train["PdDistrict"], 
                                 prefix = "PdDistrict_")

train = pd.concat([train, PdDistrict_code], axis = 1)

print(train.shape)
train.head()


# In[38]:


PdDistrict_code = pd.get_dummies(test["PdDistrict"], 
                                 prefix = "PdDistrict_")

test = pd.concat([test, PdDistrict_code], axis = 1)

print(test.shape)
test.head()


# ### 3. Analysis of the column ["Address"]

# ### 1) "/" 여부 

# Address에서 첨에 봤을 때 제일 궁금했던 것들이 몇개 있었다 
# 
# 1) 가장 눈에 띄는 AV ST 여부 
#    2) "/" 있는 것과 없는 것은 무엇인가? 
# 
# 따라서 이를 조사해보기로 한다. 먼저 가장 마지막에 단어들을 좀 살펴보고 싶었다 
# train데이터를 분석하고자 다른 것을 카피해서 진행해본다

# In[39]:


train_copy = train.copy()

# Address 맨 뒤에 나오는 것들 수를 조사해본다
train_copy["Address_last"] = train_copy["Address"].map(lambda x : x.
                                                       split("/")[-1].
                                                       split(" ")[-1])
test["Address_last"] = test["Address"].map(lambda x : x.split("/")[-1].split(" ")[-1])

train_copy["Address_last"].value_counts().head()


# 이를 만들고 보니 "/" 이부분의 문제가 발생했다
# 
# "/"를 기준으로 같은 단어가 들어있으면 상관이 없는데 앞에는 "AV" 뒤에는 "ST" 이런식으로 적혀있는 것들이 매우 많이 있다 만약 마지막을 토대로 진행을 한다고 하면 
# 잘못하다가는 데이터의 왜곡이 생길수도 있다는 생각이 들었다. 그래서 이를 진행하기 전에 "/"는 대체 무엇인지? 왜있는것인지를 먼저 파악해보는게 중요하다는 생각이 들어 진행했다 
# 
# <결과>
# 
# 구글맵으로 진행결과 다음을 알 수 있었다
# 예를들어 세번째 row에 있는 "VANNESS AV", "GREENWICH ST" 검색결과 
# 근처에 있거나 연결되는 무언가가 있다는 점을 확인했다
# 몇개를 더 검색해서 비슷한 결과가 나오자 이번에는 X, Y를 검색해보았다 
# 그 결과 그 두개가 만나는 교차점이 대부분 검색이 되서 나왔다 (두 스트리트 혹은 가로수길이 겹쳐지는 십자가 지점)
# 
# 이를 통해 체크한 사실은 다음과 같다
# 정말로 이것이 진실된 자료라면 교차점에 있는 것들과 그렇지 않은 것으로 나누어서 확인해보는 것이다
# 교차로에서 차사고, 혹은 사소한 강도사건등이 일어날 수 있다는 생각에서 이다 

# In[40]:


def find_address_checking(address):
    
    if "/" in address:
        return "intersection"
    
    else:
        return "independent"

train_copy["Address_fix"] = train_copy["Address"].apply(find_address_checking)


# In[41]:


intersection = train_copy[train_copy["Address_fix"] == "intersection"]
independent = train_copy[train_copy["Address_fix"] == "independent"]


print(train_copy.shape)
print(intersection.shape)
print(independent.shape)


plt.figure(figsize = [20,7])
sns.barplot(x = "Category", y = "Dates-second", 
              data = train_copy, estimator = len, hue = "Address_fix",
              order = category_list.index)


# 확인결과 나름대로의 유의미한 결과가 나옴 -> 따라서 활용성이 높은 데이터라고 판단됨
# 
# 1) 몇개는 아예 인디펜던트쪽이 차지하고 있다는 것 
# 
# 2) 또한 가장 높은 범죄순인 것들 역시 대다수 인디펜던트가 가지고 있다는 사실을 알 수 있다
# 
# 3) 물론 데이터 수 자체가 많기도 하지만 / 오펜스의 경우 그럼에도 비등한 사실을 알수 있다

# In[42]:


# train test에 적용

train["Address_road"] = train["Address"].str.contains("/")
test["Address_road"] = test["Address"].str.contains("/")


# ### 2) Address 중복 및 사용 여부

# 분석결과 "/"에서 중복이 되는 경우가 많다 즉 "OAK ST / LAGUNA ST"와 "LAGUNA ST / OAK ST"는 실상 같은 곳이지만 작성은 다르게 되어 다른장소로 인식하게 된다. 그렇기 때문에 이것을 하나로 통일시켜주는 작업이 필요함으로 관련 작업을 진행한다

# In[43]:


# 함수사용하여 적용

def clean_address(address):
    
    if "/" not in address:
        return address
     
    address1, address2 = address.split("/")
    address1, address2 = address1.strip(), address2.strip()
    
    if address1 > address2:
        address = f"{address1} / {address2}"
    else:
        address = f"{address2} / {address1}"

    return address

train["Address_fix"] = train["Address"].apply(clean_address)
test["Address_fix"] = test["Address"].apply(clean_address)

# 중복값 제거 후 주소 수가 상당히 감소함
print(len(train["Address"].unique()), len(train["Address_fix"].unique()))
print(len(test["Address"].unique()), len(test["Address_fix"].unique()))


# In[44]:


# 주소가 1개인 것들이나 그 위지만 100을 넘지 않는 것들의 경우 크게 의미가 없기에 100개 이상인 주소만 따로 정리한다 
train["Address_fix"].value_counts()


# In[45]:


top100_address = train["Address_fix"].value_counts()
top100_address = top100_address[top100_address >= 100]

train.loc[~train["Address_fix"].isin(top100_address.index), "Address_fix"] = "Others"
test.loc[~test["Address_fix"].isin(top100_address.index), "Address_fix"] = "Others"


# get_dummies로 분리시킨다
train_address = pd.get_dummies(train["Address_fix"])
test_address = pd.get_dummies(test["Address_fix"])


# result
print(train_address.shape)
print(test_address.shape)


# ### 4. Analysis of the column ["X, Y"] - > using MAP(folium, plotly) / ggplot

# In[46]:


# PdDistrict가 각각 어디에 위치해 있는지부터 파악해본다

train["PdDistrict"].unique()


# ***seaborn으로도 구할 수 있으나 그것보다 효율성이 높은 R기반의 ggplot을 활용해본다***
# 
# ***ggplot과 geom_point를 적용하기 전에 시각화에서 중요하게 생각해야 할 부분이 한가지 있다***
# 
# ***folium을 활용한 맵의 경우 X, Y를 주의해야 했다. 즉 보이는대로 X, Y를 적용시 지도에 제대로 표시가 안된다. 이는 [위도, 경도]를 적용해야하는데 X가 경도, Y가 위도로 적용이 되었기 때문이다. 그렇기 때문에 folium에서는 위도, 경도의 올바른 적용을 위해 X, Y의 순서를 바꾼다***
# 
# ***하지만 ggplot에서는 X, Y에 그대로 적용해야 제대로된 시각화 그래프가 나오게 된다. 이는 몇개의 PdDistrict를 추출해서 folium에 적용해보면 확인할 수 있는 사실임으로 주의한다***

# In[47]:


# PdDistrict = NORTHERN location

import folium

latitude = 37.758338
longitude = -122.437030

# north에서 500개만 추출해본다
train_north500 = train[train["PdDistrict"] == "NORTHERN"].head(500)
sanfrans_map = folium.Map([latitude, longitude], zoom_start = 12, height= 400)

for lat, lng in zip(train_north500["Y"], train_north500["X"]):
    
    folium.CircleMarker([lat, lng], 
                        radius = 2, 
                        color = "blue").add_to(sanfrans_map)
    
sanfrans_map


# In[48]:


# PdDistrict = CENTRAL location

import folium

latitude = 37.758338
longitude = -122.437030

# CENTRAL에서 500개만 추출해본다
train_CENTRAL500 = train[train["PdDistrict"] == "CENTRAL"].head(500)
sanfrans_map = folium.Map([latitude, longitude], zoom_start = 12, height= 400)

for lat, lng in zip(train_CENTRAL500["Y"], train_CENTRAL500["X"]):
    
    folium.CircleMarker([lat, lng], 
                        radius = 2, 
                        color = "red").add_to(sanfrans_map)
    
sanfrans_map


# In[49]:


from plotnine import *

print(train["PdDistrict"].value_counts())

ggplot() + geom_point(train, aes(x = "X", y = "Y", group = "PdDistrict", 
                                 color = "PdDistrict")) + ggtitle("PdDistrict")


# 범죄수와 지역을 비교분석해보면 범죄수가 가장 높은 SOUTHERN, MISSION, NORTHERN, BAYVIEW,CENTRAL의 위치가 한쪽으로 쏠려있는 모습을 파악할 수 있다.(그림 상 오른쪽 상단에 몰려있음) 
# 
# 여기서 가장 주목한 도시는 TENDERLOIN와 하위권 범죄수를 가진 도시들이다
# 
# TENDERLOIN는 지역이 가장 작음에도 불구하고 범죄지역 순위가 6번째이다. 반대로 TENDERLOIN, PARK, RICHMOND(서쪽에 있는 지역)는 면적이 큼에도 불구하고 범죄의 수가 적다. 이를 통해 2가지 사실을 추론해볼 수 있다
# 
# 1) 이 자료가 완벽하게 수집된게 아니라면 이 서쪽 지역의 범죄에 어느정도 누락이 있을 수도 있다
# -> 이 경우 서쪽지역의 범죄를 더 많이 조사해서 오차를 줄이는 작업을 해야할 것이다
# 
# 2) 이 자료가 완벽하게 수집이 된거라면 동쪽이 대도시일 확률이 크다
# -> 지도를 시각화해서 보기 전에는 막연하게 CENTRAL이 중앙이라는 뜻임으로 지도상의 가운데에 위치한다는 편견이 들어갈 수도 있으나 실상은 우측 상단에 위치해있다는건 매우 중요한 사실이다. 이를 통해 샌프란시스코의 가장 큰 지역, 즉 county는 범죄수가 모여있는 위쪽임을 추론해볼 수 있다. 그리고 지도상에서 왼쪽보다는 오른쪽 부분에 더 많은 범죄가 있는 도시(사람들이 더 많이 살 수도 있고, 더 번화가일 가능성이 매우 높다)가 많다는 사실도 파악할 수 있다.

# In[50]:


# Category별로 시각화를 시켜본다 

ggplot() + geom_point(train, aes(x = "X", 
                                 y = "Y", 
                                 group = "Category", 
                                 color = 'Category')) + facet_wrap("Category")


# In[51]:


ggplot() + geom_point(train, aes(x = "X", 
                                 y = "Y", 
                                 group = "Category", 
                                 color = 'Category'))


# In[52]:


# 위로 했을때 Category를 보면 별다른 차이를 느끼지 못한다
# 그래서 map으로 체크해보고자 한다

# 1) 시간이 오래걸림으로 샘플 1000개만 체크해본다
# 2) Category를 color화 하여 분석해본다 
# 3) color를 Category 갯수만큼(39개) 만든다
# 4) 컬러는 HTML code로 넣어놓는다

color = ["#FFBF00", "#9966CC", "#007FFF", "#7FFFD4", "#CD7F32", "#FFFF00",
        "#483C32", "#D2B48C", "#A7FC00", "#E0115F", "#CC8899","#003153",
        "#CCCCFF", "#FF4500", "#808000", "#CC7722", "#E0B0FF","#BFFF00"]

color = color + ["#29AB87", "#00FFFF", "#FFD700", "#000000", "#FF7F50", 
                 "#B87333", "#50C878", "#C8A2C8", "#FFE5B4", "#FA8072"]

color = color + ["#008080","#007BA7", "#0095B6", "#800020", "#7B3F00",
                "#6F4E37", "#702963", "#B57EDC", "#FF00AF", "#0F52BA", 
                 "#FBCEB1"]

# 만든 컬러와 category를 연결해 map을 통해 각 Category당 컬러를 부여한다
Category = train["Category"].unique()
Category_color = dict(zip(Category, color))

train["Category_color"] = train["Category"].map(Category_color)
train[["Category", "Category_color"]].head()


# In[53]:


import folium

latitude = 37.758338
longitude = -122.437030

train_1000 = train.head(1000)
sanfrans_map = folium.Map([latitude, longitude], zoom_start = 12)

# color를 추가해준다
for lat, lng, label, colors in zip(train_1000["Y"],
                                   train_1000["X"], 
                                   train_1000["Category"], 
                                   train_1000["Category_color"]):
    
    folium.CircleMarker([lat, lng], popup = label, 
                        radius = 2, color = colors).add_to(sanfrans_map)

sanfrans_map


# In[54]:


# 컬러별로 보기가 어려운 관계로 다른 시각화 패키지 plotly를 활용해본다

import plotly_express as px

fig = px.scatter_mapbox(train_1000, 
                        lat = train_1000["Y"], 
                        lon = train_1000["X"], 
                        color = train_1000["Category"], 
                        hover_name = train_1000["PdDistrict"],
                        zoom = 10)

fig.update_layout(mapbox_style = "open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# 시각화하며 얻은 사실
# 
# 지도를 시각화한다고 해서 한눈에 특정지역에 특정범죄가 많이 발생하는것을 바로 파악하기는 어렵다는 사실이다. 
# 이유는 상위권에 속한 범죄들이 보통 대부분의 지역에서도 상위권인 점(특정 범죄 제외)때문이다. 
# 
# 따라서 map을 통해서는 2가지를 파악해야 한다는 사실을 파악해볼 수 있었다
# 
# 1. 샌프란시스코 도시(지역) 특성을 파악한다. 
# -> 이는 위에서 지역별 위치 및 이름, 그리고 동서쪽 간의 범죄 차이 등을 구분하면서 파악함
# 
# 2. 전체적인 숫자 자체를 구해서 시각화하는건 큰 발견이 아니다. 그것 보다는 범죄자체의 밀도 즉 density를 구해서 대략적으로 어떤 특정지역에서 특정 범죄의 밀도수가 높은지 구해본다. 이를 적용했을 시 위에서 구한 특정 범죄가 특정 도시에서 보였던 일련의 것들이 시각화되어서 나타날 것이다 

# In[55]:


# 2번의 가설이 검증이 되는지 몇개만 따로 빼봐서 파악해본다 
# 밀도를 위해 쓰는 seaborn의 jointplot을 적용해본다(kdeplot도 가능)

# DRUG/NARCOTIC , LOITERING  추출
train_DRUG = train[train["Category"] == "DRUG/NARCOTIC"]
train_ARSON = train[train["Category"] == "ARSON"]

# DRUG
# sns.kdeplot(train_DRUG[["X", "Y"]])
sns.jointplot(x= "X", y = "Y", data = train_DRUG, kind= "kde")
plt.xlabel("DRUG")

# LOITERING
sns.jointplot(x= "X", y = "Y", data = train_ARSON, kind= "kde")
plt.xlabel("ARSON")


# DRUG/NARCOTIC의 범죄수는 54000 정도이고 / ARSON은 1500임. 
# 
# 즉 거의 35배 이상 차이나는 범죄수인데 분포도를 보면 DRUG/NARCOTIC의 범죄가 매우 적고 / ARSON의 범죄 수는 많아보이는 효과가 있다. 이는 DRUG/NARCOTIC의 범죄가 ARSON에 비해 큳버곳에 집중되어 있다는 이야기이다. 그리고 지도상 확인을 해보면 TENDERLOIN이고 이는 위에서 확인해본 결과이다. 
# 
# 이를 통해 밀도를 분석해보는 것이 효율적임을 알 수 있다

# #### Density of Category

# 총 39개의 Category범죄 중에 밀도로 보았을 시 한곳에 집중되는 경향을 보이는 몇개를 분석해본다

# In[56]:


# LARCENY/THEFT
# LARCENY/THEFT처럼 가장 범죄가 많은 경우 대다수로 분포되어 있을 가능성이 높다는점도 유의한다 

train_LARCENYTHEFT  = train[train["Category"] == "LARCENY/THEFT"]

fig = px.density_mapbox(train_LARCENYTHEFT, 
                        lat = train_LARCENYTHEFT["Y"], 
                        lon = train_LARCENYTHEFT["X"], 
                        zoom = 11, radius = 6, 
                        hover_name = "PdDistrict")

fig.update_layout(mapbox_style = "open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[57]:


# DRUG/NARCOTIC의 경우 서쪽에 압도적으로 포진이 많이 되어있다 

train_drug  = train[train["Category"] == "DRUG/NARCOTIC"]

fig = px.density_mapbox(train_drug, 
                        lat = train_drug["Y"], 
                        lon = train_drug["X"], 
                        zoom = 11, radius = 6, 
                        hover_name = "PdDistrict")

fig.update_layout(mapbox_style = "open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[58]:


# ASSAULT의 경우 Park지역에서 수가 감소한다

train_ASSAULT = train[train["Category"] == "ASSAULT"]

fig = px.density_mapbox(train_ASSAULT, 
                        lat = train_ASSAULT["Y"], 
                        lon = train_ASSAULT["X"], 
                        zoom = 11, radius = 6, 
                        hover_name = "PdDistrict")

fig.update_layout(mapbox_style = "open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[59]:


train_LOITERING = train[train["Category"] == "ROBBERY"]

fig = px.density_mapbox(train_LOITERING, 
                        lat = train_LOITERING["Y"], 
                        lon = train_LOITERING["X"], 
                        zoom = 11, radius = 6,
                        hover_name = "PdDistrict")

fig.update_layout(mapbox_style = "open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[60]:


# STOLENP ROPERTY

train_ROPERTY = train[train["Category"] == "STOLEN PROPERTY"]

fig = px.density_mapbox(train_ROPERTY, 
                        lat = train_ROPERTY["Y"], 
                        lon = train_ROPERTY["X"], 
                        zoom = 11, radius = 6,
                        hover_name = "PdDistrict")

fig.update_layout(mapbox_style = "open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[61]:


# LOITERING

train_LOITERING = train[train["Category"] == "LOITERING"]

fig = px.density_mapbox(train_LOITERING, 
                        lat = train_LOITERING["Y"], 
                        lon = train_LOITERING["X"], 
                        zoom = 11, radius = 6,
                        hover_name = "PdDistrict")

fig.update_layout(mapbox_style = "open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# 분석결과 다음과 같은 사실을 알 수 있다
# 
# 1. 범죄수가 많은 것들일수록 전국적으로 퍼져있다(drug는 제외). 이를 통해 상위범죄는 비단 특정지역에서만 일어나는게 아니라 범용적으로 일어나는 것이라 볼 수 있다.
# 
# 
# 2. 하위권으로 볼 수 있는 STOLEN PROPERTY / LOITERING를 주목해보면 일단 PROPERTY가 일어나는 곳은 가난한 곳보다는 부유 층이 많이 사는 곳일 가능성이 매우 높다. 앞서 시각화를 통해 서쪽 지역이 동쪽 지역보다 더 발전된 도시라는 사실을 감지하였고 이를 통해 다시한번 부유한 도시가 서쪽 상단부분에 몰려 있다는 사실을 암시해볼 수 있다. 따라서 이런 류의 범죄는 대부분 서쪽에서 발견될 가능성이 매우 높다. 또한 LOITERING 역시 공공장소가 상대적으로 많아야 일어날 수 있는 범죄라는 사실에 주목해야한다. 이를 통해 각 지역마다 범죄량이 차이가 보이는 것은 절대량이 많은 범죄수보다는 이처럼 하위권들에 주로 있는 단순 이상의 범죄가 서쪽 상단지역에 많이 몰려있기 때문이라는 점을 파악해야 한다.
# 
# 
# 3. 다른 하위권들과 달리 park를 시각화해보면 지역적인 특성을 고려해야한다고 본다. LARCENY/THEFT와 같은 최상위 범죄수 뿐 아니라 LOITERING를 보더라도 park중앙 부분은 대부분 범죄의 수가 적다(없는 것은 아님). 공공장소임을 감안한다면 그만큼 범죄가 벌어질 현장의 위치와 장소가 넓지 못하다는 뜻으로도 볼 수 있다. park중앙 부근에 큰 산들이 존재하고 있다는 사실을 간과해서는 안된다. 이를 감안해 범죄량을 측정해야하지 단순하게 park는 범죄수가 적다는 이유만으로 다른 부분을 놓치면 안된다

# ## Precessing

# ### feature_names / label names

# In[62]:


# year
# 만약에 "Dates-minute(fix)"를 쓸꺼면 "day"도 같이 써야 좋은 결과가 나온다
# 안쓸꺼면 그냥 minute로 간다
feature_names = ['Dates-year', 'Dates-month', 'Dates-hour', 'Dates-minute(fix)', 'Dates-day']
# weekday
feature_names = feature_names + list(['DayOfWeek_Monday', 'DayOfWeek_Tuesday','DayOfWeek_Wednesday',
                                      'DayOfWeek_Thursday', 'DayOfWeek_Friday', 'DayOfWeek_Saturday', 
                                      'DayOfWeek_Sunday'])
# PdDistrict
feature_names = feature_names + list(['PdDistrict__BAYVIEW','PdDistrict__CENTRAL', 'PdDistrict__INGLESIDE', 
                                      'PdDistrict__MISSION','PdDistrict__NORTHERN', 'PdDistrict__PARK', 
                                      'PdDistrict__RICHMOND','PdDistrict__SOUTHERN', 'PdDistrict__TARAVAL', 
                                      'PdDistrict__TENDERLOIN'])
# Address
feature_names = feature_names + list(['Address_road'])
# location 
feature_names = feature_names + list(['X', 'Y'])

# label_name
label_name = "Category"

# make
X_train = train[feature_names]
X_test = test[feature_names]
Y_train = train[label_name]


# Sparse Matrix(희소행렬) 사용 -> 값이 대부분 0일때 사용하는 툴
from scipy.sparse import csr_matrix, hstack

train_address = csr_matrix(train_address)
test_address = csr_matrix(test_address)

X_train = hstack([X_train.astype("float"), train_address]) 
X_test = hstack([X_test.astype("float"), test_address]) 


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)


# ## Selection of modeling

# #### 1) RandomForest

# In[63]:


from sklearn.ensemble import RandomForestClassifier

model_random = RandomForestClassifier(n_estimators = 10, n_jobs = -1,
                                      random_state = 37)
model_random


# #### 2) Gradient Boosting

# In[64]:


get_ipython().system('conda install -c conda-forge -y lightgbm')
from lightgbm import LGBMClassifier

model_gradiant = LGBMClassifier(n_estimators = 10, n_jobs = -1,
                       random_state = 37)
model_gradiant


# #### 3) Comparation using train_test_split

# In[65]:


from sklearn.model_selection import train_test_split

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_train, Y_train,
                                                           test_size = 0.3, 
                                                           random_state = 37)

print(X_train_t.shape)
print(y_train_t.shape)
print(X_test_t.shape)
print(y_test_t.shape)


# In[66]:


# randomforest

model_random.fit(X_train_t, y_train_t)

y_predict_random = model_random.predict_proba(X_test_t)
y_predict_random


# In[67]:


# Gradient Boosting

model_gradiant.fit(X_train_t, y_train_t)

y_predict_gradiant = model_gradiant.predict_proba(X_test_t)
y_predict_gradiant


# In[68]:


# using los_loss

# RandomForestClassifier
from sklearn.metrics import log_loss

Score_random = log_loss(y_test_t, y_predict_random)
print("RandomForestClassifier is : ", f"{Score_random : .8f}")


# LGBMClassifier
from sklearn.metrics import log_loss

Score = log_loss(y_test_t, y_predict_gradiant)
print("LGBMClassifier is : ", f"{Score : .8f}")


# Gradient Boosting 값이 log_loss가 더 낮음으로 이것으로 한다

# ## Calculating RobustScaler -> 시도하였으나 점수가 더 좋지 않음으로 패스한다

# In[69]:


# metrics 이후에 feature_names을 표준화하는 작업을 해본다

# StandardScaler, RobustScaler 중 RobustScaler 활용한다


# from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
# from sklearn.model_selection import train_test_split
# from lightgbm import LGBMClassifier
# from sklearn.metrics import log_loss   

# # feature_name standard적용
# X_train_standard = RobustScaler().fit(X_train).transform(X_train)
# X_test_standard = RobustScaler().fit(X_test).transform(X_test)

# x_train_std, x_test_std, y_train_std, y_test_std = train_test_split(X_train_standard, 
#                                                                     Y_train, 
#                                                                     test_size = 0.3, 
#                                                                     random_state = 37)

# print(x_train_std.shape)
# print(y_train_std.shape)
# print(x_test_std.shape) 
# print(y_test_std.shape) 

# model = LGBMClassifier(n_estimators = 10, n_jobs = -1,
#                        random_state = 37)

# model.fit(x_train_std, y_train_std)
# y_predict_std = model.predict_proba(x_test_std)
# y_predict_std


# # but 계산결과 점수가 더 좋아짐.
# score = log_loss(y_test_std, y_predict_std)
# score


# ## Model optimization process

# ### 1) First Search

# In[94]:


# 처음에 n_estimator를 100 ~ 300으로 정한다

from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import RobustScaler

# n_estimators = 300
random_search = 30

hyperparameter_list = []
early_stopping_rounds = 30

for loop in range(random_search):
    
    n_estimators = np.random.randint(100, 300)
    learning_rate = 10 ** -np.random.uniform(1, 2) 
    num_leaves = np.random.randint(2, 500)
    max_bin = np.random.randint(300, 500)
    min_child_samples = np.random.randint(300, 500)
    subsample = np.random.uniform(0.5, 1)
    colsample_bytree = np.random.uniform(0.5, 1) 
    reg_alpha = 10 ** - np.random.uniform(1, 10)
  #  subsample_freq = np.random.uniform(0.4, 1)
    
    model = LGBMClassifier(n_estimators = n_estimators,
                           learning_rate = learning_rate, 
                           max_bin = max_bin,
                           min_child_samples = min_child_samples,
                           subsample = subsample,
                           colsample_bytree = colsample_bytree,
                           reg_alpha = reg_alpha,
                           subsample_freq = 1,
                           n_jobs = -1,
                           class_type = "balanced",
                           random_state = 37)
    
    model.fit(X_train_t, y_train_t,
             early_stopping_rounds = early_stopping_rounds,
             verbose = 0,
             eval_set = [(X_test_t, y_test_t)])
    
    y_predict = model.predict_proba(X_test_t)
    
    score = log_loss(y_test_t, y_predict)
    
    hyperparameter = {"n_estimators" : n_estimators,
                      "score" : score,
                      "learning_rate" : learning_rate,
                      "max_bin" : max_bin,
                      "num_leaves" : num_leaves, 
                      "min_child_samples" : min_child_samples,
                      "subsample" : subsample,
                      "colsample_bytree" : colsample_bytree,
                      "reg_alpha" : reg_alpha}
    
    hyperparameter_list.append(hyperparameter)
    
    print(f"score = {score:.6f}, n_estimators = {n_estimators}, learning_rate = {learning_rate:.6f},    max_bin = {max_bin}, num_leaves = {num_leaves}, min_child_samples = {min_child_samples},    subsample = {subsample:.6f}, colsample_bytree = {colsample_bytree:.6f},    reg_alpha = {reg_alpha:.6f}")


# In[95]:


pd.DataFrame.from_dict(hyperparameter_list).sort_values(by = "score").head(10)


# ### 2) Final Search

# 위에서 나온 점수를 바탕으로 cross_validation(cross_val_score)로 바꿔서 재계산해본다

# In[55]:


from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, KFold

random_search = 20
hyperparameter_list = []
# classification임으로 StratifiedKFold를 사용한다
kf = StratifiedKFold(n_splits = 5, shuffle = False)
# early_stopping_rounds = 30

for loop in range(random_search):
    
    n_estimators = np.random.randint(100, 300)
    learning_rate = 10 ** - np.random.uniform(0.9, 3) 
    num_leaves = np.random.randint(200, 500)
    max_bin = np.random.randint(200, 500)
    min_child_samples = np.random.randint(200, 500)
    subsample = np.random.uniform(0.5, 1)
    colsample_bytree = np.random.uniform(0.1, 1) 
    reg_alpha = 10 ** - np.random.uniform(1, 10)
   # subsample_freq = np.random.uniform(0.4, 1)
    
    model = LGBMClassifier(n_estimators = n_estimators,
                           learning_rate = learning_rate, 
                           max_bin = max_bin,
                           min_child_samples = min_child_samples,
                           subsample = subsample,
                           colsample_bytree = colsample_bytree,
                           reg_alpha = reg_alpha,
                           subsample_freq = 1,
                           n_jobs = -1,
                           random_state = 37)
    
    # cross_val_score로 바꾼다
    score = -1.0 * cross_val_score(model, X_train, Y_train, cv = 5, 
                                   n_jobs = -1, scoring = 'neg_log_loss').mean()
    
    hyperparameter = {"score" : score,
                      "n_estimators" :n_estimators,
                      "learning_rate" : learning_rate,
                      "max_bin" : max_bin,
                      "num_leaves" : num_leaves, 
                      "min_child_samples" : min_child_samples,
                      "subsample" : subsample,
                      "colsample_bytree" : colsample_bytree,
                      "reg_alpha" : reg_alpha}
    
    hyperparameter_list.append(hyperparameter)
    
    print(f"score = {score:.6f}, learning_rate = {learning_rate:.6f},n_estimators = {n_estimators},    max_bin = {max_bin}, num_leaves = {num_leaves}, min_child_samples = {min_child_samples},    subsample = {subsample:.6f}, colsample_bytree = {colsample_bytree:.6f},reg_alpha = {reg_alpha:.6f}")


# In[56]:


pd.DataFrame.from_dict(hyperparameter_list).sort_values(by = "score").head(10)


# ## Outperform - LightGBM

# n_estimators는 200대, learning_rate가 0.0xxxxx 일때 점수가 가장 효율적임으로 이에 점수가 높은것을 진행한다

# In[184]:


from lightgbm import LGBMClassifier

model = LGBMClassifier(boosting_type = "gbdt",
                       n_estimators = 273,
                       learning_rate = 0.037242,
                       max_bin = 440,
                       min_child_samples = 369,
                       colsample_bytree = 0.786587,
                       num_leaves = 234, 
                       subsample = 0.936974,
                       subsample_freq = 1,
                       n_jobs = -1, 
                       random_state = 37,
                       reg_alpha = 0.002432)

model


# In[185]:


model.fit(X_train, Y_train)


# In[186]:


prediction_list = model.predict_proba(X_test)
prediction_list


# In[187]:


pre_submission = pd.read_csv("Desktop/phthon/Kaggle/sanfransico/sampleSubmission.csv", index_col = "Id")

print(pre_submission.shape)
pre_submission.head()


# In[188]:


submission = pd.DataFrame(prediction_list,
                          index = pre_submission.index,
                          columns = model.classes_)

submission.head()


# In[189]:


submission.to_csv("Desktop/phthon/Kaggle/sanfransico/Submission(final23).csv")


# ## Outperform - catboost

# In[73]:


get_ipython().system('pip install catboost')


# In[349]:


from catboost import CatBoostClassifier, CatBoost

model = CatBoostClassifier(#n_estimators = 2000,
                           iterations = 5000,
                           learning_rate = 0.01, 
                           loss_function = "MultiClass",
                           one_hot_max_size = 5,
                           # eval_metric='AUC',
                           # task_type = "CPU", 
                           # random_seed = 1234, 
                           verbose = True)

model


# In[350]:


model.fit(X_train, Y_train)#, cat_features = np.arange(len(X_train.columns)))


# In[351]:


prediction_list = model.predict_proba(X_test)
prediction_list


# In[352]:


pre_submission = pd.read_csv("Desktop/phthon/Kaggle/sanfransico/sampleSubmission.csv", index_col = "Id")

print(pre_submission.shape)
pre_submission.head()


# In[353]:


submission = pd.DataFrame(prediction_list,
                          index = pre_submission.index,
                          columns = model.classes_)

submission.head()


# In[354]:


submission.to_csv("Desktop/phthon/Kaggle/sanfransico/Submission(final4).csv")

