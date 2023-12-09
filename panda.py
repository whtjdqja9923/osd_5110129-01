import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")

def z_score(x):
    return (x - x.mean())/x.std()

df = pd.read_csv("./Sleep_Efficiency.csv", index_col='ID')
df2 = pd.read_csv("./Sleep_health_and_lifestyle_dataset.csv", index_col='Person ID')
# print(df)

df['z_Alcohol consumption'] = z_score(df['Alcohol consumption'])
df['z_Caffeine consumption'] = z_score(df['Caffeine consumption'])
df['Sleep efficiency'] = df['Sleep efficiency']*100
df.dropna(subset=['Exercise frequency'], axis=0, inplace=True)

ndf = df.loc[:, ['Gender', 'Exercise frequency', 'Sleep duration', 
                'Sleep efficiency', 'REM sleep percentage', 'Deep sleep percentage', 
                'Light sleep percentage', 'Awakenings', 'Caffeine consumption', 
                'Alcohol consumption']]

grouped_gender = ndf.groupby(['Gender'])
for key, group in grouped_gender:
    print('* key :', key)
    print('* number :', len(group))
    print(group.head(3))    
print('\n\n')        
            
grouped_two = ndf.groupby(['Gender', 'Exercise frequency'])
for key, group in grouped_two:
    print('* key :', key)
    print('* number :', len(group))
    print(group.head(3))
print('\n\n')

# 대상 : 전체 데이터
print('All data')
# 2개(성별, 활동지수)로 그룹화
print(grouped_two.agg(['mean', 'std']))
# 성별로 그룹화
print(grouped_gender.agg(['min', 'max']))
print('\n\n')

# --------------------------

# 대상 : 카페인섭취, 알코올섭취 특이값(z-score 2 이상, -2 이하) 제외한 데이터
mask_nan = (~df['z_Alcohol consumption'].isin([np.nan])) & (~df['z_Caffeine consumption'].isin([np.nan]))
mask_alcohol = (-2<=df['z_Alcohol consumption']) & (df['z_Alcohol consumption']<=2)
mask_caffine = (-2<=df['z_Caffeine consumption']) & (df['z_Caffeine consumption']<=2)

grouped_gender_z = ndf.loc[mask_nan&mask_alcohol&mask_caffine]\
                        .groupby(['Gender'])
            
grouped_two_z = ndf.loc[mask_nan&mask_alcohol&mask_caffine]\
                    .groupby(['Gender', 'Exercise frequency'])            
            
print('delete outlier')
# 2개(성별, 활동지수)로 그룹화
print(grouped_two_z.agg(['mean', 'std']))
# 성별로 그룹화
print(grouped_gender_z.agg(['min', 'max']))
print('\n\n')

# -----------------------------
pdf = pd.pivot_table(ndf,
                    index='Exercise frequency',
                    columns='Gender',
                    values=['Sleep duration', 'Sleep efficiency'],
                    aggfunc=['mean', 'std'])
print('pivot table')
print(pdf.head())
print('\n\n')

# ------------------------------
# ndf.plot(kind='scatter', y='Exercise frequency', x='Sleep efficiency', figsize=(10, 5))
# plt.show()
# plt.close()

# ------------------------------
mask_age = (0 <= df['Age']) & (df['Age'] <= 19)
mask_gender = (df['Gender'] == 'Female')

ndf = df
# grid_ndf = sns.pairplot(ndf, kind='reg')
# plt.show()
# plt.close()

fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)
sns.regplot(x='Exercise frequency', y='Sleep efficiency', data=ndf, ax=ax1)
sns.regplot(x='Sleep duration', y='Sleep efficiency', data=ndf, ax=ax2)
sns.regplot(x='Awakenings', y='Sleep efficiency', data=ndf, ax=ax3)
sns.regplot(x='Alcohol consumption', y='Sleep efficiency', data=ndf, ax=ax4)
plt.show()
plt.close()

# ------------------------------
x = df[['Exercise frequency']]
y = df['Sleep efficiency']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)
print('train data :', len(x_train))
print('test data :', len(x_test))
print('\n\n')

poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train)

# ----------------------------
lr = LinearRegression()
lr.fit(x_train, y_train)

print('*****lr*****')
print('기울기 : ', lr.coef_, end='\n')
print('y절편 : ', lr.intercept_)
print('\n\n')

y_hat = lr.predict(x_test)

plt.figure(figsize=(15, 10))
ax1_lr = sns.distplot(y, hist=False, label='y_train')
ax2_lr = sns.distplot(y_hat, hist=False, label='y_predict', ax=ax1_lr)
ax3_lr = sns.distplot(y_test, hist=False, label='y_test', ax=ax1_lr)
ax1_lr.legend(loc='best')
plt.show()
plt.close()

print('lr model mse : ')
print(metrics.mean_squared_error(y_test, y_hat))
print('\n\n')

# ---------------------------
pr = LinearRegression()
pr.fit(x_train_poly, y_train)

print('*****pr*****')
print('기울기 : ', pr.coef_, end='\n')
print('y절편 : ', pr.intercept_)
print('\n\n')

x_test_poly = poly.fit_transform(x_test)
y_hat_poly = pr.predict(x_test_poly)

plt.figure(figsize=(15, 10))
ax1_pr = sns.distplot(y, hist=False, label='y_train')
ax2_pr = sns.distplot(y_hat_poly, hist=False, label='y_predict_poly', ax=ax1_pr)
ax3_pr = sns.distplot(y_test, hist=False, label='y_test', ax=ax1_pr)
ax1_pr.legend(loc='best')
plt.show()
plt.close()

print('pr model mse : ')
print(metrics.mean_squared_error(y_test, y_hat_poly))
print('\n\n')