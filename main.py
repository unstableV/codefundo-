import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats
%matplotlib inline
plt.rcParams['figure.figsize']=(15,8)
train = pd.read_csv('train.csv')
train.head()
train.columns
bs = pd.read_csv('Building_Structure.csv')
bs.head()
print(bo.shape)
print(bs.shape)
print(building.shape)
train.shape
test = pd.read_csv('test.csv')
test.head()
test.shape
train = pd.merge(train, building, on='building_id')
test = pd.merge(test, building, on='building_id')
train.shape
test.shape
train.head()
train.columns

train.describe()
train.info()
train.duplicated().sum()
train['has_geotechnical_risk'].value_counts().plot(kind='bar')
sns.countplot(x='damage_grade', hue='has_geotechnical_risk', data=train)
sns.countplot(x='damage_grade', hue='has_geotechnical_risk_fault_crack', data=train)
for col in train.columns.tolist():
    if 'has' in col:
        sns.countplot(x='damage_grade', hue=col, data=train)
        fig = plt.figure()
[col for col in train.columns.tolist() if 'has' in col]
train.drop(['has_geotechnical_risk_fault_crack','has_geotechnical_risk_flood', 'has_geotechnical_risk_liquefaction',
            'has_geotechnical_risk_other', 'has_geotechnical_risk_land_settlement', 'has_geotechnical_risk_liquefaction',
            'has_geotechnical_risk_other', 'has_superstructure_cement_mortar_stone','has_superstructure_bamboo',
            'has_superstructure_rc_engineered', 'has_superstructure_other', 'has_secondary_use_hotel', 
            'has_secondary_use_rental', 'has_secondary_use_institution', 'has_secondary_use_school', 
            'has_secondary_use_industry', 'has_secondary_use_health_post', 'has_secondary_use_gov_office', 
            'has_secondary_use_use_police','has_secondary_use_other',], axis=1, inplace=True)
for col in train.columns.tolist():
    if 'has' in col:
        sns.countplot(x='damage_grade', hue=col, data=train)
        fig = plt.figure()
train.info()
for i, v in enumerate(train.columns.tolist()):
    print(i, v)
features = train.iloc[:, np.r_[0:3, 4:8, 12:33, 38, 39]]
features.info()
for i, v in enumerate(features.columns.tolist()):
    print(i, v)

from sklearn.preprocessing import LabelEncoder
obj_vars = features.iloc[:,np.r_[0, 13:20, 27]]
objToNumCat = obj_vars.apply(LabelEncoder().fit_transform)
features.iloc[:, np.r_[0, 13:20, 27]] = objToNumCat
# doing separately for label.
def repl(a):
    if '1' in a:
        return 0
    elif '2' in a:
        return 1
    elif '3' in a:
        return 2
    elif '4' in a:
        return 3
    else:
        return 4

features['damage_grade'] = features['damage_grade'].apply(lambda x: repl(x))
features.info()
plt.rcParams['figure.figsize']=(15, 8)
fig, ax = plt.subplots(ncols=2, nrows=1)
sns.countplot(x='count_floors_pre_eq', data=features, ax=ax[0])
sns.countplot(x='count_floors_post_eq', data=features, ax=ax[1])
plt.tight_layout()
plt.rcParams['figure.figsize']=(15,10)
f, ax = plt.subplots(ncols=2, nrows=2)
sns.boxplot(x='damage_grade', y='count_floors_pre_eq', data=features, ax=ax[0][0])
ax[0][0].set_title('pre')
sns.boxplot(x='damage_grade', y='count_floors_post_eq', data=features, ax=ax[0][1])
ax[0][1].set_title('post')
sns.countplot(x='count_floors_pre_eq', data=features, ax=ax[1][0])
ax[1][0].set_title('countplot_floors_pre')
sns.countplot(x='count_floors_post_eq', data=features, ax=ax[1][1])
ax[1][1].set_title('countplot_floors_post')
plt.tight_layout()
idcollist = features[(features['count_floors_pre_eq']>=6 )| (features['count_floors_post_eq']>=6 )]['building_id'].tolist()
len(idcollist)
for id in idcollist:
    features.drop(features[features['building_id']==id].index, inplace=True)

f, ax = plt.subplots(ncols=2, nrows=2)
sns.boxplot(x='damage_grade', y='count_floors_pre_eq', data=features, ax=ax[0][0])
ax[0][0].set_title('pre')
sns.boxplot(x='damage_grade', y='count_floors_post_eq', data=features, ax=ax[0][1])
ax[0][1].set_title('post')
sns.countplot(x='count_floors_pre_eq', data=features, ax=ax[1][0])
ax[1][0].set_title('countplot_floors_pre')
sns.countplot(x='count_floors_post_eq', data=features, ax=ax[1][1])
ax[1][1].set_title('countplot_floors_post')
plt.tight_layout()
sns.boxplot(x='damage_grade', y='age_building', data=features)
bid_list = features[features['age_building']==999]['building_id'].tolist()
for bid in bid_list:
    features.drop(features[features['building_id'] == bid].index, inplace=True)
#     features = features.drop(features[features['building_id']==bid].index)
plt.rcParams['figure.figsize']=(15, 8)
f, ax = plt.subplots(ncols=2, nrows=1)
sns.boxplot(x='damage_grade', y='plinth_area_sq_ft', data=features, ax=ax[0])
sns.distplot(features['plinth_area_sq_ft'], ax=ax[1], fit=norm)
plt.tight_layout()
print('Skewness of plinth_area_sq_ft', features['plinth_area_sq_ft'].skew())
features['plinth_area_normal'] = np.log(features['plinth_area_sq_ft'])
sns.distplot(features['plinth_area_normal'], fit=norm)
fig = plt.figure()
res = stats.probplot(features['plinth_area_normal'], plot=plt);
sns.distplot(features['height_ft_pre_eq'], fit=norm)

features['height_ft_pre_eq'].describe()
sns.boxplot(x='damage_grade', y='height_ft_pre_eq', data=features)
temp_list = features[features['height_ft_pre_eq'] >=60]['building_id'].tolist()
for id in temp_list:
    features.drop(features[features['building_id'] == id].index, inplace=True)
f, ax = plt.subplots(ncols=2, nrows=1)
sns.boxplot(x='damage_grade', y='height_ft_pre_eq', data=features, ax=ax[0])
sns.distplot(features['height_ft_pre_eq'], fit=norm, ax=ax[1])
log_pre_ht = np.log(features['height_ft_pre_eq'])
sns.distplot(log_pre_ht, fit=norm, bins=15)
features['height_pre_normal'] = log_pre_ht
f, ax = plt.subplots(ncols=2, nrows=1)
print('for height post eq')
sns.boxplot(x='damage_grade', y='height_ft_post_eq', data=features, ax=ax[0])
sns.distplot(features['height_ft_post_eq'], fit=norm, ax=ax[1])
temp_list = features[features['height_ft_post_eq'] >=60]['building_id'].tolist()
for id in temp_list:
    features.drop(features[features['building_id'] == id].index, inplace=True)
f, ax = plt.subplots(ncols=2, nrows=1)
print('for height post eq')
sns.boxplot(x='damage_grade', y='height_ft_post_eq', data=features, ax=ax[0])
sns.distplot(features['height_ft_post_eq'], fit=norm, ax=ax[1])
plt.tight_layout()
features.columns
floor_pos_feat = ['land_surface_condition', 'foundation_type','roof_type', 'ground_floor_type', 
                  'other_floor_type', 'position', 'plan_configuration', 'condition_post_eq']
for item in floor_pos_feat:
    sns.countplot(x='damage_grade', hue=item, data=features)
    fig = plt.figure()
features.drop(['plan_configuration', 'land_surface_condition', 'foundation_type', 'position'], axis=1, inplace=True)
features.isnull().sum().sort_values(ascending=False)
f, ax = plt.subplots(ncols=2, nrows=1)
sns.countplot(hue='has_repair_started', ax=ax[0],  data=features, x='damage_grade')
sns.countplot(x='has_repair_started', data=features, ax=ax[1], hue='damage_grade')
plt.tight_layout()
def fix_repair(cols):
    val = cols[0]
    label = cols[1]
    if pd.isnull(val):
        if label>=2:
            return 0
        else:
            return 1
    else:
        return val
features['has_repair_started'] = features[['has_repair_started', 'damage_grade']].apply(fix_repair, axis=1)
features.isnull().sum()
features.columns
# To do some experimental steps!
features_cat = features.copy()
features_cat = features.iloc[:, np.r_[2:7, 13:24, 25, 26]]
features_cat.drop('plinth_area_normal', axis=1, inplace=True)
features_cat.head()
features_cat.info()
features_cat.columns
features_cat.shape
from scipy.stats import chisquare
chi_test = chisquare(features_cat, axis=0)
for col in features_cat.columns:
    print(pd.crosstab(features_cat[col].values, features_cat['damage_grade'], rownames=[col], colnames=['damage_grade']), '\n\n')
features.drop(['has_secondary_use_agriculture', 'condition_post_eq'], axis=1, inplace=True)
features_cat.drop(['has_secondary_use_agriculture', 'condition_post_eq'], axis=1, inplace=True)
from scipy.stats import chi2_contingency
for col in features_cat.columns:
    print(col, ':')
    print('\n', chi2_contingency(pd.crosstab(features[col],features['damage_grade'])), '\n\n')
for col in features_cat.columns:
    pd.crosstab(features_cat['damage_grade'], features_cat[col].values, rownames=['damage_grade'], colnames=[col]).plot(kind='bar')
    fig = plt.figure()
features.drop(['has_geotechnical_risk_rock_fall',  'has_superstructure_stone_flag',  'ground_floor_type'], axis=1, inplace=True)
features.columns
from scipy.stats import chisquare
df_fact=features_cat.apply(lambda x : pd.factorize(x)[0])+1
df_fact_chi = pd.DataFrame([chisquare(df_fact[x].values,f_exp=df_fact.values.T,axis=1)[0] for x in df_fact])

df_fact_chi
from sklearn.model_selection import train_test_split
X = features.drop(['damage_grade', 'building_id', 'plinth_area_sq_ft', 'height_ft_pre_eq'], axis=1)

y = features['damage_grade']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=14)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

clf = RandomForestClassifier(n_estimators=150)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
from sklearn.metrics import accuracy_score
print('Accuracy from RandomForest, without OneHotEncoding', accuracy_score(y_test, predictions))
print('Accuracy from RandomForest, without OneHotEncoding', accuracy_score(y_test, predictions))
knn_clf = KNeighborsClassifier(n_neighbors=35)
knn_clf.fit(X_train, y_train)
prediction_knn = knn_clf.predict(X_test)
print('Accuracy from KNNC is without OneHotEncoding', accuracy_score(y_test, prediction_knn))
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
onevsrest_rf = OneVsRestClassifier(RandomForestClassifier(n_estimators=200))
onevsrest_rf.fit(X_train, y_train)
predictions_RF = onevsrest_rf.predict(X_test)
from sklearn.metrics import accuracy_score
print('Accuracy from RandomForestClassifier(one vs rest)', accuracy_score(y_test, predictions_RF))
X.columns
onevsrest_rf.fit(X, y)
test.shape
test.drop(['has_geotechnical_risk_fault_crack','has_geotechnical_risk_flood', 'has_geotechnical_risk_liquefaction',
            'has_geotechnical_risk_other', 'has_geotechnical_risk_land_settlement', 'has_geotechnical_risk_liquefaction',
            'has_geotechnical_risk_other', 'has_superstructure_cement_mortar_stone','has_superstructure_bamboo',
            'has_superstructure_rc_engineered', 'has_superstructure_other', 'has_secondary_use_hotel', 
            'has_secondary_use_rental', 'has_secondary_use_institution', 'has_secondary_use_school', 
            'has_secondary_use_industry', 'has_secondary_use_health_post', 'has_secondary_use_gov_office', 
            'has_secondary_use_use_police','has_secondary_use_other',], axis=1, inplace=True)
for i, v in enumerate(test.columns.tolist()):
    print(i, v)
test_features = test.iloc[:, np.r_[0, 1, 3:7, 11:32, 37, 38]]
for i, v in enumerate(test_features.columns.tolist()):
    print(i, v)
test_features.drop(['plan_configuration', 'land_surface_condition', 'foundation_type', 'position', 'has_secondary_use_agriculture', 'condition_post_eq'], axis=1, inplace=True)
test_features.drop(['has_superstructure_stone_flag', 'ground_floor_type', 'has_geotechnical_risk_rock_fall'], axis=1, inplace=True)
for i, v in enumerate(test_features.columns):
    print(i, v)
obj_vars = test_features.iloc[:,np.r_[0, 11, 12]]
objToNumCat = obj_vars.apply(LabelEncoder().fit_transform)
test_features.iloc[:, np.r_[0, 11, 12]] = objToNumCat
test_features.info()
test_features['plinth_area_normal'] = np.log(test_features['plinth_area_sq_ft'])
log_pre_ht = np.log(test_features['height_ft_pre_eq']+1)
test_features['height_pre_normal'] = log_pre_ht
test_features.isnull().sum().sort_values(ascending=False)
pd.crosstab(features_cat['has_repair_started'], features_cat['roof_type'].values, rownames=['has_repair_started'], colnames=['roof_type']).plot(kind='bar')
def fix_repair_test(cols):
    repair = cols[0]
    roof_type = cols[1]
    
    if pd.isnull(repair):
        if roof_type == 0:
            return 0
        elif roof_type== 2:
            return 1
        else:
            return 0
    else:
        return repair
test_features['has_repair_started'] = test_features[['has_repair_started', 'roof_type']].apply(fix_repair_test, axis=1)
test_features.isnull().sum().sort_values(ascending=False)
for i, v in enumerate(test_features.columns):
    print(i, v)
testing_data = test_features.iloc[:, np.r_[0, 2:8, 10, 11:21, 21]]
for i, v in enumerate(testing_data.columns):
    print(i,v)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=150)
classifier.fit(X, y)
from sklearn.multiclass import OneVsRestClassifier
ovr_clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=150))
ovr_clf.fit(X, y)
classifier_200 = RandomForestClassifier(n_estimators=200)
classifier_200.fit(X, y)
prediction_150 = classifier.predict(testing_data)
prediction_ovr = ovr_clf.predict(testing_data)
prediction_200 = classifier_200.predict(testing_data)
prediction_150
prediction_ovr
prediction_200
len(prediction_150)
len(prediction_200)
def numToGrade(value):
    if value == 0:
        return 'Grade 1'
    elif value == 1:
        return 'Grade 2'
    elif value == 2:
        return 'Grade 3'
    elif value == 3:
        return 'Grade 4'
    elif value == 4:
        return 'Grade 5'
prediction_df = pd.DataFrame({'building_id': test['building_id'], 'damage_grade': prediction_ovr})
prediction_df.head()
prediction_dfcp = prediction_df.copy()
prediction_dfcp['damage_grade'] = prediction_dfcp['damage_grade'].apply(numToGrade)
prediction_dfcp.head()
submission_cp.to_csv('predictions.csv', index=False)
prediction_knn = knn_clf.predict(X_test)
print('Accuracy from KNNC is without OneHotEncoding', accuracy_score(y_test, prediction_knn))
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=np.r_[0, 9:11, 19].tolist())
X_ohe = onehotencoder.fit_transform(X).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=14)
knn_clf.fit(X_train, y_train)
prediction_knn_ohe = knn_clf.predict(X_test)
print('Accuracy from KNNC is with OneHotEncoding', accuracy_score(y_test, prediction_knn_ohe))
