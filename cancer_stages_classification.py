from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.manifold import TSNE
from numpy import array
from numpy import argmax
from ggplot import *
import pandas as pd
import numpy as np




#######################
### ~ import data ~ ###
#######################


df_train = pd.read_table('./BRCA_train.data')
df_test = pd.read_table('./BRCA_test.data')
df_valid = pd.read_table('./BRCA_valid.data')
df_solution = pd.read_table('./BRCA_train.solution')

print(df_train.shape)
print(df_test.shape)
print(df_valid.shape)


#getting dimensions of data
dims = df_train.shape
dim_rows = dims[0]
dim_columns = dims[1]
dim_genetics = dim_columns - 17
df_train_genetic = df_train.iloc[:,0:dim_genetics]

#general stats
stages = df_solution['x'].tolist()
num_stage_1 = stages.count('stage1')
num_stage_2 = stages.count('stage2')
num_stage_3 = stages.count('stage3')

print('#stage 1 : ' + str(num_stage_1))
print('#stage 2 : ' + str(num_stage_2))
print('#stage 3 : ' + str(num_stage_3))

min_count = min(num_stage_1,num_stage_2,num_stage_3)



##########################################
###      ~ Pre-processing phase ~      ###
##########################################



# ~ Normalization of numerical values
x = df_train_genetic.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_train_genetic_scaled = pd.DataFrame(x_scaled,index=df_train_genetic.index)


# ~ Applying PCA to reduce dimensionality 
n_components = 25
pca = PCA(n_components=n_components)
x_pca = pca.fit_transform(x_scaled)
df_train_genetic_reduced = pd.DataFrame(x_pca,columns=['PCA%i' % i for i in range(n_components)], index=df_train_genetic_scaled.index)


#explained variation per principal component - pourcentage of total data variation conveyed by each PCA component
pca.explained_variance_ratio_


# ~ Non-genetical data

df_train_non_genetic = df_train.iloc[:,dim_genetics+1:dim_columns]

non_genetic_non_numerical = ['tumortissuesite','pathologyTstage','pathologyNstage','pathologyMstage','gender','daystolastknownalive','radiationtherapy','histologicaltype','race','ethnicity']
non_genetic_numerical = [var for var in list(df_train_non_genetic) if var not in non_genetic_non_numerical]


# scaling of non genetic numerical data
df_train_non_genetic_numerical = df_train_non_genetic[non_genetic_numerical]
df_train_non_genetic_numerical = df_train_non_genetic_numerical.fillna(0)


x = df_train_non_genetic_numerical.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_train_non_genetic_numerical_scaled = pd.DataFrame(x_scaled,index=df_train_non_genetic_numerical.index)


# categories for non genetic non numerical data
df_train_non_genetic_non_numerical = df_train_non_genetic[non_genetic_non_numerical]
df_train_non_genetic_non_numerical = df_train_non_genetic_non_numerical.fillna('0')


for var in non_genetic_non_numerical:
	df_train_non_genetic_non_numerical[var] = df_train_non_genetic_non_numerical[var].astype('category')

cat_columns = df_train_non_genetic_non_numerical.select_dtypes(['category']).columns
df_train_non_genetic_non_numerical_vectorized = df_train_non_genetic_non_numerical[cat_columns].apply(lambda x: x.cat.codes)


#merging all together
df_train_non_genetic_pre_processed = pd.merge( df_train_non_genetic_numerical_scaled, df_train_non_genetic_non_numerical_vectorized, left_index=True, right_index=True, how='outer')



# ~ Sampling each classes


df_train_both = pd.merge(df_train_genetic_reduced, df_train_non_genetic_pre_processed, left_index=True, right_index=True, how='outer')
df_train_both_with_target = pd.merge( df_train_both, df_solution, left_index=True, right_index=True, how='outer')


def sampling_dataset(df,target,count):
    columns = df.columns
    class_df_sampled = pd.DataFrame(columns = columns)
    temp = []
    for c in df[target].unique():
        class_indexes = df[df[target] == c].index
        random_indexes = np.random.choice(class_indexes, count, replace=False)
        temp.append(df.loc[random_indexes])
        
    for each_df in temp:
        class_df_sampled = pd.concat([class_df_sampled,each_df],axis=0)
    
    return class_df_sampled


df_train_both_with_target_sampled = sampling_dataset( df_train_both_with_target, 'x', min_count)



#############################
###   ~ Model training ~  ###
#############################


target_name = 'x'
features = df_train_both_with_target_sampled.drop(target_name,axis=1)
target = df_train_both_with_target_sampled[target_name]


rf = RandomForestClassifier(bootstrap=True, criterion='gini', max_features='sqrt') 
rf.fit(features, target)


#cross val
scores = cross_val_score(rf, features, target, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



##################################
###   ~ Test and validation ~  ###
##################################


### ~ valid data ~ ###


df_valid_genetic = df_valid.iloc[:,0:dim_genetics]

# ~ Normalization of numerical values
x = df_valid_genetic.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_valid_genetic_scaled = pd.DataFrame(x_scaled,index=df_valid_genetic.index)


# ~ Applying PCA to reduce dimensionality 
n_components = 25
pca = PCA(n_components=n_components)
x_pca = pca.fit_transform(x_scaled)
df_valid_genetic_reduced = pd.DataFrame(x_pca,columns=['PCA%i' % i for i in range(n_components)], index=df_valid_genetic_scaled.index)


#explained variation per principal component - pourcentage of total data variation conveyed by each PCA component
pca.explained_variance_ratio_


# ~ Non-genetical data

df_valid_non_genetic = df_valid.iloc[:,dim_genetics+1:dim_columns]

non_genetic_non_numerical = ['tumortissuesite','pathologyTstage','pathologyNstage','pathologyMstage','gender','daystolastknownalive','radiationtherapy','histologicaltype','race','ethnicity']
non_genetic_numerical = [var for var in list(df_valid_non_genetic) if var not in non_genetic_non_numerical]


# scaling of non genetic numerical data
df_valid_non_genetic_numerical = df_valid_non_genetic[non_genetic_numerical]
df_valid_non_genetic_numerical = df_valid_non_genetic_numerical.fillna(0)


x = df_valid_non_genetic_numerical.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_valid_non_genetic_numerical_scaled = pd.DataFrame(x_scaled,index=df_valid_non_genetic_numerical.index)


# categories for non genetic non numerical data
df_valid_non_genetic_non_numerical = df_valid_non_genetic[non_genetic_non_numerical]
df_valid_non_genetic_non_numerical = df_valid_non_genetic_non_numerical.fillna('0')


for var in non_genetic_non_numerical:
	df_valid_non_genetic_non_numerical[var] = df_valid_non_genetic_non_numerical[var].astype('category')

cat_columns = df_valid_non_genetic_non_numerical.select_dtypes(['category']).columns
df_valid_non_genetic_non_numerical_vectorized = df_valid_non_genetic_non_numerical[cat_columns].apply(lambda x: x.cat.codes)



#merging all together
df_valid_non_genetic_pre_processed = pd.merge( df_valid_non_genetic_numerical_scaled, df_valid_non_genetic_non_numerical_vectorized, left_index=True, right_index=True, how='outer')


df_valid_both = pd.merge(df_valid_genetic_reduced, df_valid_non_genetic_pre_processed, left_index=True, right_index=True, how='outer')


### ~ test data ~ ###


df_test_genetic = df_test.iloc[:,0:dim_genetics]

# ~ Normalization of numerical values
x = df_test_genetic.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_test_genetic_scaled = pd.DataFrame(x_scaled,index=df_test_genetic.index)


# ~ Applying PCA to reduce dimensionality 
n_components = 25
pca = PCA(n_components=n_components)
x_pca = pca.fit_transform(x_scaled)
df_test_genetic_reduced = pd.DataFrame(x_pca,columns=['PCA%i' % i for i in range(n_components)], index=df_test_genetic_scaled.index)


#explained variation per principal component - pourcentage of total data variation conveyed by each PCA component
pca.explained_variance_ratio_


# ~ Non-genetical data

df_test_non_genetic = df_test.iloc[:,dim_genetics+1:dim_columns]

non_genetic_non_numerical = ['tumortissuesite','pathologyTstage','pathologyNstage','pathologyMstage','gender','daystolastknownalive','radiationtherapy','histologicaltype','race','ethnicity']
non_genetic_numerical = [var for var in list(df_test_non_genetic) if var not in non_genetic_non_numerical]


# scaling of non genetic numerical data
df_test_non_genetic_numerical = df_test_non_genetic[non_genetic_numerical]
df_test_non_genetic_numerical = df_test_non_genetic_numerical.fillna(0)


x = df_test_non_genetic_numerical.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_test_non_genetic_numerical_scaled = pd.DataFrame(x_scaled,index=df_test_non_genetic_numerical.index)


# categories for non genetic non numerical data
df_test_non_genetic_non_numerical = df_test_non_genetic[non_genetic_non_numerical]
df_test_non_genetic_non_numerical = df_test_non_genetic_non_numerical.fillna('0')

for var in non_genetic_non_numerical:
	df_test_non_genetic_non_numerical[var] = df_test_non_genetic_non_numerical[var].astype('category')


cat_columns = df_test_non_genetic_non_numerical.select_dtypes(['category']).columns
df_test_non_genetic_non_numerical_vectorized = df_test_non_genetic_non_numerical[cat_columns].apply(lambda x: x.cat.codes)


#merging all together
df_test_non_genetic_pre_processed = pd.merge( df_test_non_genetic_numerical_scaled, df_test_non_genetic_non_numerical_vectorized, left_index=True, right_index=True, how='outer')

df_test_both = pd.merge(df_test_genetic_reduced, df_test_non_genetic_pre_processed, left_index=True, right_index=True, how='outer')


#############################
###   ~ Prediction ~  ###
#############################


valid_predict = rf.predict(df_valid_both.values)
test_predict = rf.predict(df_test_both.values)

# One-Hot encoding of data

def onehot_encoding(data):

	result = []

	for elem in data:
		if elem == 'stage1':
			result.append([1,0,0])
		if elem == 'stage2':
			result.append([0,1,0])
		if elem == 'stage3':
			result.append([0,0,1])

	return result


encoded_valid = pd.DataFrame(onehot_encoding(valid_predict))
encoded_test = pd.DataFrame(onehot_encoding(test_predict))


with open('./BRCA_valid.predict','w') as outfile:
    encoded_valid.to_string(outfile)

with open('./BRCA_test.predict','w') as outfile:
    encoded_test.to_string(outfile)



#######################
###   ~ Data Viz ~  ###
#######################


df_train_both = pd.merge(df_train_genetic_scaled , df_train_non_genetic_pre_processed, left_index=True, right_index=True, how='outer')
df_train_both_with_target = pd.merge( df_train_both, df_solution, left_index=True, right_index=True, how='outer')


#Applying t-SNE to vizualize data
tsne = TSNE(n_components=3, verbose=1, perplexity=40, n_iter=300)
x = df_train_both.values
x_tsne = tsne.fit_transform(x)

df_tsne = pd.DataFrame()
df_tsne['x-tsne'] = x_tsne[:,0]
df_tsne['y-tsne'] = x_tsne[:,1]
df_tsne['label'] = df_solution.values


chart = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='label') ) \
        + geom_point(size=70,alpha=0.1) \
        + ggtitle("tSNE dimensions colored by cancer stage")

print(chart)
