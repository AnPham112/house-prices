import pandas as pd
import numpy as np
from sklearn.preprocessing import scale, LabelEncoder
from sklearn.model_selection import train_test_split

import neural_network as net
# begin cleaning data
train_data = pd.read_csv('kaggle/input/train.csv')
test_data = pd.read_csv('kaggle/input/test.csv')

data = train_data
features = ['LotArea', 'BldgType', 'OverallQual', 'YearBuilt', 'Foundation', 'Electrical', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'GarageCars', 'SalePrice']
test_features = ['LotArea', 'BldgType', 'OverallQual', 'YearBuilt', 'Foundation', 'Electrical', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'GarageCars']
headings = list(data)
miss_columns = dict()
for name in headings:
    if (data[name].isnull().mean() > 0.8):
        miss_columns[name] = data[name].isnull().sum()

df = pd.DataFrame(data={'MSx': miss_columns.keys(), 'MSy': miss_columns.values()})
df = df.sort_values(by=['MSy'], ascending=False)

# Drop missing values and Id column
train_data = train_data.drop(['Id'] + list(miss_columns), axis=1)
train_data.fillna(0, inplace=True)
train_data = train_data[features]
test_data = test_data.drop(['Id'] + list(miss_columns), axis=1)
test_data.fillna(0, inplace=True)
test_data = test_data[test_features]

num_features = train_data.select_dtypes(exclude='object').columns
cat_features = train_data.select_dtypes(include='object').columns

for cat_feature in cat_features:
    lbl = LabelEncoder() 
    lbl.fit(list(train_data[cat_feature].values)) 
    train_data[cat_feature] = lbl.transform(list(train_data[cat_feature].values))

for cat_feature in cat_features:
    lbl = LabelEncoder() 
    lbl.fit(list(test_data[cat_feature].values)) 
    test_data[cat_feature] = lbl.transform(list(test_data[cat_feature].values))

train, test = train_test_split(train_data, test_size=0.3, random_state=0)

y_train = train[['SalePrice']]
X_train = train.drop(['SalePrice'], axis=1)
y_test = test[['SalePrice']]
X_test = test.drop(['SalePrice'], axis=1)


# Normalize with average 0 and std = 1
X_train = scale(X_train)
X_test = scale(X_test)
test_data = scale(test_data)

y_train = np.divide(np.float32(y_train), 100000.0)
y_test = np.divide(np.float32(y_test), 100000.0)
X_train = np.float32(X_train)
X_test = np.float32(X_test)
test_data = np.float32(test_data)

input = X_train.shape[1]
output = y_train.shape[1]

nn = net.NeuralNetwork([
    net.Layer(input,output,is_last=True)])

for i_epochs in range(500):
  net.train_nn(nn,X_train,y_train,learning_rate=0.1)
  train_loss = net.loss_mse(y_train,nn(X_train))
  test_loss = net.loss_mse(y_test,nn(X_test))
  print("Training loss in epoch {0} = {1}.   Test loss = {2}".format(i_epochs,train_loss.numpy(),test_loss.numpy()))

test_predictions = np.multiply(nn(test_data).numpy(), 100000).flatten()

submission_df = pd.DataFrame({'SalePrice': test_predictions}, index=range(1461, 2920))
print(submission_df)
submission_df.to_csv('submit.csv')