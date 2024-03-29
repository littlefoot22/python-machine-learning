# Basic machine learning with python

a basic machine learning script based on this tutorial https://machinelearningmastery.com/machine-learning-in-python-step-by-step/. This tutorial is referenced multiple times here as the 'iris example'

tools needed
* scipy
* numpy
* matplotlib
* pandas
* sklearn

install instructions here: https://www.scipy.org/install.html

dataset used http://archive.ics.uci.edu/ml/datasets/Wine

## names

first the names need to be added, we can pull these attributes from the wine.names file. One difference here from the iris example is the class is now a number and its the first entry into the names array

```python
names = ['class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'phenols', 'Flavanoids', 'Nonflavanoid', 'Proanthocyanins', 'Color', 'Hue', 'diluted wines', 'Proline']
```

### dimensions of dataset
just like we did in the iris example from the above tutorial we can peek at the data a little bit:

```python
print(dataset.shape)

#output
#(178, 14)

print(dataset.head(20))

#output
#    class  Alcohol  Malic acid   Ash  Alcalinity of ash  Magnesium  phenols  Flavanoids  Nonflavanoid  Proanthocyanins  Color   Hue  diluted wines  Proline
#0       1    14.23        1.71  2.43               15.6        127     2.80        3.06          0.28             2.29   5.64  1.04           3.92     1065
#1       1    13.20        1.78  2.14               11.2        100     2.65        2.76          0.26             1.28   4.38  1.05           3.40     1050
#2       1    13.16        2.36  2.67               18.6        101     2.80        3.24          0.30             2.81   5.68  1.03           3.17     1185
#3       1    14.37        1.95  2.50               16.8        113     3.85        3.49          0.24             2.18   7.80  0.86           3.45     1480
#4       1    13.24        2.59  2.87               21.0        118     2.80        2.69          0.39             1.82   4.32  1.04           2.93      735
#5       1    14.20        1.76  2.45               15.2        112     3.27        3.39          0.34             1.97   6.75  1.05           2.85     1450
#6       1    14.39        1.87  2.45               14.6         96     2.50        2.52          0.30             1.98   5.25  1.02           3.58     1290
#7       1    14.06        2.15  2.61               17.6        121     2.60        2.51          0.31             1.25   5.05  1.06           3.58     1295
#8       1    14.83        1.64  2.17               14.0         97     2.80        2.98          0.29             1.98   5.20  1.08           2.85     1045
#9       1    13.86        1.35  2.27               16.0         98     2.98        3.15          0.22             1.85   7.22  1.01           3.55     1045
#10      1    14.10        2.16  2.30               18.0        105     2.95        3.32          0.22             2.38   5.75  1.25           3.17     1510
#11      1    14.12        1.48  2.32               16.8         95     2.20        2.43          0.26             1.57   5.00  1.17           2.82     1280
#12      1    13.75        1.73  2.41               16.0         89     2.60        2.76          0.29             1.81   5.60  1.15           2.90     1320
#13      1    14.75        1.73  2.39               11.4         91     3.10        3.69          0.43             2.81   5.40  1.25           2.73     1150
#14      1    14.38        1.87  2.38               12.0        102     3.30        3.64          0.29             2.96   7.50  1.20           3.00     1547
#15      1    13.63        1.81  2.70               17.2        112     2.85        2.91          0.30             1.46   7.30  1.28           2.88     1310
#16      1    14.30        1.92  2.72               20.0        120     2.80        3.14          0.33             1.97   6.20  1.07           2.65     1280
#17      1    13.83        1.57  2.62               20.0        115     2.95        3.40          0.40             1.72   6.60  1.13           2.57     1130
#18      1    14.19        1.59  2.48               16.5        108     3.30        3.93          0.32             1.86   8.70  1.23           2.82     1680
#19      1    13.64        3.10  2.56               15.2        116     2.70        3.03          0.17             1.66   5.10  0.96           3.36      845

print(dataset.describe())

#output
#            class     Alcohol  Malic acid         Ash  Alcalinity of ash  ...  Proanthocyanins       Color         Hue  diluted wines      Proline
#count  178.000000  178.000000  178.000000  178.000000         178.000000  ...       178.000000  178.000000  178.000000     178.000000   178.000000
#mean     1.938202   13.000618    2.336348    2.366517          19.494944  ...         1.590899    5.058090    0.957449       2.611685   746.893258
#std      0.775035    0.811827    1.117146    0.274344           3.339564  ...         0.572359    2.318286    0.228572       0.709990   314.907474
#min      1.000000   11.030000    0.740000    1.360000          10.600000  ...         0.410000    1.280000    0.480000       1.270000   278.000000
#25%      1.000000   12.362500    1.602500    2.210000          17.200000  ...         1.250000    3.220000    0.782500       1.937500   500.500000
#50%      2.000000   13.050000    1.865000    2.360000          19.500000  ...         1.555000    4.690000    0.965000       2.780000   673.500000
#75%      3.000000   13.677500    3.082500    2.557500          21.500000  ...         1.950000    6.200000    1.120000       3.170000   985.000000
#max      3.000000   14.830000    5.800000    3.230000          30.000000  ...         3.580000   13.000000    1.710000       4.000000  1680.000000

print(dataset.groupby('class').size())

#output
#[8 rows x 14 columns]
#class
#1    59
#2    71
#3    48

```

looks good!

## histogram

after we run the histogram function and out put to file we get an idea the distribution between variables

```python
# histograms
dataset.hist()
plt.savefig('hist.png')
```
![alt text](https://github.com/littlefoot22/python-machine-learning/blob/master/images/hist.png "Histogram")

again we see lots of signs of normal distribution, which is a good sign for algorithms and predictions!

## scatterplot

after we run the scatter_matrix function and output to file we get an idea of the correlation between the different variables. This image is a bit harder to read then the iris example, but we can see some signs of highly positive correlation! This suggests a predictable relationship within the data.

```python
# histograms
scatter_matrix(dataset)
plt.savefig('scattermatrix.png')
```

![alt text](https://github.com/littlefoot22/python-machine-learning/blob/master/images/scattermatrix.png "Scatter Plot")


![alt text](http://www.cqeacademy.com/wp-content/uploads/2018/06/Scatter-Plots-and-Correlation-Examples.png "Correlation Examples")

## training and validation data

setting up the training and validation data is a little different then the iris example for this dataset. we need to split the X array from index 1 to 13 and the Y array at index 0. This is because our class attribute is now at the 0 index of the array and our data attributes make up the rest.

We can leave the split at 80/20 so we leave the validation_size at .20

```python
# Split-out validation dataset
array = dataset.values
X = array[:,1:13]
Y = array[:,0]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'
```


## selecting a model

now lets run the same models we did in the iris example and see if we get a hit

```python
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
#output
#LR: 0.930000 (0.070536)
#LDA: 0.979048 (0.032029)
#KNN: 0.808571 (0.115211)
#CART: 0.914762 (0.077033)
#NB: 0.944286 (0.068778)
#SVM: 0.859048 (0.077495)
```

nice! Looks like we have a score of 97 for the LinearDiscriminantAnalysis model, lets plug it into the data

```python
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)
predictions = lda.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

#output
#0.9722222222222222
#[[ 7  0  0]
# [ 0 16  1]
# [ 0  0 12]]
#              precision    recall  f1-score   support

#         1.0       1.00      1.00      1.00         7
#         2.0       1.00      0.94      0.97        17
#         3.0       0.92      1.00      0.96        12

#    accuracy                           0.97        36
#   macro avg       0.97      0.98      0.98        36
#weighted avg       0.97      0.97      0.97        36
```

looks like we hit 97% accuracy on our predictions against our training data and validation set! Thats even higher then the iris example! 

How exciting :D

