# Basic machine learning with python

a basic machine learning script based on this turtorial https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

tools needed
* scipy
* numpy
* matplotlib
* pandas
* sklearn

install instructions here: https://www.scipy.org/install.html

data set used http://archive.ics.uci.edu/ml/datasets/Wine

## names

first the names need to be added, we can pull these attributes from the wine.names file. One diffrence here from the iris example is the class is now a number and its the first entry into the names array

```python
names = ['class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'phenols', 'Flavanoids', 'Nonflavanoid', 'Proanthocyanins', 'Color', 'Hue', 'diluted wines', 'Proline']
```

### Dimensions of Dataset
just like we did in the iris example from the above turtorial we can peek at the data a little bit:

```python
print(dataset.shape)
```
output:
```bash
(178, 14)
```


```python
print(dataset.head(20))
```
output:
```bash
    class  Alcohol  Malic acid   Ash  Alcalinity of ash  Magnesium  phenols  Flavanoids  Nonflavanoid  Proanthocyanins  Color   Hue  diluted wines  Proline
0       1    14.23        1.71  2.43               15.6        127     2.80        3.06          0.28             2.29   5.64  1.04           3.92     1065
1       1    13.20        1.78  2.14               11.2        100     2.65        2.76          0.26             1.28   4.38  1.05           3.40     1050
2       1    13.16        2.36  2.67               18.6        101     2.80        3.24          0.30             2.81   5.68  1.03           3.17     1185
3       1    14.37        1.95  2.50               16.8        113     3.85        3.49          0.24             2.18   7.80  0.86           3.45     1480
4       1    13.24        2.59  2.87               21.0        118     2.80        2.69          0.39             1.82   4.32  1.04           2.93      735
5       1    14.20        1.76  2.45               15.2        112     3.27        3.39          0.34             1.97   6.75  1.05           2.85     1450
6       1    14.39        1.87  2.45               14.6         96     2.50        2.52          0.30             1.98   5.25  1.02           3.58     1290
7       1    14.06        2.15  2.61               17.6        121     2.60        2.51          0.31             1.25   5.05  1.06           3.58     1295
8       1    14.83        1.64  2.17               14.0         97     2.80        2.98          0.29             1.98   5.20  1.08           2.85     1045
9       1    13.86        1.35  2.27               16.0         98     2.98        3.15          0.22             1.85   7.22  1.01           3.55     1045
10      1    14.10        2.16  2.30               18.0        105     2.95        3.32          0.22             2.38   5.75  1.25           3.17     1510
11      1    14.12        1.48  2.32               16.8         95     2.20        2.43          0.26             1.57   5.00  1.17           2.82     1280
12      1    13.75        1.73  2.41               16.0         89     2.60        2.76          0.29             1.81   5.60  1.15           2.90     1320
13      1    14.75        1.73  2.39               11.4         91     3.10        3.69          0.43             2.81   5.40  1.25           2.73     1150
14      1    14.38        1.87  2.38               12.0        102     3.30        3.64          0.29             2.96   7.50  1.20           3.00     1547
15      1    13.63        1.81  2.70               17.2        112     2.85        2.91          0.30             1.46   7.30  1.28           2.88     1310
16      1    14.30        1.92  2.72               20.0        120     2.80        3.14          0.33             1.97   6.20  1.07           2.65     1280
17      1    13.83        1.57  2.62               20.0        115     2.95        3.40          0.40             1.72   6.60  1.13           2.57     1130
18      1    14.19        1.59  2.48               16.5        108     3.30        3.93          0.32             1.86   8.70  1.23           2.82     1680
19      1    13.64        3.10  2.56               15.2        116     2.70        3.03          0.17             1.66   5.10  0.96           3.36      845
```
looks good!

