#Paul Sheehan - C11443788

import pandas as pd

#remove warning of chained indexing when replacing outliers in the dataframe
pd.options.mode.chained_assignment = None

#dataset = pd.read_csv('data/training_features.csv', low_memory=False)
dataset = pd.read_csv('data/training_features.csv', low_memory=False)
# fn= pd.read_csv('data/featureNames.txt', header=None)

#list of intger values which are used to index the continuous data
headers = dataset.dtypes.index
#Column names for our categorical report
contColumns = ['Feature', 'Count', 'Cardinality', 'Min',
               'Quart1', 'Mean','Median', 'Quart3', 'Max', 'Standard Dev']

#We do the exact same process for our continuous data
#empty list of statistics
count = []
cardinality = []
min = []
quart1 = []
mean = []
median = []
quart3 = []
max = []
stdDev = []


for i in headers:
    print("1 Column Processed")

    #1st quartile
    q1 = dataset[:][i].quantile(.25)
    #3rd quartile
    q3 = dataset[:][i].quantile(.75)

    #Turkey method to identify outliers
    innerFence = (((q3 - q1) * 1.5) - q1)
    outerFence = (((q3 - q1) * 3) + q3)

    catMean = round(dataset.iloc[:][i].mean())
    count.append(dataset[i].size)
    cardinality.append(len(dataset[:][i].unique()))
    min.append(dataset[:][i].min())
    mean.append(catMean)
    median.append(round(dataset[:][i].median()))
    max.append(dataset[:][i].max())
    quart1.append(q1)
    quart3.append(q3)
    stdDev.append(round(dataset[:][i].std()))

contReport = pd.DataFrame(index=headers, columns=contColumns)
print("Continuous data complete")
contReport.loc[:]['Feature'] = headers
contReport.loc[:]['Count'] = count
contReport.loc[:]['Cardinality'] = cardinality
contReport.loc[:]['Min'] = min
contReport.loc[:]['Quart1'] = quart1
contReport.loc[:]['Mean'] = mean
contReport.loc[:]['Median'] = median
contReport.loc[:]['Quart3'] = quart3
contReport.loc[:]['Max'] = max
contReport.loc[:]['Standard Dev'] = stdDev

contReport.to_csv('data/feature_analysis.csv', index=False)

print (contReport)


