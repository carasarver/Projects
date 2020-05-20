# Explores k means clustering through the classic Iris dataset
# Model predictions not in accordance to standard matches -- needs resolving

from sklearn import datasets
from sklearn.cluster import KMeans
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

iris = datasets.load_iris()                 # iris included in sklearn
df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
df['species'] = iris['target']
df['species name'] = iris['target_names'][ iris['target'] ]
print(df.sample(5))

# kmeans clustering model with 3 clusters

model = KMeans(n_clusters = 3)
model = model.fit(df.iloc[:, 0:5])

# species prediction

df['predicted species'] = model.predict(df.iloc[:, 0:5])                # many rows showing incorrect clusters
print(df.sample(10))

# find the accuracy of prediction

count = 0
for i in range(len(df)):
    if df['predicted species'][i] == df['species'][i]:
        count += 1

accuracy = count / len(df.index) * 100
print("The accuracy of the model is ", accuracy)                # seems very low