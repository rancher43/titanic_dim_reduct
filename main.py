
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition


def DR():

    tit = pd.read_csv('titanic.csv')
    df = tit[['Age', 'Pclass', 'Survived', 'Fare']]
    #print(df.isnull().sum())
    df = df.dropna(axis=0, how='any')
    pca = decomposition.PCA(n_components=2)
    pca.fit(df)
    new_df = pca.transform(df)
    plt.scatter(new_df[:,0], new_df[:,1])
    plt.show()

if __name__ == "__main__":
    DR()
