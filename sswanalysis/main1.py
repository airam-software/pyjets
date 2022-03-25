# Train the classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from ssw.pnj_traintest import pnj_traintest

classifiers = [
    GaussianNB(),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0),
    KNeighborsClassifier(n_neighbors=3),
]

opt = ['compare', 'classify']
pathh5 = 'TBD'
pathclassifier = './results/traintest/'
pathplot = './plots/traintest/'

pnj_traintest(pathh5, pathclassifier, '010', opt[1],
              [DecisionTreeClassifier(max_depth=5)], './',
              DecisionTreeClassifier(max_depth=5), pathplot)
