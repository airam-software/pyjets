# Train the classifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from pnj2f.pnj_traintest import pnj_traintest

classifiers = [
    GaussianNB(),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0),
    KNeighborsClassifier(n_neighbors=3),
]

classifiers = [DecisionTreeClassifier(max_depth=5)]

mainpath = 'TBD'

training_method_vec = ['compare', 'classify']
pathh5 = mainpath + 'datos/h5remapped/'
pathclassifier = './results/traintest/'
pathplot = './plots/traintest/'

rg_threshold = 0.75
pnj_traintest(pathh5, pathclassifier, '010', training_method_vec[1], classifiers, rg_threshold, './',
              DecisionTreeClassifier(max_depth=5), pathplot)
