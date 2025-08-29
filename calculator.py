import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans

def mlf(x):
    '''
    Input: string describing an array x in MATLAB format
    output: np array x
    '''
    arr = [[i] for i in x.split(";")]
    arr = [list(filter(None,i[0].split(" "))) for i in arr]
    arr = np.array([[float(i) for i in j] for j in arr])
    return arr

def addbias(x):
    '''
    Input: np array x
    output: np array (1|x)
    '''
    x = np.hstack((np.ones((np.shape(x)[0],1)),x))
    return x

def onehot(x):
    '''
    input: np vector x
    output: one-hot np array x_ohe
    '''
    scaler = OneHotEncoder()
    scaler.fit(x)
    x_ohe = scaler.transform(x).toarray()
    return x_ohe

def sign(x):
    '''
    input: np vector x
    output: signum-ed np vector x_sig
    '''
    return np.sign(x)


def poly_expand(X,deg):
    '''
    Input: np array x, int degree 
    Output: np array polynomial expansion of x. Bias is included.
    '''
    if len(np.array(X).shape) == 1:
        X = np.array([X])
        model = PolynomialFeatures(deg)
        P = model.fit_transform(X)
        P = P.flatten()
        return P
    else:
        X = np.array(X)
        model = PolynomialFeatures(deg)
        P = model.fit_transform(X)
        return P

def regress(x,y, verbose = False):
    '''
    Input: np array x, y, bool verbose -> displays matrix type if true. Default: False
    Output: np array w
    '''
    x = np.array(x)
    y = np.array(y)
    if (np.shape(x)[0]>np.shape(x)[1]):
        return (np.linalg.inv(x.T @ x) @ x.T @ y) if not verbose else (np.linalg.inv(x.T @ x) @ x.T @ y,'Tall')
        
    elif (np.shape(x)[0]<np.shape(x)[1]):
        return (x.T @ np.linalg.inv(x @x.T) @ y) if not verbose else (x.T @ np.linalg.inv(x @x.T) @ y,'Wide')
    else:
        return (np.linalg.inv(x) @ y) if not verbose else (np.linalg.inv(x) @ y,"Square")

def ridge_regress(X,y,ld, verbose = False):
    '''
    Input: np array x, y, float ld, bool verbose -> displays matrix type if true. Default: False
    Output: np array w, with ridge regression
    '''
    X = np.array(X)
    y = np.array(y)
    
    if X.shape[0] > X.shape[1]:
        w = (np.linalg.inv((X.T @ X)+ (np.identity(X.shape[1])*ld)) @ X.T @ y)
        form = "primal form" 
    elif X.shape[0] < X.shape[1]:
        w = (X.T @ np.linalg.inv((X @ X.T) + ld *np.identity(X.shape[0])) @ y)
        form = "dual form"
    else:
        form = "X is square. Reverting to primal form"
        w = (np.linalg.inv((X.T @ X)+ (np.identity(X.shape[1])*ld)) @ X.T @ y)
    return (w,form) if verbose else w

def predict(x,w):
    '''
    Input: np array x, np vector/array w
    Output: np vector/array y_predicted
    '''
    x = np.array(x)
    w = np.array(w)
    return x @ w

def impurity(splits, tpe = 'g'):
    '''
    Input: np array of splits. If you have n nodes, splits should have n subarrays, each with its elements inside.
    Inout: tpe -> type of impurity. 'g' for gini, 'e' for entropy, 'm' for misclass rate
    Output: float of impurity
    '''
    num_elements = len(np.concatenate(splits).ravel())
    count_list = [len(i) for i in splits]
    prob_list = []
    for node in splits:
        temp_list = []
        uniques = np.unique(node)
        for element in uniques:
            prob = node.count(element)
            temp_list.append(prob/len(node))
        prob_list.append(temp_list)
    
    total = 0
    
    if (tpe == 'g'):
        for i,node in enumerate(prob_list):
            gini = 1 - sum([p**2 for p in node])
            total += (count_list[i]/num_elements)*gini
    elif (tpe == 'e'):
        for i,node in enumerate(prob_list):
            entropy = -sum([p*np.log2(p) for p in node])
            total += (count_list[i]/num_elements)*entropy
    elif (tpe == 'm'):
        for i,node in enumerate(prob_list):
            misclass = 1 - max(node)
            total += (count_list[i]/num_elements)*misclass

    #possible to return an exception if tpe is neither 'g' nor 'e' nor 'm'
        
    return total

def kmeans_train(X,k,c = None):
    '''
    Input X: training data
    Input k: number of nodes
    Input c: centroids to initialise if any
    Output: trained centroids
    '''
    if (type(c) == list):
        km = KMeans(n_clusters=k, n_init=1, init = np.array(c)).fit(X)
    else:
        km = KMeans(n_clusters=k, n_init=1, init = c).fit(X)
        
    return km.cluster_centers_
        
def predict_kmeans(X,c):
    '''
    Input X: training data
    Input c: centroids 
    Output: predicted class based on centroids
    '''
    X = np.array(X)
    clusters = [[] for i in range(np.array(c).shape[0])]
    for x in X:
        closest_c = np.argmin(np.sqrt(np.sum((x-c)**2, axis=1)))
        clusters[closest_c].append(x)
    return clusters   
