# EE2211_Calculator
A python script containing useful functions for the module EE2211 at NUS

## How to use
Run the script and use the functions 
OR
Place the script inside your working folder and import it

## Available Functions
```
mlf(x):

Input: string describing an array x in MATLAB format
output: np array x

```
```
addbias(x):

Input: np array x
output: np array (1|x)
```
```
onehot(x):

input: np vector x
output: one-hot np array x_ohe

```
```
sign(x):

input: np vector x
output: signum-ed np vector x_sig

```
```
sign(x):

input: np vector x
output: signum-ed np vector x_sig

```
```
poly_expand(X,deg):

Input: np array x, int degree 
Output: np array polynomial expansion of x. Bias is included.

```
```
regress(x,y, verbose = False):

Input: np array x, y, bool verbose -> displays matrix type if true. Default: False. This solves Y = Xw
Output: np array w

```
```
ridge_regress(X,y,ld, verbose = False):

Input: np array x, y, float ld (lambda), bool verbose -> displays matrix type if true. Default: False
Output: np array w, with ridge regression

```
```
predict(x,w):

Input: np array x, np vector/array w . This predicts Y using Y=Xw
Output: np vector/array y_predicted

```
```
impurity(splits, tpe = 'g'):

Input: np array of splits. If you have n nodes, splits should have n subarrays, each with its elements inside.
Input: tpe -> type of impurity. 'g' for gini, 'e' for entropy, 'm' for misclass rate
Output: float of impurity

```
```
kmeans_train(X,k,c = None):

Input X: training data
Input k: number of nodes
Input c: centroids to initialise if any
Output: trained centroids

```
```
predict_kmeans(X,c):

Input X: training data
Input c: centroids 
Output: predicted class based on centroids

```
