import glob
import os
import numpy as np
import matplotlib
import sys
import matplotlib.pyplot as plt
import scipy.stats as sp
from sklearn import svm

coeff_val = 1.96

def get_coff_range(train_set_filepath):
    curr_path = os.path.join(train_set_filepath,'label')
    curr_files = os.listdir(curr_path)
    diff_x = []
    diff_y = []
    coeff_range = []
    bbox_position = []
    mu_x = []
    mu_y = []
    var_x = []
    var_y = []
    diff_mu = []
    test_cnt = 0
    for full_path in curr_files:
        f = open(os.path.join(curr_path,full_path))
        bbox_list = f.read()
        bbox_list = bbox_list.split('\n')
        length = len(bbox_list)
        if len(bbox_position) == 0:
            bbox_position = [[] for _ in range(length)]        
        for l in range(1,len(bbox_list)):
            prev_rect = bbox_list[l - 1].split(',')[1:5]
            curr_rect = bbox_list[l].split(',')[1:5]

           
            if len(prev_rect) > 0 and len(curr_rect) > 0 and len(bbox_position) > l:
                bbox_position[l - 1].append([int(prev_rect[0]),int(prev_rect[1])])
                x = np.abs(int(prev_rect[0]) - int(curr_rect[0]))
                y = np.abs(int(prev_rect[1]) - int(curr_rect[1]))

                if y >= 2:
                    diff_mu.append([x,y])
            
    for o in range(0,len(bbox_position)):
        for l in range(0,len(bbox_position[o]) - 1):

            mu_x.append(np.mean(bbox_position[o][0]))
            var_x.append(np.var(bbox_position[o][0]))
            mu_y.append(np.mean(bbox_position[o][1]))
            var_y.append(np.var(bbox_position[o][1]))

    for o in range(0,len(mu_x)):
        coeff_range.append([[mu_x[o] - (coeff_val * var_x[o] / np.sqrt(len(mu_x) - 1)),mu_x[o] + (coeff_val * var_x[o] / np.sqrt(len(mu_x) - 1))]
                            ,[mu_y[o] - (coeff_val * var_y[o] / np.sqrt(len(mu_y)- 1)),mu_y[o] + (coeff_val * var_y[o] / np.sqrt(len(mu_y)- 1))]]) # stddev
    
    return diff_mu,coeff_range

def get_classifier():
    traing_path = r'E:\개인 프로젝트\텐서플로\goturn\test_set'
    train_data,coeff_range = get_coff_range(traing_path)
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(train_data)

    return clf,coeff_range

def is_diff_outlier(diff,ranges):
    return True

def is_position_outlier(rect,ranges):     
    return True


traing_path = r'E:\개인 프로젝트\텐서플로\goturn\test_set'
train_data,coeff_range = get_coff_range(traing_path)
train_data = np.array(train_data)
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(train_data)
pred_test = clf.predict(train_data)
#print(coeff_range)
#pos_range,diff_range = get_coff_range(traing_path)
#print(pos_range)
#print(diff_range)

xx, yy = np.meshgrid(np.linspace(-1,10, 500), np.linspace(-1, 10, 500))
# plot the line, the points, and the nearest vectors to the plane
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("outlier Detection")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred')


s = 40
b1 = plt.scatter(train_data[:, 0], train_data[:, 1], c='white', s=s, edgecolors='k')
plt.axis('tight')
plt.xlim((-1, 10))
plt.ylim((-1,10))
plt.show()
'''
# fit the model
a = np.array([2,4,6])
target = np.array([8,3,0])
var = np.sqrt(np.var(a))
mu = np.mean(a)
print(var)
print(mu)
x = np.linspace(0, 5, 10, endpoint=False)
y = multivariate_normal(mean=mu, cov=var).pdf(target)
y_cdf = multivariate_normal(mean=mu, cov=var).cdf(target)

cof = [mu - (1.96 * var / np.sqrt(len(a) - 1)),mu +(1.96 * var / np.sqrt(len(a)- 1))]

print(a)
print(y)
print(cof)

'''
'''
mu = (x / len(x_diff))
y_mu = (y / len(y_diff))
stddev_sum = 0
stddev_sum_y = 0
for l in range(0,len(x_diff)):
    stddev_sum += (x_diff[l] - mu)**2
    stddev_sum_y += (y_diff[l] - y_mu)**2

var = np.var(x_diff)
mean = np.mean(x_diff)
stddev = stddev_sum / (len(x_diff) - 1)
stddev_y = stddev_sum_y / (len(y_diff) - 1)
p =  1/(np.sqrt(stddev) * np.sqrt(2 * np.pi)) * np.exp( - (x_diff[0] - mu)**2 / (2 * stddev))

z = (9 - mu) / np.sqrt(stddev)

x_diff = np.array(x_diff).T
print(x_diff)
y_diff = np.array(y_diff).T
print(y_diff)
#rv = sp.stats.multivariate_normal([mu,y_mu], [stddev,stddev_y])
print(np.cov(x_diff,y_diff))
'''

'''
X =  tf.placeholder(tf.float32, shape=[None, 1, 1,1] ,name="prev_frame")
lap = tf.distributions.Laplace(1.0,0.067)
result = lap.prob(X)
with tf.Session() as sess:
    x = [[[[3.0]]]]
    sess.run(tf.global_variables_initializer())
    print(sess.run(result,feed_dict = {X:x}))

fig, ax = plt.subplots(1, 1)
x = np.linspace(laplace.ppf(0.01,1,0.067),laplace.ppf(0.99,1,0.067), 100)
ax.plot(x, laplace.pdf(x,1,0.067),'r-', lw=5, alpha=0.6, label='laplace pdf')
r = laplace.rvs(size=1000)
ax.legend(loc='best', frameon=False)
plt.show()
'''
'''
train_set_filepath = r'E:\개인 프로젝트\텐서플로\goturn\test_set'
files = os.listdir(path=train_set_filepath)
curr_path = os.path.join(train_set_filepath,'label')
curr_files = os.listdir(curr_path)
f = open(os.path.join(curr_path,curr_files[0]))
bbox_list = f.read()

bbox_list = bbox_list.split('\n')
x_diff = []
y_diff = []
x = 0
y = 0
for l in range(1,len(bbox_list)):
    arr1 = bbox_list[l - 1].split(',')[1:5]
    arr2 = bbox_list[l].split(',')[1:5]
    x_diff.append([np.abs(int(arr1[0]) - int(arr2[0])),np.abs(int(arr1[1]) - int(arr2[1]))])
    #y_diff.append([int(arr1[1]) - int(arr2[1])])
    #x += np.abs(int(arr1[0]) - int(arr2[0]))
    #y += np.abs(int(arr1[1]) - int(arr2[1]))

'''

