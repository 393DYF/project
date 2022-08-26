import numpy as np
import pandas as pd
#I=np.identity(d)
#cov_matrix = sigma+(1-sigma)*I
#S_{t+1} = phi*S_{t} + e
S1 = np.ones((1,1))
for i in range(5003):
    mean1 = [0]
    cov1 = [[1]]
    e1 = np.random.multivariate_normal(mean1, cov1)
    phi=0.2
    S1 = S1*phi+e1  # alternate form
    test1 = pd.DataFrame(S1)
    test1.to_csv("C:/Users/lenovo/Desktop/linkage/synthetic_data_release/VAR_try1_0.2.csv", mode='a+', index=None, header=None)
S2 = np.ones((1,2))
for i in range(5003):
    mean2 = [0,0]
    cov2 = [[1, 0.8],
           [0.8, 1]]
    e2 = np.random.multivariate_normal(mean2, cov2)
    phi=0.2
    S2 = S2*phi+e2  # alternate form
    test2 = pd.DataFrame(S2)
    test2.to_csv("C:/Users/lenovo/Desktop/linkage/synthetic_data_release/VAR_try2_0.2.csv", mode='a+', index=None, header=None)
S3 = np.ones((1,3))
for i in range(5003):
    mean3 = [0, 0, 0]
    cov3 = [[1, 0.8, 0.8],
           [0.8, 1, 0.8],
           [0.8, 0.8, 1]]
    e3 = np.random.multivariate_normal(mean3, cov3)
    phi=0.2
    S3 = S3*phi+e3  # alternate form
    test3 = pd.DataFrame(S3)
    test3.to_csv("C:/Users/lenovo/Desktop/linkage/synthetic_data_release/VAR_try3_0.2.csv", mode='a+', index=None, header=None)
S4 = np.ones((1,1))
for i in range(5003):
    mean4 = [0]
    cov4 = [[1]]
    e4 = np.random.multivariate_normal(mean4, cov4)
    phi=0.5
    S4 = S4*phi+e4  # alternate form
    test4 = pd.DataFrame(S4)
    test4.to_csv("C:/Users/lenovo/Desktop/linkage/synthetic_data_release/VAR_try1_0.5.csv", mode='a+', index=None, header=None)
S5 = np.ones((1,2))
for i in range(5003):
    mean5 = [0,0]
    cov5 = [[1, 0.8],
           [0.8, 1]]
    e5 = np.random.multivariate_normal(mean5, cov5)
    phi=0.5
    S5 = S5*phi+e5  # alternate form
    test5 = pd.DataFrame(S5)
    test5.to_csv("C:/Users/lenovo/Desktop/linkage/synthetic_data_release/VAR_try2_0.5.csv", mode='a+', index=None, header=None)
S6 = np.ones((1,3))
for i in range(5003):
    mean6 = [0, 0, 0]
    cov6 = [[1, 0.8, 0.8],
           [0.8, 1, 0.8],
           [0.8, 0.8, 1]]
    e6 = np.random.multivariate_normal(mean6, cov6)
    phi=0.5
    S6 = S6*phi+e6  # alternate form
    test6 = pd.DataFrame(S6)
    test6.to_csv("C:/Users/lenovo/Desktop/linkage/synthetic_data_release/VAR_try3_0.5.csv", mode='a+', index=None, header=None)
S7 = np.ones((1,2))
for i in range(5003):
    mean7 = [0,0]
    cov7 = [[1, 0.2],
           [0.2, 1]]
    e7 = np.random.multivariate_normal(mean7, cov7)
    phi=0.8
    S7 = S7*phi+e7  # alternate form
    test7 = pd.DataFrame(S7)
    test7.to_csv("C:/Users/lenovo/Desktop/linkage/synthetic_data_release/VAR_try2_sigma=0.2.csv", mode='a+', index=None, header=None)
S8 = np.ones((1,3))
for i in range(5003):
    mean8 = [0, 0, 0]
    cov8 = [[1, 0.2, 0.2],
           [0.2, 1, 0.2],
           [0.2, 0.2, 1]]
    e8 = np.random.multivariate_normal(mean8, cov8)
    phi=0.8
    S8 = S8*phi+e8  # alternate form
    test8 = pd.DataFrame(S8)
    test8.to_csv("C:/Users/lenovo/Desktop/linkage/synthetic_data_release/VAR_try3_sigma=0.2.csv", mode='a+', index=None, header=None)
S9 = np.ones((1,2))
for i in range(5003):
    mean9 = [0,0]
    cov9 = [[1, 0.5],
           [0.5, 1]]
    e9 = np.random.multivariate_normal(mean9, cov9)
    phi=0.8
    S9 = S9*phi+e9  # alternate form
    test9 = pd.DataFrame(S9)
    test9.to_csv("C:/Users/lenovo/Desktop/linkage/synthetic_data_release/VAR_try2_sigma=0.5.csv", mode='a+', index=None, header=None)
S10 = np.ones((1,3))
for i in range(5003):
    mean10 = [0, 0, 0]
    cov10 = [[1, 0.5, 0.5],
           [0.5, 1, 0.5],
           [0.5, 0.5, 1]]
    e10 = np.random.multivariate_normal(mean10, cov10)
    phi=0.8
    S10 = S10*phi+e10  # alternate form
    test10 = pd.DataFrame(S10)
    test10.to_csv("C:/Users/lenovo/Desktop/linkage/synthetic_data_release/VAR_try3_sigma=0.5.csv", mode='a+', index=None, header=None)