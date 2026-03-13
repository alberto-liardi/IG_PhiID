import matplotlib.pyplot as plt
import numpy as np
from gaussian_phiid import ig_phiid_gaussian
from discrete_phiid import ig_phiid_discrete
from discrete_utils import sample_from_discrete

if "__main__" == __name__:

    print("Simulate random discrete distribution and process...")
    prob = np.random.rand(2,2,2,2)
    prob /= prob.sum()
    data = sample_from_discrete(prob, n_samples=int(1e4))
    pt_phiid, av_phiid = ig_phiid_discrete(prob, verbose=False, 
                                    pointwise=True, data=data, as_dict=True)
    plt.figure(figsize=(16,5))
    plt.subplot(1,4,1)
    plt.plot(pt_phiid["rtr"][::100], label="rtr")
    plt.xlabel("Time (every 100 steps)")
    plt.ylabel("Pointwise PhiID (bits)")
    plt.legend()
    plt.subplot(1,4,2)
    plt.plot(pt_phiid["xta"][::100], label="xta")
    plt.xlabel("Time (every 100 steps)")
    plt.legend()
    plt.subplot(1,4,3)
    plt.plot(pt_phiid["yts"][::100], label="yts")
    plt.xlabel("Time (every 100 steps)")
    plt.legend()
    plt.subplot(1,4,4)
    plt.plot(pt_phiid["sts"][::100], label="sts")
    plt.xlabel("Time (every 100 steps)")
    plt.legend()
    
    print("Simulate random Gaussian process...")
    n = 4
    cov = np.random.rand(n,n)
    cov = cov @ cov.T /2
    d = np.sqrt(np.diag(cov))
    cov = cov / np.outer(d, d)
    data = np.random.multivariate_normal(np.zeros(cov.shape[0]), cov, size=10000).T
    pt_phiid, av_phiid = ig_phiid_gaussian(cov, n1=n//4, n2=n//4,verbose=False, pointwise=True, data=data, as_dict=True)

    plt.figure(figsize=(16,5))
    plt.subplot(1,4,1)
    plt.plot(pt_phiid["rtr"][::100], label="rtr")
    plt.xlabel("Time (every 100 steps)")
    plt.ylabel("Pointwise PhiID (bits)")
    plt.legend()
    plt.subplot(1,4,2)
    plt.plot(pt_phiid["xta"][::100], label="xta")
    plt.xlabel("Time (every 100 steps)")
    plt.legend()
    plt.subplot(1,4,3)
    plt.plot(pt_phiid["yts"][::100], label="yts")
    plt.xlabel("Time (every 100 steps)")
    plt.legend()
    plt.subplot(1,4,4)
    plt.plot(pt_phiid["sts"][::100], label="sts")
    plt.xlabel("Time (every 100 steps)")
    plt.legend()

    print("Simulate Gaussian VAR(1) process...")
    n = 2
    A = np.random.rand(n,n)/2
    V = np.eye(n)
    data = np.zeros((10000, n))
    for t in range(1, data.shape[0]):
        data[t] = A @ data[t-1] + np.random.multivariate_normal(np.zeros(n), V)
    data = data.T
    past_data = data[:,:-1]
    future_data = data[:,1:]
    cov = np.cov(np.vstack([past_data, future_data]))
    pt_phiid, av_phiid = ig_phiid_gaussian(cov, n1=n//2, n2=n//2,verbose=False, pointwise=True, data=np.vstack([past_data, future_data]), as_dict=True)

    plt.figure(figsize=(16,5))
    plt.subplot(1,4,1)
    plt.plot(pt_phiid["rtr"][::100], label="rtr") 
    plt.xlabel("Time (every 100 steps)")
    plt.ylabel("Pointwise PhiID (bits)")    
    plt.legend()
    plt.subplot(1,4,2)
    plt.plot(pt_phiid["xta"][::100], label="xta")
    plt.xlabel("Time (every 100 steps)")
    plt.legend()
    plt.subplot(1,4,3)
    plt.plot(pt_phiid["yts"][::100], label="yts")
    plt.xlabel("Time (every 100 steps)")
    plt.legend()
    plt.subplot(1,4,4)
    plt.plot(pt_phiid["sts"][::100], label="sts")
    plt.xlabel("Time (every 100 steps)")
    plt.legend()
    
    plt.show()