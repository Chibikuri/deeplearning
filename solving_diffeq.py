# ref https://becominghuman.ai/neural-networks-for-solving-differential-equations-fa230ac5e04c
# from scipy.special import sigmoid
from autograd import grad
from autograd.core import primitive
import matplotlib.pyplot as plt
import autograd.numpy as np


def neuralnet(W, x):
    '''
    W -> weight
    x -> feature
    '''
    A1 = sigmoid(np.dot(x, W[0]))
    return np.dot(A1, W[1])


def sigmoid_grad(x):
    return sigmoid(x)*(1-sigmoid(x))


def sigmoid(x):
    return 1/(1+np.exp(x))


def difneural(W, x, k=1):
    return np.dot(np.dot(W[1].T, W[0].T**k), sigmoid_grad(x))


def loss(W, x):
    loss_sum = 0.
    for i in x:
        out = neuralnet(W, i)[0][0]
        psy_t = 1. + i*out
        dout = difneural(W, i)[0][0]
        dpsy_t = out + i * dout
        func = f(i, psy_t)
        errsq = (dpsy_t - func)**2
        loss_sum += errsq
    return loss_sum


def alpha(x):
    return x+(1.+3.*(x**2))/(1.+x+(x**3))


def beta(x):
    return (x**3)+2.*x+(x**2)*((1.+3.*(x**2))/(1.+x+(x**3)))


def f(x, psi):
    return beta(x)-psi*alpha(x)


def psi_analytic(x):
    return (np.exp((-x**2)/2.))/(1.+x+(x**3))+(x**2)

if __name__ == '__main__':
    nx = 10
    dx = 1./nx
    x_space = np.linspace(0, 1, nx)
    y_space = psi_analytic(x_space)
    psi_fd = np.zeros_like(y_space)
    psi_fd[0] = 1.

    for i in range(1, len(x_space)):
        psi_fd[i] = psi_fd[i-1]+beta(x_space[i])*dx-psi_fd[i-1]*(x_space[i])*dx

    W = [np.random.randn(1, 10), np.random.randn(10, 1)]
    lmb = 0.001

    for i in range(1000):
        loss_grad = grad(loss)(W, x_space)
        W[0] = W[0] - lmb*loss_grad[0]
        W[1] = W[1] - lmb*loss_grad[1]

    print(loss(W, x_space))
    res = [1 + xi*neuralnet(W, xi)[0][0] for xi in x_space]
    print(W)
    plt.figure()
    plt.plot(x_space, y_space, label="y_sp")
    plt.plot(x_space, psi_fd, label="psi_fd")
    plt.plot(x_space, res, label="res")
    plt.legend()
    plt.show()
