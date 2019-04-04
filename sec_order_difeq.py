from autograd import grad
from autograd.core import primitive
import matplotlib.pyplot as plt
import autograd.numpy as np


def sigmoid(x):
    return 1./(1. + np.exp(-x))


def neuralnet(W, x):
    A = sigmoid(np.dot(x, W[0]))
    return np.dot(A, W[1])


def neuralnet_x(x):
    A = sigmoid(np.dot(x, W[0]))
    return np.dot(A, W[1])


def psi_trial(xi, out):
    return xi+xi**2*out


def loss(W, x):
    loss_sum = 0
    psi_grad = grad(psi_trial)
    psi_grad2 = grad(psi_grad)

    for xi in x:
        out = neuralnet(W, xi)[0][0]
        out_d = grad(neuralnet_x)(xi)
        out_d2 = grad(grad(neuralnet_x))(xi)

        psi_t = psi_trial(xi, out)
        grad_trial = psi_grad(xi, out)
        grad2_trial = psi_grad2(xi, out)

        func = f(xi, psi_t, grad_trial)

        errsq = (grad2_trial-func)**2
        loss_sum += errsq
    return loss_sum


def f(x, psi, dpsi):
    return -1./5.*np.exp(-x/5.)*np.cos(x)-1./5.*dpsi-psi


def psi_analytic(x):
    return np.exp(-x/5.)*np.sin(x)


if __name__ == '__main__':
    nx = 10
    dx = 1./nx

    x_space = np.linspace(0, 2, nx)
    y_space = psi_analytic(x_space)
    W = [np.random.randn(1, 10), np.random.randn(10, 1)]
    lmb = 0.001

    for i in range(100):
        loss_grad = grad(loss)(W, x_space)

        W[0] = W[0] - lmb*loss_grad[0]
        W[1] = W[1] - lmb*loss_grad[1]

    print(loss(W, x_space))
    res = [psi_trial(xi, neuralnet(W, xi)[0][0]) for xi in x_space]

    plt.figure()
    plt.plot(x_space, y_space, label="y_space")
    plt.plot(x_space, res, label="res")
    plt.legend()
    plt.show()
