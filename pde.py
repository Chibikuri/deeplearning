from autograd import grad, jacobian
from autograd.core import primitive
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import autograd.numpy as np


def analytic_solution(x):
    return (1/(np.exp(np.pi) - np.exp(-np.pi)))*np.sin(np.pi*x[0])*(np.exp(np.pi*x[1]) - np.exp(-np.pi*x[1]))


def f(x):
    return 0.


def sigmoid(x):
    return 1./(1.+np.exp(-x))


def neuralnet(W, x):
    A = sigmoid(np.dot(x, W[0]))
    return np.dot(A, W[1])


def neuralnetx(x):
    A = sigmoid(np.dot(x, W[0]))
    return np.dot(A, W[1])


def alpha(x):
    return x[1]*np.sin(np.pi*x[0])


def psi_trial(x, out):
    return alpha(x) + x[0]*(1-x[0])*x[1]*(1-x[1])*out


def loss(W, x, y):
    loss_sum = 0

    for xi in x:
        for yi in y:
            inputp = np.array([xi, yi])
            out = neuralnet(W, inputp)[0]
            out_jacobi = jacobian(neuralnetx)(inputp)
            out_hessian = jacobian(jacobian(neuralnetx))(inputp)

            psi_t = psi_trial(inputp, out)
            psit_jacobi = jacobian(psi_trial)(inputp, out)
            psit_hessian = jacobian(jacobian(psi_trial))(inputp, out)

            grad2x = psit_hessian[0][0]
            grad2y = psit_hessian[1][1]

            func = f(inputp)

            error_sqr = ((grad2x + grad2y) - func)**2
            loss_sum += error_sqr

    return loss_sum


if __name__ == '__main__':
    nx = 10
    ny = 10
    dx = 1./nx
    dy = 1./ny
    x_space = np.linspace(0, 1, nx)
    y_space = np.linspace(0, 1, ny)

    surface = np.zeros((ny, nx))
    for i, x in enumerate(x_space):
        for j, y in enumerate(y_space):
            surface[i][j] = analytic_solution([x, y])
    
    # # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # X, Y = np.meshgrid(x_space, y_space)
    # surf = ax.plot_surface(X, Y, surface, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=False)

    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # ax.set_zlim(0, 2)
    # ax.set_xlabel('$x$')
    # ax.set_ylabel('$y$')
    W = [np.random.randn(2, 10),np.random.randn(10, 1)]
    lmb = 0.001

    print(neuralnet(W, np.array([1, 1])))

    for i in range(100):
        loss_grad = grad(loss)(W, x_space, y_space)
        W[0] = W[0] - lmb*loss_grad[0]
        W[1] = W[1] - lmb*loss_grad[1]
    
    print(loss(W, x_space, y_space))
    surface2 = np.zeros((ny, nx))
    surface = np.zeros((ny, nx))

    for i, x in enumerate(x_space):
        for j, y in enumerate(y_space):
            surface[i][j] = analytic_solution([x, y])
    
    for i, x in enumerate(x_space):
        for j, y in enumerate(y_space):
            out_t = neuralnet(W, [x, y])[0]
            surface2[i][j] = psi_trial([x, y], out_t)

    print(surface[2])
    print(surface2[2])

    fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x_space, y_space)
surf = ax.plot_surface(X, Y, surface, rstride=1, cstride=1, cmap=cm.viridis,
        linewidth=0, antialiased=False)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 3)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$');


fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x_space, y_space)
surf = ax.plot_surface(X, Y, surface2, rstride=1, cstride=1, cmap=cm.viridis,
        linewidth=0, antialiased=False)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 3)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.show()