import torch
import torch.nn as nn
from torch._functorch.apis import vmap, grad
from typing import Callable
import argparse
from torch._functorch.functional_call import functional_call
import torchopt # https://github.com/metaopt/torchopt
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
    def __init__(self, inputs: int = 1, layers: int = 1, neurons: int = 5):
        super().__init__()
        self.inputs = inputs
        self.layers = layers
        self.neurons = neurons

        layer_list = []

        layer_list.append(nn.Linear(self.inputs, self.neurons))
        
        for _ in range(layers): # keep adding layers per the argument
            layer_list.extend([nn.Linear(neurons, neurons), nn.Tanh()])

        # output
        layer_list.append(nn.Linear(neurons, 1))

        self.net = nn.Sequential(*layer_list)

    def forward(self, x: torch.Tensor):
        return self.net(x.reshape(-1, 1)).squeeze()
        
R = 1.0  # rate of maximum population growth parameterizing the equation
X_BOUNDARY = 0.0  # boundary condition coordinate
F_BOUNDARY = 0.5  # boundary condition value

def make_loss_fn(f: Callable, dfdx: Callable) -> Callable:

    def loss_fn(params: torch.Tensor, x: torch.Tensor):

        # interior loss
        f_value = f(x, params)
        interior = dfdx(x, params) - R * f_value * (1 - f_value)

        # boundary loss
        x0 = X_BOUNDARY
        f0 = F_BOUNDARY
        x_boundary = torch.tensor([x0])
        f_boundary = torch.tensor([f0])
        boundary = f(x_boundary, params) - f_boundary

        loss = nn.MSELoss()
        loss_value = loss(interior, torch.zeros_like(interior)) + loss(
            boundary, torch.zeros_like(boundary)
        )

        return loss_value

    return loss_fn


def make_forward_fn(model: nn.Module, derivative_order: int = 1,) -> list[Callable]:

    # notice that `functional_call` supports batched input by default
    # thus there is not need to call vmap on it, as it's instead the case
    # for the derivative calls
    def f(x: torch.Tensor, params: dict[str, torch.nn.Parameter] | tuple[torch.nn.Parameter, ...]) -> torch.Tensor:
        
        # the functional optimizer works with parameters represented as a tuple instead
        # of the dictionary form required by the `functional_call` API 
        # here we perform the conversion from tuple to dictionary
        if isinstance(params, tuple):
            params_dict = tuple_to_dict_parameters(model, params)
        else:
            params_dict = params

        return functional_call(model, params_dict, (x, ))

    fns = []
    fns.append(f)

    dfunc = f
    for _ in range(derivative_order):

        # first compute the derivative function
        dfunc = grad(dfunc)

        # then use vmap to support batching
        dfunc_vmap = vmap(dfunc, in_dims=(0, None))

        fns.append(dfunc_vmap)

    return fns


def tuple_to_dict_parameters(
        model: nn.Module, params: tuple[torch.nn.Parameter, ...]
) -> OrderedDict[str, torch.nn.Parameter]:
    keys = list(dict(model.named_parameters()).keys())
    values = list(params)
    return OrderedDict(({k:v for k,v in zip(keys, values)}))


if __name__ == "__main__":
    # choose the configuration for the training loop
    batch_size = 30  # number of colocation points to sample in the domain
    num_iter = 100  # maximum number of iterations
    learning_rate = 1e-1  # learning rate
    domain = (-5.0, 5.0)  # ;ogistic equation domain

    # choose optimizer with functional API using functorch
    optimizer = torchopt.FuncOptimizer(torchopt.adam(lr=learning_rate))

    model = Net()
    funcs = make_forward_fn(model, derivative_order=1)

    f = funcs[0]
    dfdx = funcs[1]
    loss_fn = make_loss_fn(f, dfdx)

    # initial parameters randomly initialized
    params = tuple(model.parameters())

    # train the model
    loss_evolution = []
    for i in range(num_iter):

        # sample points in the domain randomly for each epoch
        x = torch.FloatTensor(batch_size).uniform_(domain[0], domain[1])

        # compute the loss with the current parameters
        loss = loss_fn(params, x)

        # update the parameters with functional optimizer
        params = optimizer.step(loss, params)

        print(f"Iteration {i} with loss {float(loss)}")
        loss_evolution.append(float(loss))

    # plot solution on the given domain
    x_eval = torch.linspace(domain[0], domain[1], steps=100).reshape(-1, 1)
    f_eval = f(x_eval, params)
    analytical_sol_fn = lambda x: 1.0 / (1.0 + (1.0/F_BOUNDARY - 1.0) * np.exp(-R * x))
    x_eval_np = x_eval.detach().numpy()
    x_sample_np = torch.FloatTensor(batch_size).uniform_(domain[0], domain[1]).detach().numpy()

    fig, ax = plt.subplots()

    ax.scatter(x_sample_np, analytical_sol_fn(x_sample_np), color="red", label="Sample training points")
    ax.plot(x_eval_np, f_eval.detach().numpy(), label="PINN final solution")
    ax.plot(
        x_eval_np,
        analytical_sol_fn(x_eval_np),
        label=f"Analytic solution",
        color="green",
        alpha=0.75,
    )
    ax.set(title="Logistic equation solved with NNs", xlabel="t", ylabel="f(t)")
    ax.legend()

    fig, ax = plt.subplots()
    ax.semilogy(loss_evolution)
    ax.set(title="Loss evolution", xlabel="# epochs", ylabel="Loss")
    ax.legend()

    plt.show()

