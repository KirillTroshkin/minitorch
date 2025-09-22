"""
Implementation of the autodifferentiation Functions for Tensor.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend
from .tensor_data import index_to_position

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)

# Constructors


class Function:

    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:

        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:

        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:

        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:

        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:

        return grad_output.f.neg_map(grad_output)


class Inv(Function):

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:

        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:

        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:

        ctx.save_for_backward(t1, t2)
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:

        t1, t2 = ctx.saved_values

        if t1.shape == grad_output.shape:
            t1_grad = grad_output
        else:
            t1_grad = Add._reduce_grad(grad_output, t1.shape)

        if t2.shape == grad_output.shape:
            t2_grad = grad_output
        else:
            t2_grad = Add._reduce_grad(grad_output, t2.shape)

        return t1_grad, t2_grad

    @staticmethod
    def _reduce_grad(grad_output: Tensor, target_shape: UserShape) -> Tensor:

        if grad_output.shape == target_shape:
            return grad_output

        grad_data = [0.0] * int(operators.prod(target_shape))

        for i in range(grad_output.size):
            big_index = [0] * len(grad_output.shape)
            remaining = i
            for j in range(len(grad_output.shape) - 1, -1, -1):
                big_index[j] = remaining % grad_output.shape[j]
                remaining = remaining // grad_output.shape[j]

            big_shape_list = list(grad_output.shape)
            shape_list = list(target_shape)
            while len(shape_list) < len(big_shape_list):
                shape_list.insert(0, 1)

            out_index = [0] * len(shape_list)
            for k in range(len(shape_list)):
                if k < len(big_index):
                    if shape_list[k] == 1:
                        out_index[k] = 0
                    else:
                        out_index[k] = big_index[k]
                else:
                    out_index[k] = 0

            out_index = out_index[: len(target_shape)]

            out_idx = 0
            for j in range(len(out_index)):
                out_idx = out_idx * target_shape[j] + out_index[j]

            grad_data[out_idx] += grad_output._tensor._storage[i]

        return minitorch.Tensor.make(
            grad_data, target_shape, backend=grad_output.backend
        )


class Mul(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:

        ctx.save_for_backward(a, b)
        return a.f.mul_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:

        (a, b) = ctx.saved_values
        grad_a = grad_output.f.mul_zip(grad_output, b)
        grad_b = grad_output.f.mul_zip(grad_output, a)

        if a.shape == grad_a.shape:
            grad_a_final = grad_a
        else:
            grad_a_final = Add._reduce_grad(grad_a, a.shape)

        if b.shape == grad_b.shape:
            grad_b_final = grad_b
        else:
            grad_b_final = Add._reduce_grad(grad_b, b.shape)

        return grad_a_final, grad_b_final


class Sigmoid(Function):

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:

        ctx.save_for_backward(t1)
        return t1.f.sigmoid_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:

        (t1,) = ctx.saved_values
        return grad_output.f.sigmoid_back_zip(t1, grad_output)


class ReLU(Function):

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:

        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:

        (t1,) = ctx.saved_values
        return grad_output.f.relu_back_zip(t1, grad_output)


class Log(Function):

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:

        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:

        (t1,) = ctx.saved_values
        return grad_output.f.log_back_zip(t1, grad_output)


class Exp(Function):

    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:

        ctx.save_for_backward(t1)
        return t1.f.exp_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:

        (t1,) = ctx.saved_values
        return grad_output.f.exp_back_zip(t1, grad_output)


class Sum(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:

        ctx.save_for_backward(a.shape, dim)
        return a.f.add_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:

        a_shape, dim = ctx.saved_values
        dim_val = int(dim.item())

        if grad_output.size == 1:
            grad_value = grad_output.item()
            grad_data = [grad_value] * int(operators.prod(a_shape))
        else:
            grad_data = [0.0] * int(operators.prod(a_shape))
            out_shape = list(a_shape)
            out_shape.pop(dim_val)

            for i in range(int(operators.prod(a_shape))):
                a_index = [0] * len(a_shape)
                remaining = i
                for j in range(len(a_shape) - 1, -1, -1):
                    a_index[j] = remaining % a_shape[j]
                    remaining = remaining // a_shape[j]

                out_index = a_index[:dim_val] + a_index[dim_val + 1 :]

                out_idx = 0
                for j in range(len(out_index)):
                    out_idx = out_idx * out_shape[j] + out_index[j]

                if out_idx < grad_output.size:
                    grad_data[i] = grad_output._tensor._storage[out_idx]

        grad_tensor = minitorch.Tensor.make(
            grad_data, a_shape, backend=grad_output.backend
        )
        return grad_tensor, 0.0


class All(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:

        ctx.save_for_backward(a.shape, dim)
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:

        a_shape, dim = ctx.saved_values
        return grad_output, 0.0


class LT(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:

        ctx.save_for_backward(a, b)
        return a.f.lt_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:

        (a, b) = ctx.saved_values

        if a.shape == grad_output.shape:
            a_grad = grad_output.zeros(a.shape)
        else:
            reduced = Add._reduce_grad(grad_output, a.shape)
            a_grad = reduced.zeros(a.shape)

        if b.shape == grad_output.shape:
            b_grad = grad_output.zeros(b.shape)
        else:
            reduced = Add._reduce_grad(grad_output, b.shape)
            b_grad = reduced.zeros(b.shape)

        return a_grad, b_grad


class EQ(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:

        ctx.save_for_backward(a, b)
        return a.f.eq_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:

        (a, b) = ctx.saved_values

        if a.shape == grad_output.shape:
            a_grad = grad_output.zeros(a.shape)
        else:
            reduced = Add._reduce_grad(grad_output, a.shape)
            a_grad = reduced.zeros(a.shape)

        if b.shape == grad_output.shape:
            b_grad = grad_output.zeros(b.shape)
        else:
            reduced = Add._reduce_grad(grad_output, b.shape)
            b_grad = reduced.zeros(b.shape)

        return a_grad, b_grad


class GT(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:

        ctx.save_for_backward(a, b)
        return a.f.gt_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:

        (a, b) = ctx.saved_values

        if a.shape == grad_output.shape:
            a_grad = grad_output.zeros(a.shape)
        else:
            reduced = Add._reduce_grad(grad_output, a.shape)
            a_grad = reduced.zeros(a.shape)

        if b.shape == grad_output.shape:
            b_grad = grad_output.zeros(b.shape)
        else:
            reduced = Add._reduce_grad(grad_output, b.shape)
            b_grad = reduced.zeros(b.shape)

        return a_grad, b_grad


class IsClose(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:

        ctx.save_for_backward(a, b)
        return a.f.is_close_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:

        (a, b) = ctx.saved_values
        return grad_output.zeros(a.shape), grad_output.zeros(b.shape)


class Permute(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:

        ctx.save_for_backward(order)
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:

        (order,) = ctx.saved_values
        order_list = [int(order[i].item()) for i in range(order.size)]
        inv_order = [0] * len(order_list)
        for i, j in enumerate(order_list):
            inv_order[j] = i
        return grad_output.permute(*inv_order), 0.0


class View(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:

        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:

        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:

        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:

        return grad_output


class MatMul(Function):

    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:

        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:

        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:

            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )

# Helpers for Constructing tensors


def zeros(

    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """
    Produce a zero tensor of size `shape`.

    Args:
        shape : shape of tensor
        backend : tensor backend
        requires_grad : whether the tensor requires gradients

    Returns:
        new tensor
    """
    tensor = minitorch.Tensor.make(
        [0] * int(operators.prod(shape)), shape, backend=backend
    )
    if requires_grad:
        tensor.requires_grad_(True)
    return tensor


def rand(

    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """
    Produce a random tensor of size `shape`.

    Args:
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(

    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """
    Produce a tensor with data ls and shape `shape`.

    Args:
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
        new tensor
    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(

    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """
    Produce a tensor with data and shape from ls

    Args:
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """

    def shape(ls: Any) -> List[int]:

        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:

        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)

# Gradient check for tensors


def grad_central_difference(

    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-3, ind: UserIndex
) -> float:
    x = vals[arg]
    up = zeros(x.shape, requires_grad=True)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:

    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )


class AddConstant(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:

        return a.f.add_zip(a, a.f.constant_map(a))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:

        return grad_output


class Square(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:

        ctx.save_for_backward(a)
        return a.f.mul_zip(a, a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:

        (a,) = ctx.saved_values
        return grad_output.f.mul_zip(grad_output, a.f.constant_2_map(a))


class Cube(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:

        ctx.save_for_backward(a)
        return a.f.mul_zip(a.f.mul_zip(a, a), a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:

        (a,) = ctx.saved_values
        return grad_output.f.mul_zip(grad_output, a.f.constant_3_map(a))


class SubConstant(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:

        return a.f.add_zip(a, a.f.constant_neg1_map(a))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:

        return grad_output


class MultConstant(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:

        ctx.save_for_backward(a)
        return a.f.mul_zip(a, a.f.constant_5_map(a))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:

        (a,) = ctx.saved_values
        return grad_output.f.mul_zip(grad_output, a.f.constant_5_map(a))


class Div(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:

        ctx.save_for_backward(a)
        return a.f.div_zip(a, a.f.constant_5_map(a))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:

        (a,) = ctx.saved_values
        return grad_output.f.div_zip(grad_output, a.f.constant_5_map(a))


class Sig(Function):

    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:

        ctx.save_for_backward(a)
        return a.f.sigmoid_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:

        (a,) = ctx.saved_values
        return grad_output.f.sigmoid_back_zip(a, grad_output)
