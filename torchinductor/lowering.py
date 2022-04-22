import functools
import itertools
from typing import List

import sympy
import torch
import torch.fx

from . import ir
from .ir import ExpandView
from .ir import PermuteView
from .ir import Pointwise
from .ir import Reduction
from .ir import SqueezeView
from .ir import TensorBox
from .ir import View
from .virtualized import graph
from .virtualized import ops

lowerings = {}
aten = torch.ops.aten


def _register_lowering(aten_fn, decomp_fn, broadcast, type_promote):
    """
    Add a lowering to lowerings dict

    Arguments:
        aten_fn: torch.ops.aten.* fn we are lowering
        decomp_fn: alternate implementation on our IR
        broadcast: True to apply broadcasting to tensor inputs
        type_promote: True to apply type promotion to tensor inputs
    """

    @functools.wraps(decomp_fn)
    def wrapped(*args, **kwargs):
        args = list(args)
        # Only look at args that are Tensors
        indices = [i for i, x in enumerate(args) if isinstance(x, TensorBox)]
        # kwargs tensors not supported yet
        assert not any(isinstance(x, TensorBox) for x in kwargs.values())

        if type_promote and indices:
            dtype = functools.reduce(
                torch.promote_types, [args[i].get_dtype() for i in indices]
            )
            for i in indices:
                args[i] = to_dtype(args[i], dtype)

        if broadcast and indices:
            for i, x in zip(indices, broadcast_tensors(*[args[i] for i in indices])):
                args[i] = x

        return decomp_fn(*args, **kwargs)

    lowerings[aten_fn] = wrapped
    return wrapped


def register_lowering(aten_fn, broadcast=False, type_promote=True):
    """
    Shim to support decorator syntax.
    """
    return functools.partial(
        _register_lowering, aten_fn, broadcast=broadcast, type_promote=type_promote
    )


def broadcast_symbolic_shapes(a, b):
    """
    Broadcasting logic based on symbolic shapes.

    We give the shapes 0 and 1 concrete values, while all other shapes
    are symbolic sympy formulas.
    """
    output = []
    for a, b in itertools.zip_longest(
        reversed(a), reversed(b), fillvalue=sympy.Integer(1)
    ):
        if b == 1:
            output.append(a)
        elif a == 1:
            output.append(b)
        else:
            graph.sizevars.guard_equals(a, b)
            if len(sympy.expand(b).free_symbols) < len(sympy.expand(a).free_symbols):
                output.append(b)  # prefer shorter formula
            else:
                output.append(a)
    return tuple(reversed(output))


def make_pointwise(fn, override_dtype=None, override_device=None):
    def inner(*inputs: List[TensorBox]):
        loaders = [x.make_loader() for x in inputs]
        return Pointwise.create(
            device=override_device or inputs[0].get_device(),
            dtype=override_dtype or inputs[0].get_dtype(),
            inner_fn=lambda index: fn(*[load(index) for load in loaders]),
            ranges=inputs[0].get_size(),
        )

    return inner


def to_dtype(x: TensorBox, dtype: torch.dtype):
    if x.get_dtype() == dtype:
        return x

    def _to_dtype(x):
        return ops.to_dtype(x, dtype)

    return make_pointwise(_to_dtype, override_dtype=dtype)(x)


def register_pointwise(aten_fn, name=None, broadcast=True, type_promote=True):
    """A pointwise function that maps ops.{name} to inputs"""
    name = name or aten_fn.__name__

    @register_lowering(aten_fn, broadcast=broadcast, type_promote=type_promote)
    @make_pointwise
    def fn(*args, **kwargs):
        return getattr(ops, name)(*args, **kwargs)

    return fn


@register_lowering(aten.broadcast_tensors, broadcast=False, type_promote=False)
def broadcast_tensors(*inputs):
    target = functools.reduce(
        broadcast_symbolic_shapes, [x.get_size() for x in inputs], ()
    )
    outputs = []
    for x in inputs:
        sizes = x.get_size()
        if len(sizes) != len(target) or any(
            ((a == 1 and b != 1) or (a != 1 and b == 1)) for a, b in zip(sizes, target)
        ):
            x = expand(x, target)
        outputs.append(x)
    return outputs


@register_lowering(aten.detach)
def detach(x):
    assert isinstance(x, TensorBox)
    return x  # AOT autograd handles this for us


@register_lowering(aten.squeeze)
def squeeze(x, dim=None):
    assert isinstance(x, TensorBox)
    if dim is None:
        return TensorBox(SqueezeView.create(x.data))

    dim = _validate_dim(x, dim, 0)
    new_shape = list(x.get_size())
    removed = new_shape.pop(dim)
    assert removed == 1, removed
    return view(x, new_shape)


@register_lowering(aten.expand)
def expand(x, sizes):
    assert isinstance(x, TensorBox)
    assert isinstance(sizes, (list, tuple))
    return TensorBox(ExpandView.create(x.data, tuple(sizes)))


@register_lowering(aten.view)
def view(x, sizes):
    assert isinstance(x, TensorBox)
    assert isinstance(sizes, (list, tuple))
    return TensorBox(View.create(x.data, sizes))


@register_lowering(aten.permute)
def permute(x, dims):
    assert isinstance(x, TensorBox)
    assert isinstance(dims, (list, tuple))
    return TensorBox(PermuteView.create(x.data, tuple(dims)))


@register_lowering(aten.slice)
def slice_(x, dim, start, end, step=1):
    assert isinstance(x, TensorBox)
    dim = _validate_dim(x, dim, 0)
    return TensorBox(ir.SliceView.create(x.data, dim, start, end, step))


@register_lowering(aten.select)
def select(x, dim, idx):
    idx = View.handle_negative_index(idx, x.get_size()[dim])
    return squeeze(slice_(x, dim, idx, idx + 1), dim)


@register_lowering(aten.split)
def split(x, sizes, dim):
    dim = _validate_dim(x, dim, 0)
    x_size = graph.sizevars.guard_static_shape(x.get_size()[dim])
    if isinstance(sizes, int):
        sizes = [sizes] * ((x_size + sizes - 1) // sizes)
    result = []
    start = 0
    for size in sizes:
        end = start + size
        result.append(slice_(x, dim, start, end))
        start = end
    return result


@register_lowering(aten.unsqueeze)
def unsqueeze(x, dim):
    dim = _validate_dim(x, dim, 1)
    new_shape = list(x.get_size())
    new_shape.insert(dim, sympy.Integer(1))
    return view(x, new_shape)


def _validate_dim(x, dim, offset):
    assert isinstance(dim, int)
    ndim = len(x.get_size())
    if dim < 0:
        dim += ndim + offset
    assert 0 <= dim < ndim + offset
    return dim


@register_lowering(aten.mm)
def mm(a: TensorBox, b: TensorBox):
    return TensorBox.create(ir.MatrixMultiply.create(a, b))


@register_lowering(aten.bmm)
def bmm(a: TensorBox, b: TensorBox):
    return TensorBox.create(ir.BatchMatrixMultiply.create(a, b))


@register_lowering(aten.convolution)
def convolution(
    x: TensorBox,
    weight: TensorBox,
    bias: TensorBox,
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    transposed: bool,
    output_padding: List[int],
    groups: int,
):
    return TensorBox.create(ir.Convolution.create(
        x,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        ))


@register_lowering(torch.arange)
def arange(start, end=None, step=1, *, dtype=None, device=None):
    if end is None:
        end = start
        start = 0

    assert isinstance(start, int)
    assert isinstance(end, int)
    assert isinstance(step, int)

    dtype = dtype or torch.get_default_dtype()
    length = (end - start) // step
    start = sympy.Integer(start)
    step = sympy.Integer(step)

    return Pointwise.create(
        device=device or torch.tensor(0.0).device,
        dtype=dtype,
        inner_fn=lambda index: ops.index_expr(step * index[0] + start, dtype),
        ranges=[sympy.Integer(length)],
    )


@register_lowering(aten.gather, type_promote=False)
def gather(x, dim, index):
    assert isinstance(x, TensorBox)
    assert isinstance(dim, int)
    assert "int" in str(index.get_dtype())
    assert 0 <= dim < len(x.get_size())

    x_loader = x.make_loader()
    index_loader = index.make_loader()

    def fn(idx):
        idx = list(idx)
        var_index = index_loader(idx)
        idx[dim] = sympy.Symbol(str(var_index))
        return x_loader(idx)

    return Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=fn,
        ranges=index.get_size(),
    )


def make_reduction(reduction_type: str):
    def inner(x, axis=None, keepdims=False):
        size = x.get_size()
        if axis is None:
            axis = range(len(size))
        else:
            assert reduction_type == "sum", "min/max need to find index"
        axis = list(axis)
        for i in range(len(axis)):
            if axis[i] < 0:
                axis[i] += len(size)
            assert 0 <= axis[i] < len(size)
        assert len(set(axis)) == len(axis), "reduction axis not unique"
        axis = set(axis)

        kept_sizes = []
        kept_idx = []
        reduced_sizes = []
        reduced_idx = []
        for i in range(len(size)):
            if i in axis:
                reduced_idx.append(i)
                reduced_sizes.append(size[i])
            else:
                kept_idx.append(i)
                kept_sizes.append(size[i])

        def loader(index, reduction_index):
            assert len(reduction_index) == len(reduced_idx)
            if keepdims:
                assert len(index) == len(size)
                assert all(index[i] == 0 for i in reduced_idx)
                index = [index[i] for i in kept_idx]
            assert len(index) == len(kept_idx)
            new_index = [None] * (len(index) + len(reduction_index))
            for idx, var in itertools.chain(
                zip(kept_idx, index), zip(reduced_idx, reduction_index)
            ):
                new_index[idx] = var
            return inner_loader(new_index)

        if keepdims:
            new_size = list(size)
            for i in reduced_idx:
                new_size[i] = sympy.Integer(1)
        else:
            new_size = kept_sizes

        inner_loader = x.make_loader()
        result = Reduction.create(
            device=x.get_device(),
            dtype=x.get_dtype(),
            inner_fn=loader,
            ranges=new_size,
            reduction_ranges=reduced_sizes,
            reduction_type=reduction_type,
        )
        result.realize()
        return result

    return inner


sum = register_lowering(aten.sum)(make_reduction("sum"))
register_lowering(aten.max)(make_reduction("max"))
register_lowering(aten.min)(make_reduction("min"))
register_pointwise(aten.add)
register_pointwise(aten.div)
register_pointwise(aten.abs)
register_pointwise(aten.sub)
register_pointwise(aten.log)
register_pointwise(aten.mul)
register_pointwise(aten.maximum)
register_pointwise(aten.minimum)
register_pointwise(aten.sigmoid)
register_pointwise(aten.silu)
relu = register_pointwise(aten.relu)
exp = register_pointwise(aten.exp)