# coding=utf-8
# Copyright (c) 2021 Kartik Chandra; see MIT license attached

'''
This patch adds a function jax.value_and_jacfwd, which is the
forward-mode version of jax.value_and_grad. It allows returning
the value of a function in addition to its derivative, so that
you don't need to evaluate the function twice to get both the
value and its derivative as you would using plain jax.jacfwd.
For example:

>>> import jax, value_and_jacfwd
>>> def g(x):
>>>     return (x ** 2).sum()
>>> dg = jax.value_and_jacfwd(g, has_aux=False)
>>> y, dg = dg(np.arange(3) * 1.)
>>> print(f'g(x) = {y}')
g(x) = 5.0
>>> print(f'dg(x) = {dg}')
dg(x) = [0. 2. 4.]

You can also export auxiliary values using the has_aux=True parameter,
again by analogy to jax.value_and_grad. For example:

>>> import jax, value_and_jacfwd
>>> def f(x):
>>>     return (x ** 2).sum(), x.sum()
>>> df = jax.value_and_jacfwd(f, has_aux=True)
>>> (y, aux), df = df(np.arange(3) * 1.)
>>> print(f'f(x) = {y}')
f(x) = 5.0
>>> print(f'df(x) = {df}')
df(x) = [0. 2. 4.]
>>> print(f'aux = {aux}')
aux = 3.0

This patch addresses the following Github issue:
  https://github.com/google/jax/pull/762
'''

from jax._src.api import (
    _check_callable,
    _ensure_index,
    _check_input_dtype_jacfwd,
    _check_output_dtype_jacfwd,
    _std_basis,
    _unravel_array_into_pytree,
    _dtype
)
from jax._src.api import *


def _jvp(fun: lu.WrappedFun, primals, tangents, has_aux=False):
  """Variant of jvp() that takes an lu.WrappedFun."""
  if (not isinstance(primals, (tuple, list)) or
      not isinstance(tangents, (tuple, list))):
    raise TypeError("primal and tangent arguments to jax.jvp must be tuples or lists; "
                    f"found {type(primals).__name__} and {type(tangents).__name__}.")

  ps_flat, tree_def = tree_flatten(primals)
  ts_flat, tree_def_2 = tree_flatten(tangents)
  if tree_def != tree_def_2:
    raise TypeError("primal and tangent arguments to jax.jvp must have the same tree "
                    f"structure; primals have tree structure {tree_def} whereas tangents have "
                    f"tree structure {tree_def_2}.")
  for p, t in safe_zip(ps_flat, ts_flat):
    if core.primal_dtype_to_tangent_dtype(_dtype(p)) != _dtype(t):
      raise TypeError("primal and tangent arguments to jax.jvp do not match; "
                      "dtypes must be equal, or in case of int/bool primal dtype "
                      "the tangent dtype must be float0."
                      f"Got primal dtype {_dtype(p)} and so expected tangent dtype "
                      f"{core.primal_dtype_to_tangent_dtype(_dtype(p))}, but got "
                      f"tangent dtype {_dtype(t)} instead.")
    if np.shape(p) != np.shape(t):
      raise ValueError("jvp called with different primal and tangent shapes;"
                       f"Got primal shape {np.shape(p)} and tangent shape as {np.shape(t)}")

  if has_aux:
    flat_fun, out_aux_trees = flatten_fun_nokwargs2(fun, tree_def)
    main, aux = ad.jvp(flat_fun, has_aux=True)
    out_primals, out_tangents = main.call_wrapped(ps_flat, ts_flat)
    aux = aux()
    out_tree, aux_tree = out_aux_trees()
    return (tree_unflatten(out_tree, out_primals),
            tree_unflatten(out_tree, out_tangents)), tree_unflatten(aux_tree, aux)
  else:
    flat_fun, out_tree = flatten_fun_nokwargs(fun, tree_def)
    out_primals, out_tangents = ad.jvp(flat_fun).call_wrapped(ps_flat, ts_flat)
    return (tree_unflatten(out_tree(), out_primals),
            tree_unflatten(out_tree(), out_tangents))


def value_and_jacfwd(fun: Callable, argnums: Union[int, Sequence[int]] = 0,
           holomorphic: bool = False, has_aux: bool = False) -> Callable:
  """
  [Constructed by analogy to value_and_grad -- see help(value_and_grad) for more.]
  """
  _check_callable(fun)
  argnums = _ensure_index(argnums)

  def jacfun(*args, **kwargs):
    f = lu.wrap_init(fun, kwargs)
    f_partial, dyn_args = argnums_partial(f, argnums, args) #require_static_args_hashable=False)
    tree_map(partial(_check_input_dtype_jacfwd, holomorphic), dyn_args)
    if has_aux:
        pushfwd = partial(_jvp, f_partial, dyn_args, has_aux=True)
        (y, jac), aux = vmap(pushfwd, out_axes=((None, -1), None))(_std_basis(dyn_args))
    else:
        pushfwd = partial(_jvp, f_partial, dyn_args)
        (y, jac) = vmap(pushfwd, out_axes=(None, -1))(_std_basis(dyn_args))
    tree_map(partial(_check_output_dtype_jacfwd, holomorphic), y)
    example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
    if has_aux:
        return (y, aux), tree_map(partial(_unravel_array_into_pytree, example_args, -1), jac)
    else:
        return y, tree_map(partial(_unravel_array_into_pytree, example_args, -1), jac)

  return jacfun
jax.value_and_jacfwd = value_and_jacfwd  # !!
