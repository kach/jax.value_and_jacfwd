value_and_jacfwd.py
Copyright (c) 2021 Kartik Chandra; see MIT license attached

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
