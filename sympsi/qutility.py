"""
Utitility functions for working with operators
"""

__all__ = [
    'latex_align',
    'collect_by_nc',
    'collect_by_order',
    'extract_operators',
    'extract_operator_products',
    'extract_all_operators',
    'operator_order',
    'operator_sort_by_order',
    'drop_higher_order_terms',
    'drop_terms_containing',
    'drop_c_number_terms',
    'lindblad_dissipator',
    'master_equation',
    'operator_lindblad_dissipator',
    'operator_master_equation',
    ]

from collections import defaultdict
from sympy import (Basic, Add, Mul, Pow, exp, I, S, factor,
                   diff, Function, Eq, latex)

from sympy.physics.quantum import Operator, Commutator, Dagger

debug = False

# -----------------------------------------------------------------------------
# IPython notebook related functions
#
from IPython.display import Latex

def latex_align(data, env="align*", delim=None, breaks=None): # or set col_delim="&" for auto align
    if isinstance(data, list):
        delim = " " if delim is None else delim
        body = " \\\\\n".join([delim.join([latex(col) for col in row])
                               for row in data])

    if isinstance(data, Basic):
        args = Add.make_args(data)
        delim = "& " if delim is None else delim
        breaks = range(4, len(args)-3, 4) if breaks is None else breaks
        breaks = zip([0] + list(breaks), list(breaks) + [len(args)])
        def fmt_line(i, j):
            line = latex(Add(*args[i:j]))
            if i != 0 and latex(Add(*args[i:j]))[0] != '-':
                line = "+" + line
            return delim + line
        body = "\\\\\n".join([fmt_line(i,j) for i,j in breaks])

    return Latex("\\begin{{{0}}}\n{1}\n\\end{{{0}}}".format(env, body))

# -----------------------------------------------------------------------------
# Utility functions for manipulating operator expressions
#


def collect_by_nc(expr, evaluate=True):
    collected, disliked = defaultdict(list), S.Zero

    for arg in Add.make_args(expr):
        c, nc = arg.args_cnc()
        if nc: collected[Mul(*nc)].append(Mul(*c))
        else: disliked += Mul(*c)

    collected = {k: Add(*v) for k, v in collected.items()}
    if disliked is not S.Zero:
        collected[S.One] = disliked
    if evaluate:
        return Add(*[key*val for key, val in collected.items()])
    else:
        return collected

def collect_by_order(expr, evaluate=True):
    """
    return dict d such that expr == Add(*[d[n] for n in d])
    where Expr d[n] contains only terms with operator order n
    """
    args = Add.make_args(expr)
    d = {}
    for arg in args:
        n = operator_order(arg)
        if n in d: d[n] += arg
        else: d[n] = arg

    d = {n : factor(collect_by_nc(arg)) for n, arg in d.items()}
    if evaluate:
        return Add(*[arg for arg in d.values()], evaluate=False)
    else:
        return d


def extract_operators(e, independent=False):
    return list(set([e for e in preorder_traversal(O)
                     if isinstance(e, Operator)]))


def extract_operator_products(expr):
    """
    Return a list of unique quantum operator products in the expression e.
    """
    if isinstance(expr, Operator):
        return [expr]

    elif isinstance(expr, Add):
        return list(set([op for arg in expr.args
                         for op in extract_operator_products(arg)]))

    c, nc = expr.args_cnc()
    return [Mul(*nc)] if nc else []

def extract_operator_subexprs(expr):
    args = Mul.make_args(expr)
    return [Mul(*args[i:j]) for i in range(len(args) + 1)
            for j in range(i + 1, len(args) + 1)]

def extract_all_operators(expr):
    """
    Extract all unique operators in the normal ordered for of a given
    operator expression, including composite operators. The resulting list
    of operators are sorted in increasing order.
    """
    ops = extract_operator_products(expr)

    return list(set([op_sub for op in ops
                     for op_sub in extract_operator_subexprs(op)]))

def operator_order(op):
    if isinstance(op, Operator):
        return 1

    if isinstance(op, Mul):
        return sum([operator_order(arg) for arg in op.args])

    if isinstance(op, Pow):
        return operator_order(op.base) * op.exp

    return 0

def operator_sort_by_order(ops):
    return sorted(sorted(ops, key=repr), key=operator_order)

def drop_higher_order_terms(e, order):
    """
    Drop any terms with operator order greater than order arg
    """
    if isinstance(e, Add):
        e = Add(*(arg for arg in e.args if operator_order(arg) <= order))
    return e

def drop_terms_containing(e, e_drops):
    """
    Drop terms contaning factors in the list e_drops
    """
    if isinstance(e, Add):
        # fix this
        #e = Add(*(arg for arg in e.args if not any([e_drop in arg.args
        #                                            for e_drop in e_drops])))

        new_args = []

        for term in e.args:

            keep = True
            for e_drop in e_drops:
                if e_drop in term.args:
                    keep = False

                if isinstance(e_drop, Mul):
                    if all([(f in term.args) for f in e_drop.args]):
                        keep = False

            if keep:
        #        new_args.append(arg)
                new_args.append(term)
        e = Add(*new_args)
        #e = Add(*(arg.subs({key: 0 for key in e_drops}) for arg in e.args))

    return e

def drop_c_number_terms(e):
    """
    Drop commuting terms from the expression e
    """
    if isinstance(e, Add):
        return Add(*(arg for arg in e.args if not arg.is_commutative))

    return e

# ----------------------------------------------------------------------------
# Master equations and adjoint master equations
#
def lindblad_dissipator(a, rho):
    """
    Lindblad dissipator
    """
    return (a*rho*Dagger(a) - rho*Dagger(a)*a/2 - Dagger(a)*a*rho/2)

def operator_lindblad_dissipator(a, rho):
    """
    Lindblad operator dissipator
    """
    return (Dagger(a)*rho*a - rho*Dagger(a)*a/2 - Dagger(a)*a*rho/2)

def master_equation(rho_t, t, H, a_ops, use_eq=True):
    """
    Lindblad master equation
    """
    lhs = diff(rho_t, t)
    rhs = (-I*Commutator(H, rho_t) +
           sum([lindblad_dissipator(a, rho_t) for a in a_ops]))

    return Eq(lhs, rhs) if use_eq else (lhs, rhs)

def operator_master_equation(op_t, t, H, a_ops, use_eq=True):
    """
    Adjoint master equation
    """
    lhs = diff(op_t, t)
    rhs = (I*Commutator(H, op_t) +
           sum([operator_lindblad_dissipator(a, op_t) for a in a_ops]))

    return Eq(lhs, rhs) if use_eq else (lhs, rhs)
