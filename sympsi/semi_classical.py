"""
Utitility functions for working with operator transformations in
sympsi.
"""

__all__ = [
    'generate_eqm',
    'eqm_to_semi_classical',
    'sc_eqm_to_ode',
    'sc_ode_to_matrix'
    ]

from collections import namedtuple
from sympy import (Add, Mul, Pow, exp, Symbol, symbols,
                   I, pi, simplify, oo, 
                   diff, Function, Derivative, Eq, 
                   Matrix, MatMul)

from sympy.physics.quantum import Operator, Commutator, Dagger
from sympy.physics.quantum.operatorordering import normal_ordered_form

from sympsi.expectation import Expectation
from sympsi.qutility import (operator_order, operator_sort_by_order,
                             drop_terms_containing, operator_master_equation)

debug = False

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

# -----------------------------------------------------------------------------
# Semiclassical equations of motion
#

def generate_eqm(H, c_ops, t, independent=True, max_order=2,
                 discard_unresolved=True):
    """
    Generate a set of semiclassical equations of motion from a Hamiltonian
    and set of collapse operators. Equations of motion for all operators that
    are included in either the Hamiltonian or the list of collapse operators
    will be generated, as well as any operators that are included in the
    equations of motion for the orignal operators. If the system of equations
    for the averages of operators does not close, the highest order operators
    will be truncated and removed from the equations.
    """
    op_eqm = {}

    #get_ops = lambda expr: extract_operator_products(
    get_ops = lambda expr: extract_all_operators(
        normal_ordered_form(expr, independent=independent).doit().expand())

    ops = get_ops(H + sum(c_ops))

    if debug:
        print("Hamiltonian operators: ", ops)

    while ops:
        order, idx = min((val, idx) for (idx, val)
                         in enumerate([operator_order(op) for op in ops]))
        if order > max_order:
            print("Warning: system did not close with max_order =", max_order)
            break

        op = ops.pop(idx)
        lhs, rhs = operator_master_equation(op, t, H, c_ops, use_eq=False)

        op_eqm[op] = normal_ordered_form(rhs.doit(
            independent=independent).expand(), independent=independent)

        new_ops = get_ops(op_eqm[op])

        for new_op in new_ops:
            if new_op not in op_eqm and new_op not in ops:
                if debug:
                    print(new_op, "not included, adding")
                ops.append(new_op)

    unresolved_ops = ops
    if debug:
        print("unresolved ops: ", unresolved_ops)

    if discard_unresolved:
        for op, eqm in op_eqm.items():
            op_eqm[op] = drop_terms_containing(eqm, unresolved_ops)

    # in python 3.6+ dictionaries are sorted
    op_eqm = {op: op_eqm[op] for op in operator_sort_by_order(op_eqm.keys())}
    return op_eqm, unresolved_ops

def _sceqm_factor_op(uop, ops):
    if uop in ops: return [uop]
    args = []
    for a in Mul.make_args(uop):
        args += [a.base for n in range(a.exp)] if isinstance(a, Pow) else [a]
    if len(args) < 2: return None # nothing to factor

    # first try to partition
    for i in range(1, len(args)):
        if Mul(*(args[:i])) in ops and Mul(*(args[i:])) in ops:
            return [Mul(*(args[:i])), Mul(*(args[i:]))]

    # then do recursive search
    for i in range(1, len(args)):
        head, tail = Mul(*(args[:i])), Mul(*(args[i:]))
        head_fact = _sceqm_factor_op(head, ops)
        tail_fact = _sceqm_factor_op(tail, ops)
        cands = []
        if head in ops and tail_fact is not None:
            cands += [[head] + tail_fact]
        if tail in ops and head_fact is not None:
            cands += [tail_fact + [head]]
        if head_fact is not None and tail_fact is not None:
            cands += [head_fact + tail_fact]
        if cands:
            _, idx = min((len(c), i) for (i, c) in enumerate(cands))
            return cands[idx]

    #for i in range(1, len(uop.args)):
    #    for j in range(i+1, len(uop.args)+1):
    #        if Mul(*(args[i:j])) in ops:
    #            first = _sceqm_factor_op(Mul(*(args[:i])), ops)
    #            if first is not None:
    #                last = _sceqm_factor_op(Mul(*(args[j:])), ops)
    #                if last is not None:
    #                    return first + [Mul(*(args[i:j]))] + last

    #if isinstance(uop, Pow):
    #    op_pows = [1] if uop.base in ops else []
    #    op_pows += [op.exp for op in ops
    #                if isinstance(op, Pow) and op.base == uop.base]
    #    # do the naive thing so this doesn't turn into solving subset sum
    #    for n in sorted(op_pows):
    #        if uop.exp % n == 0:
    #            return [Pow(uop.base, n)] * (uop.exp // n)

    #    raise Exception("Failed to find factorization of %r" % uop)
    return None


def eqm_to_semi_classical(op_eqm, unresolved_ops, partition=False):
    op_factorization = {}
    sc_eqm = {}
    for op, eqm in op_eqm.items():
        sc_eqm[op] = Expectation(eqm).expand(expectation=True)

        for uop in unresolved_ops:
            sub_ops = _sceqm_factor_op(uop, op_eqm.keys())
            if sub_ops is None:
                raise Exception("Failed to find factorization of %r" % uop)
            factored_expt = Mul(*(Expectation(o) for o in sub_ops))
            op_factorization[Expectation(uop)] = factored_expt
            sc_eqm[op] = sc_eqm[op].subs(Expectation(uop), factored_expt)

    return sc_eqm, op_factorization

def sc_eqm_to_ode(sc_eqm, t, op_label_map=None):
    op_func_map = {}
    for op in sc_eqm:
        label = repr(op) if op_label_map is None else op_label_map[op]
        op_func_map[op] = Function(label)(t)

    if debug:
        print("Operator -> Function map: ", op_func_map)

    op_subs = {Expectation(op) : op_func_map[op] for op in sc_eqm}

    sc_ode = {}
    for op, eqm in sc_eqm.items():
        sc_ode[op] = Eq(Derivative(Expectation(op).subs(op_subs), t),
                        eqm.subs(op_subs))

    return sc_ode, op_func_map

def sc_ode_to_matrix(sc_ode, op_func_map, t):
    """
    Convert a set of semiclassical equations of motion to matrix form.
    """
    ops = operator_sort_by_order(sc_ode.keys())
    As = [op_func_map[op] for op in ops]
    A = Matrix(As)
    A_sub_0 = {A: 0 for A in As}
    B = Matrix([[sc_ode[op].rhs.subs(A_sub_0)] for op in ops])

    def make_row(row_op):
        row_A = op_func_map[row_op]
        A_sub_0 = {A: 0 for A in As if A != row_A}
        return [((sc_ode[col_op].rhs - B[col]).subs(A_sub_0)/row_A).expand()
                for col, col_op in enumerate(ops)]
    
    M = Matrix([make_row(row_op) for row_op in ops]).T

    return Eq(-Derivative(A, t), Add(B, MatMul(M, A), evaluate=False),
              evaluate=False), A, M, B
