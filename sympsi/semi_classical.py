"""
Utitility functions for working with operator transformations in
sympsi.
"""

__all__ = [
    'semi_classical_eqm',
    'semi_classical_eqm_matrix_form'
    ]

from collections import namedtuple
from sympy import (Add, Mul, Pow, exp, Symbol, symbols,
                   I, pi, simplify, oo, 
                   diff, Function, Derivative, Eq, 
                   Matrix, MatMul)

from sympy.physics.quantum import Operator, Commutator, Dagger
from sympy.physics.quantum.operatorordering import normal_ordered_form

from sympsi.qutility import (operator_order, drop_c_number_terms,
                             qsimplify, drop_terms_containing,
                             operator_master_equation)
from sympsi.expectation import Expectation

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

def operator_sort_by_order(ops):
    return sorted(sorted(ops, key=repr), key=operator_order)

# -----------------------------------------------------------------------------
# Semiclassical equations of motion
#

def _operator_to_func(e, op_func_map):

    if isinstance(e, Expectation):
        if e.expression in op_func_map:
            return op_func_map[e.expression]
        else:
            return e.expression

    if isinstance(e, Add):
        return Add(*(_operator_to_func(term, op_func_map) for term in e.args))

    if isinstance(e, Mul):
        return Mul(*(_operator_to_func(factor, op_func_map)
                     for factor in e.args))

    return e


def _sceqm_factor_op(op, ops):

    if isinstance(op, Pow):
        for n in range(1, op.exp):
            if Pow(op.base, op.exp - n) in ops and Pow(op.base, n) in ops:
                return op.base, Pow(op.base, op.exp - 1)

        raise Exception("Failed to find factorization of %r" % op)

    if isinstance(op, Mul):
        args = []
        for arg in op.args:
            if isinstance(arg, Pow):
                for n in range(arg.exp):
                    args.append(arg.base)
            else:
                args.append(arg)

        for n in range(1, len(op.args)):
            if Mul(*(args[:n])) in ops and Mul(*(args[n:])) in ops:
                return Mul(*(args[:n])), Mul(*(args[n:]))

        raise Exception("Failed to find factorization of %r" % op)

    return op.args[0], Mul(*(op.args[1:]))


def semi_classical_eqm(H, c_ops, t=Symbol("t", positive=True), max_order=2,
                       discard_unresolved=True, independent=True):
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
            print("Warning: system did not close with max_order=", max_order)
            break

        op = ops.pop(idx)
        lhs, rhs = operator_master_equation(op, t, H, c_ops, use_eq=False)

        op_eqm[op] = qsimplify(normal_ordered_form(rhs.doit(
            independent=independent).expand(), independent=independent))

        new_ops = get_ops(op_eqm[op])

        for new_op in new_ops:
            if new_op not in op_eqm.keys() and new_op not in ops:
                if debug:
                    print(new_op, "not included, adding")
                ops.append(new_op)

    ops_unresolved = ops
    if debug:
        print("unresolved ops: ", ops_unresolved)

    if discard_unresolved:
        for op, eqm in op_eqm.items():
            op_eqm[op] = drop_terms_containing(op_eqm[op], ops_unresolved)

    op_factorization = {}
    sc_eqm = {}
    for op, eqm in op_eqm.items():
        ops = get_ops(eqm)
        sc_eqm[op] = Expectation(eqm).expand(expectation=True)

        #for unresolved in ops_unresolved:
        #    sub_ops = _sceqm_factor_op(unresolved, op_eqm.keys())
        #    factored_expt = Mul(*(Expectation(o) for o in sub_ops))
        #    op_factorization[Expectation(unresolved)] = factored_expt
        #    sc_eqm[op] = sc_eqm[op].subs(Expectation(unresolved), factored_expt)

    op_func_map = {}
    op_index_map = {}
    for n, op in enumerate(op_eqm):
        op_func_map[op] = Function("A%d" % n)(t)
        op_index_map[op] = n

    if debug:
        print("Operator -> Function map: ", op_func_map)

    sc_ode = {}
    for op, eqm in sc_eqm.items():
        sc_ode[op] = Eq(Derivative(_operator_to_func(Expectation(op), op_func_map), t),
                        _operator_to_func(eqm, op_func_map))

    ops = operator_sort_by_order(op_func_map.keys())

    #for eqm in op_eqm:
    #    eqm_ops = extract_all_operators(op_eqm[op])

    SemiClassicalEQM = namedtuple('SemiClassicalEQM',
                                  ['operators',
                                   'operators_unresolved',
                                   'operator_eqs',
                                   'operator_factorization',
                                   'sc_eqs',
                                   'sc_ode',
                                   'op_func_map',
                                   'op_index_map',
                                   't'
                                   ])

    return SemiClassicalEQM(ops, ops_unresolved,
                            op_eqm, op_factorization, sc_eqm, sc_ode,
                            op_func_map, op_index_map, t)


def semi_classical_eqm_matrix_form(sc_eqm):
    """
    Convert a set of semiclassical equations of motion to matrix form.
    """
    ops = operator_sort_by_order(sc_eqm.op_func_map.keys())
    As = [sc_eqm.op_func_map[op] for op in ops]
    A = Matrix(As)
    b = Matrix([[sc_eqm.sc_ode[op].rhs.subs({A: 0 for A in As})] for op in ops])

    M = Matrix([[((sc_eqm.sc_ode[op1].rhs - b[m]).subs(
        {A: 0 for A in (set(As) - set([sc_eqm.op_func_map[op2]]))}) /
                  sc_eqm.op_func_map[op2]).expand()
                 for m, op1 in enumerate(ops)]
                for n, op2 in enumerate(ops)]).T

    return Eq(-Derivative(A, sc_eqm.t),
              Add(b, MatMul(M, A), evaluate=False), evaluate=False), A, M, b
