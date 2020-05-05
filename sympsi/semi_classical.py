"""
Semiclassical equations of motion
"""

__all__ = [
    'generate_eqm',
    'eqm_to_semi_classical',
    'sc_eqm_to_ode',
    'sc_ode_to_matrix'
    ]

from sympy import (Add, Mul, Pow, exp, Symbol, I, pi, simplify, oo, 
                   Function, Derivative, Eq, 
                   Matrix, MatMul, linear_eq_to_matrix)

from sympy.physics.quantum.operatorordering import normal_ordered_form

from sympsi.expectation import Expectation
from sympsi.qutility import (extract_all_operators,
                             operator_order, operator_sort_by_order,
                             drop_terms_containing, operator_master_equation)

debug = False

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
    A = Matrix([op_func_map[op] for op in ops])
    subs = [(op_func_map[op], Symbol(op_func_map[op].name)) for op in ops]
    eqns = [sc_ode[op].rhs.subs(subs) for op in ops]
    M, C = linear_eq_to_matrix(eqns, list(zip(*subs))[1])
    A_eq = Eq(-Derivative(A, t), Add(-C, MatMul(M, A), evaluate=False),
              evaluate=False)
    return A_eq, A, M, -C
