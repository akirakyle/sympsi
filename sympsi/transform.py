"""
Unitary and Hamiltonian operator transformations
"""

__all__ = [
    'bch_special_closed_form',
    'unitary_transformation',
    'hamiltonian_transformation'
    ]

from sympy import (Add, Mul, Pow, exp, S, I, diff, simplify)
from sympy.core.basic import preorder_traversal
from sympy.physics.quantum import Operator, Commutator, Dagger

from sympsi.qutility import collect_by_nc, operator_order

debug = False

def bch_special_closed_form(X, Y, independent=False):
    """
    Return the exact solution to exp(X)*Y*exp(-X) in the special case that 
    [X, Y] = uX + vY + cI otherwise returns exp(X)*Y*exp(-X)
    
    See https://arxiv.org/abs/1501.02506v2 for derivation of this special-case 
    closed form to the Baker–Campbell–Hausdorff formula.
    """
    if not isinstance(Y, Operator):
        raise ValueError("Y must be an Operator")
    if debug: print("bch_special_closed_form()\nX =", X, "\nY =", Y)

    comm = Commutator(X, Y)
    while True: # this should be implemented in Commutator class
        expr = comm.expand(commutator=True)
        if comm == expr: break
        else: comm = expr

    comm = simplify(comm.doit(independent=independent)).expand()
    if debug: print("comm: ", comm)
    if comm == 0: return Y

    # this will fail for X or Y if they are Adds, need better collect
    collected = collect_by_nc(comm, evaluate=False)
    u = collected[X] if X in collected else S.Zero
    v = collected[Y] if Y in collected else S.Zero
    c = collected[S.One] if S.One in collected else S.Zero

    if debug: print("u: ", u, "v: ", v, "c: ", c)
    if simplify((u*X + v*Y + c - comm).expand()) == 0:
        e = Y + comm * ((exp(v) - S.One)/v) # Eq. 52 in above paper
        if v == 0: return Y + u*X + c # instead of NaN
        #else: return e.expand()
        else: return exp(v)*Y + (u*X + c)*(1-exp(v))/v
    else:
        print("warning: special closed form doesn't apply...")
        return exp(X)*Y*exp(-X)


def unitary_transformation(U, O, N=None, collect_operators=None,
                           independent=False, expansion_search=True):
    """
    Perform a unitary transformation

        O = U O U^\dagger


    Where U is of the form U = exp(A)
    and automatically try to identify series expansions in the resulting
    operator expression.
    """
    if not isinstance(U, exp):
        raise ValueError("U must be a unitary operator on the form U = exp(A)")

    A = U.exp
    if debug: print("unitary_transformation: using A = ", A)

    ops = list(set([e for e in preorder_traversal(O)
                    if isinstance(e, Operator)]))
    if debug: print("ops: ", ops)

    subs = {op: bch_special_closed_form(A, op, independent=independent)
            for op in ops}
    if debug: print("\n".join(["sub {}: {}".format(o,s) for o,s in subs.items()]))

    return O.subs(subs, simultaneous=True)


def hamiltonian_transformation(U, H, t=None, hbar=1, N=None,
                               collect_operators=None,
                               independent=False, expansion_search=True):
    """
    Apply an unitary basis transformation to the Hamiltonian H:

        H = U H U^\dagger + i hbar (d/dt U) U^\dagger

    Where U is of the form U = exp(A)
    """
    t_sym = [s for s in U.exp.free_symbols if str(s) == 't']
    if t_sym and not t:
        print("Warning: a symbol t was found in U but time kwarg not passed")

    H_st = unitary_transformation(U, H, N=N,
                                  collect_operators=collect_operators,
                                  independent=independent,
                                  expansion_search=expansion_search)
    H_td = - I * hbar * U* diff(Dagger(U), t) if t else 0
    return H_st + H_td

