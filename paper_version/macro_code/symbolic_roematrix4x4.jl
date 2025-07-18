using SymPy, LinearAlgebra

"""
Code to compute the Roe matrix for my system. The eigenvalues of the Roe matrix 
with a 1D velocity `u` are u, c₂u, c₂u + √Δ and c₂u - √Δ with the discriminant Δ = λ + u²(c₂²-c₂)
and c₁ = 1, c₂ = K₂/K₁ and λ = K₂.
 The corresponding eigenvectors are (0,1,0,0), (0,0,0,1), (1,w,λ1,z_p), (1,w,λ2,z_m).
"""

# Symbolic variables 
@syms  c2 lam u v w Δ λ1 λ2 λ3 λ4 σ1 σ2 σ3 σ4 z_p z_m

#Known eigenvector of the eigenvalues 
r1 = sympy.Matrix([1, w, λ1, z_p])  
r2 = sympy.Matrix([1, w, λ2, z_m])
r3 = sympy.Matrix([0, 1, 0, 0]) 
r4 = sympy.Matrix([0, 0, 0, 1]) 


P = hcat(r1, r2, r3, r4)  # P = [r1 | r2 | r3 |r4 ]
P_inv = inv(P)

# In the diagonal matrix there are the absolute values of the eigenvalues to avoid sonic rarefaction wave
D = diagm(0 => [σ1, σ2, σ3, σ4])


println("\nCompute  P * D * P^-1 :")
A_expr = P * D * P_inv
A_expr_simpl = sympy.simplify(A_expr)
display(A_expr_simpl)
