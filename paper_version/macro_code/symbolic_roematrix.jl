using SymPy, LinearAlgebra

# Suppose you have a 3×3 system with known eigenvalues/eigenvectors.
# I want to confirm that in the known case, the shape of the Roe matrix is the one computed by Antoine
# in order to adapt it to my case 

#Symbolic variables 
@syms  c2 lam u v w Δ λ1 λ2 λ3 σ1 σ2 σ3 z_p z_m


"""
The eigenvalues of the Roe matrix with a 1D velocity `u` are u, c₂u, c₂u + √Δ and c₂u - √Δ
with the discriminant Δ = λ + u²(c₂²-c₂).
Known eigenvector of the eigenvalues """
r1 = sympy.Matrix([1, λ1, z_p])  
r2 = sympy.Matrix([1, λ2, z_m])
r3 = sympy.Matrix([0, 0, 1]) 


P = hcat(r1, r2, r3)  # P = [r1 | r2 | r3]
P_inv = inv(P)

# In the diagonal matrix there are the absolute values of the eigenvalues to avoid sonic rarefaction wave
D = diagm(0 => [σ1, σ2, σ3])


println("\nCompute  P * D * P^-1 :")
A_expr = P * D * P_inv
A_expr_simpl = sympy.simplify(A_expr)
display(A_expr_simpl)


"""In this way I get exactly the same shape of matrix Antoine has in his code flux"""
J1 = sympy.Matrix([0, 1, 0])  
J2 = sympy.Matrix([-c2* u^2 + lam, 2 * c2 * u, 0])
J3 = sympy.Matrix([-c2 * u * v, c2 * v, c2 * u])

J = hcat(J1, J2, J3) 
J = J.transpose()

