"""
We need to solve the following conservative system:

∂ₜρ + c₁∇ₓ⋅(ρΩ) = 0,
∂ₜ(ρω) + c₁∇ₓ⋅(ρωΩ) = 0,
∂ₜ(ρΩ) + c₂∇ₓ⋅(ρΩ⊗Ω) + λ∇ₓρ = 0,

which can be written in vectorial form as: 
    ∂ₜ U + ∇ₓ⋅F(U) = 0.

To solve this system, we use a dimensional splitting method, 
which reduces the original 2D problem into a sequence of 1D problems. 
Each 1D problem is then solved using the Roe method.

The system consists of nonlinear conservation laws, so we need to linearize it. 
To do this, we approximate the Jacobian matrix ∇ₓ⋅F(U). 
The Roe method evaluates the Jacobian at an averaged state of U rather than at U itself.

In order to construct the Roe matrix, we start by computing 
the eigenvalues of the original Jacobian matrix (which depends on U). 
These eigenvalues are then evaluated at an averaged state of U.

In our case, the parameters are:
c₁ = 1, c₂ = K₂/K₁, λ = K₂

and the eigenvalues of the Roe matrix for a 1D velocity `u` are:

    u, c₂u, c₂u + √Δ, c₂u - √Δ

where the discriminant Δ is given by:

    Δ = λ + u²(c₂² - c₂).

The corresponding eigenvectors are:

    (0,1,0,0), (0,0,0,1), (1,w,λ1,z_p), (1,w,λ2,z_m).

The Roe matrix is computed starting from the matrix that has as columns the eigenvectors 

This provides the necessary framework for implementing the Roe method 
to solve the conservative system efficiently.
"""

function eigenvalues_Roe(u,c1,c2,λ)
    Δ = (c2^2 - c1*c2) * u^2 + λ*c1
    eval_w = u
    eval_0 = c2*u
    eval_p = c2*u + sqrt(Δ)
    eval_m = c2*u - sqrt(Δ)
    return  eval_p, eval_m, eval_0, eval_w
end

"""
    abs_Roe_matrix(ρl,ρr,ul,ur,vl,vr,c1,c2,λ)

Return the absolute value of the Roe matrix for the 1D Riemannn problem with respective left and right densities
`(ρl,ul,vl,wl)` and `(ρr,ur,vr,wr)`. The eigenvalues are increased to avoid sonic rarefaction wave (LLF method).
"""
function abs_Roe_matrix(ρl,ρr,wl,wr,ul,ur,vl,vr,c1,c2,λ)

    #### Roe average
    um = (sqrt(ρl)*ul + sqrt(ρr)*ur) / (sqrt(ρl) + sqrt(ρr))
    vm = (sqrt(ρl)*vl + sqrt(ρr)*vr) / (sqrt(ρl) + sqrt(ρr))
    wm = (sqrt(ρl)*wl + sqrt(ρr)*wr) / (sqrt(ρl) + sqrt(ρr))

    #### Eigenvalues and eigenvectors
    eval_p, eval_m, eval_0, eval_w= eigenvalues_Roe(um,c1,c2,λ)
    z_p = (c1*c2*um*vm - c2*vm*eval_p) / (c2*um - eval_p)
    z_m = (c1*c2*um*vm - c2*vm*eval_m) / (c2*um - eval_m)
    detP = c1 * (eval_m - eval_p)

    #### Increase the eigenvalues
    eval_l = eigenvalues_Roe(ul,c1,c2,λ)
    eval_r = eigenvalues_Roe(ur,c1,c2,λ)
    eval_p_fix = max(abs(eval_l[1]),abs(eval_r[1]))
    eval_m_fix = max(abs(eval_l[2]),abs(eval_r[2]))
    eval_0_fix = max(abs(eval_l[3]),abs(eval_r[3]))
    eval_w_fix = max(abs(eval_l[4]),abs(eval_r[4]))

    #### Compute Roe matrix A = P*D*P^{-1}, where P is the matrix given by the eigenvectors
    a11 = c1*(eval_p_fix*eval_m - eval_m_fix*eval_p)/detP
    a12 = 0.
    a13 = c1^2*(eval_m_fix - eval_p_fix)/detP
    a14 = 0.
    a21 = wm* (eval_p_fix*eval_m - eval_m_fix*eval_p + eval_w_fix*(eval_p - eval_m))/detP
    a22 = eval_w_fix
    a23 = wm*(eval_m_fix-eval_p_fix)/detP
    a24 = 0.
    a31 = eval_m*eval_p*(eval_p_fix - eval_m_fix)/detP
    a32 = 0.
    a33 = c1*(eval_m_fix*eval_m - eval_p_fix*eval_p)/detP
    a34 = 0.
    a41 = (-eval_m_fix*eval_p*z_m + eval_p_fix*eval_m*z_p + eval_0_fix*(eval_p*z_m - eval_m*z_p))/detP
    a42 = 0.
    a43 = c1*(eval_m_fix*z_m - eval_p_fix*z_p + eval_0_fix*(z_p - z_m))/detP
    a44 = eval_0_fix
    return a11,a12,a13,a14,a21,a22,a23,a24,a31,a32,a33,a34,a41,a42,a43,a44
end

"""
    flux_x(ρl,ρr,ul,ur,vl,vr,c1,c2,λ,method)

Return the flux along the x-axis for the 1D Riemannian problem given by the respective left and right
densities `(ρl,wl,ul,vl)` and `(ρr,wr,ur,vr)`. The chosen method `method` can be either `"Roe"` or `"HLLE"`.
The Roe flux has the following shape F = 0.5((F(U_l) + F(U_r)) - 0.5|A_Roe|(U_r - U_l)
"""
function flux_x(ρl,ρr,wl,wr,ul,ur,vl,vr,c1,c2,λ,method)
    if ρl<1e-9 && ρr<1e-9
        return 0.,0.,0.,0.
    end
    if method == "Roe"
        a11,a12,a13,a14,a21,a22,a23,a24,a31,a32,a33,a34,a41,a42,a43,a44 = abs_Roe_matrix(ρl,ρr,wl,wr,ul,ur,vl,vr,c1,c2,λ)
        avg = 0.5 .* (c1*ρl*ul + c1*ρr*ur, c1*ρl*wl*ul + c1*ρr*wr*ur,
                    c2*(ρl*ul^2 + ρr*ur^2) + λ*(ρl + ρr),
                    c2*(ρl*ul*vl + ρr*ur*vr))
        Roe_term = 0.5 .* (a11*(ρr - ρl) + a12*(ρr*wr - ρl*wl) + a13*(ρr*ur - ρl*ul) + a14*(ρr*vr - ρl*vl),
                        a21*(ρr - ρl) + a22*(ρr*wr - ρl*wl) + a23*(ρr*ur - ρl*ul) + a24*(ρr*vr - ρl*vl),
                        a31*(ρr - ρl) + a32*(ρr*wr - ρl*wl) + a33*(ρr*ur - ρl*ul) + a34*(ρr*vr - ρl*vl), 
                        a41*(ρr - ρl) + a42*(ρr*wr - ρl*wl) + a43*(ρr*ur - ρl*ul) + a44*(ρr*vr - ρl*vl))
        return avg .- Roe_term
    elseif method == "HLLE"
        um = (sqrt(ρl)*ul + sqrt(ρr)*ur) / (sqrt(ρl) + sqrt(ρr))
        vp_l = eigenvalues_Roe(ul,c1,c2,λ)
        vp_r = eigenvalues_Roe(ur,c1,c2,λ)
        vp_Roe = eigenvalues_Roe(um,c1,c2,λ)
        s_l = minimum(min(vp_l,vp_Roe))
        s_r = maximum(max(vp_r,vp_Roe))
        s_lm = min(s_l,0.)
        s_rp = max(s_r,0.)

        f_l = (c1*ρl*ul, c2*ρl*ul^2 + λ*ρl, c2*ρl*ul*vl)
        f_r = (c1*ρr*ur, c2*ρr*ur^2 + λ*ρr, c2*ρr*ur*vr)
        U_l = (ρl,ρl*ul,ρl*vl)
        U_r = (ρr,ρr*ur,ρr*vr)
        return ((s_rp.*f_l .- s_lm.*f_r) .+ (s_rp*s_lm) .* (U_r .- U_l)) ./ (s_rp - s_lm)
    else
        error("Method not defined!")
    end

end
