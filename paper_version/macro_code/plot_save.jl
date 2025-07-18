using CairoMakie, ProgressMeter, Statistics

"""
    plot_rhoUV(
        ρ,w,u,v,
        Δx,Δy,
        range,
        save_plot=true,save_video=false,dir_name=pwd(),file_name="0.png";
        theme="dark", resolution=(800,600),arrow_step=20,fps=40,kwargs...
    )

Return the figure, axes, heatmap and arrows associated to the density `ρ`, the angular velocity 'w' and velocity field
`(u,v)`. The plot is saved in the directory `dir_name` in the file `file_name`. Two themes
are available, either `"dark"` (default) or `"light"`.
"""
function plot_rhoUV(
    ρ,w,u,v,
    Δx,Δy,
    range1,
    save_plot=true,save_video=false,dir_name=pwd(),file_name="0.png";
    theme="light", resolution=(800,600),arrow_step=20,fps=40,kwargs...
)
    if theme == "dark"
        set_theme!(theme_black())
        colormap = :linear_kryw_5_100_c67_n256
        arrowcolor = :black
    elseif theme == "light"
        set_theme!()
        colormap = Reverse(:linear_kryw_5_100_c67_n256)
        arrowcolor = :black
    else
        AttributeError("Theme not defined!")
    end
    ncellx = size(ρ)[1] - 2
    ncelly = size(ρ)[2] - 2
    Lx = ncellx*Δx
    Ly = ncelly*Δy
    x = Δx .* collect(0:(ncellx-1)) .+ Δx/2
    y = Δy .* collect(0:(ncelly-1)) .+ Δy/2
    rho = ρ[2:ncellx+1,2:ncelly+1]
    fig = Figure(resolution=resolution)
    ax = Axis(fig[1, 1], title="time=0.00")
    hm1 = heatmap!(ax,x,y,rho,colorrange=range1,colormap=colormap)
    ax.aspect = AxisAspect(1)
    xlims!(ax,0,Lx)
    ylims!(ax,0,Ly)
    Colorbar(fig[1,2],hm1)

    extract_x = floor(Int,arrow_step/2):arrow_step:ncellx
    extract_y = floor(Int,arrow_step/2):arrow_step:ncelly
    xx = x[extract_x]
    yy = y[extract_y]
    u_grid = u[2:ncellx+1,2:ncelly+1]
    v_grid = v[2:ncellx+1,2:ncelly+1]
    uu = u_grid[extract_x,extract_y]
    vv = v_grid[extract_x,extract_y]
    arrows = arrows!(ax,xx,yy,uu,vv)
    arrows.color = arrowcolor
    arrows.lengthscale = 2/3*arrow_step*Δx
    arrows.arrowsize = floor(Int,1/80*resolution[1])
    arrows.linewidth = floor(Int,3/800*resolution[1])
    arrows.origin = :center
    if save_plot
        save(joinpath(dir_name,file_name),fig)
    end
    if save_video
        stream1 = VideoStream(fig,framerate=fps)
        return fig, ax, hm1, arrows, stream1
    else
        return fig, ax, hm1, arrows
    end
end



function plot_thetaUV(
    ρ,w,u,v,
    Δx,Δy,
    range2,
    save_plot=true,save_video=false,dir_name=pwd(),file_name="0.png";
    theme="light", resolution=(800,600),arrow_step=20,fps=40,kwargs...
)
    if theme == "dark"
        set_theme!(theme_black())
        colormap = :hsv
        arrowcolor = :black
    elseif theme == "light"
        set_theme!()
        colormap = Reverse(:linear_kryw_5_100_c67_n256)
        arrowcolor = :black
    else
        AttributeError("Theme not defined!")
    end
    ncellx = size(ρ)[1] - 2
    ncelly = size(ρ)[2] - 2
    Lx = ncellx*Δx
    Ly = ncelly*Δy
    x = Δx .* collect(0:(ncellx-1)) .+ Δx/2
    y = Δy .* collect(0:(ncelly-1)) .+ Δy/2
    rho = ρ[2:ncellx+1,2:ncelly+1]
    fig = Figure(resolution=resolution)
    ax = Axis(fig[1, 1], title="time=0.00")
    # compute θ ∈ [0,2π)
    u_grid = u[2:end-1, 2:end-1]
    v_grid = v[2:end-1, 2:end-1]
    θ  = atan.(v_grid, u_grid)
    θ .= ifelse.(θ .< 0, θ .+ 2π, θ)
    hm2 = heatmap!(ax,x,y,θ,colorrange = range2, colormap=colormap)
    ax.aspect = AxisAspect(1)
    xlims!(ax,0,Lx)
    ylims!(ax,0,Ly)
    Colorbar(fig[1,2],hm2)

    if save_video
        stream2 = VideoStream(fig,framerate=fps)
        return fig, ax, hm2, stream2
    else
        return fig, ax, hm2
    end
end



"""
    update_plot!(fig,ax,hm,arrows,ρ,w,u,v,title,dir_name,file_name,save_plot=true,stream=nothing)

Update a plot created with the function `plot_rhoUV` and save it.
"""
function update_plot!(fig,ax,hm1,arrows,ρ,w,u,v,title,dir_name,file_name,save_plot=true,stream=nothing)
    ncellx = size(ρ)[1] - 2
    ncelly = size(ρ)[2] - 2
    rho = ρ[2:ncellx+1,2:ncelly+1]
    hm1[3] = rho
    u_grid = u[2:ncellx+1,2:ncelly+1]
    v_grid = v[2:ncellx+1,2:ncelly+1]
    extract_x = 10:20:ncellx
    extract_y = 10:20:ncelly
    uu = u_grid[extract_x,extract_y]
    vv = v_grid[extract_x,extract_y]
    arrows[:directions] = vec(Vec2f.(uu,vv))
    ax.title = title
    if save_plot
        save(joinpath(dir_name,file_name),fig)
    end
    if !isnothing(stream)
        recordframe!(stream)
    end
end


"""
    update_plot!(fig,ax,hm,arrows,ρ,w,u,v,title,dir_name,file_name,save_plot=true,stream=nothing)

Update a plot created with the function `plot_rhoUV` and save it.
"""
function update_plot_theta!(fig,ax,hm2,ρ,w,u,v,title,dir_name,file_name,save_plot=true,stream=nothing)
    ncellx = size(ρ)[1] - 2
    ncelly = size(ρ)[2] - 2
    u_grid = u[2:end-1, 2:end-1]
    v_grid = v[2:end-1, 2:end-1]
    θ = atan.(v_grid, u_grid)
    θ .= ifelse.(θ .< 0, θ .+ 2π, θ)
    hm2[3]  = θ
    ax.title = title
    if save_plot
        save(joinpath(dir_name,file_name),fig)
    end
    if !isnothing(stream)
        recordframe!(stream)
    end
end



"""
    save_data!(
        data,
        ρ,w,u,v,
        dir_name,file_name;key="data")

Update the dictionary `data` with the values `(ρ,w,u,v)` and save it.
"""
function save_data!(
    data,
    ρ,w,u,v,
    dir_name,file_name;key="data")
    data[:ρ] = ρ
    data[:w] = w
    data[:u] = u
    data[:v] = v
    save(joinpath(dir_name,file_name),key,data)
end

"""
    radial_density(ρ,Δx,Δy,K=400)

Compute the radial density as the mean of `ρ` in `K` annuli of
equal lengths and evenly spread radii between 0 and the maximal
radius in the box. Return the vector of radii and the radial density.
"""
function radial_density(ρ,Δx,Δy,K=400)
    ncellx = size(ρ)[1] - 2
    ncelly = size(ρ)[2] - 2
    Lx = ncellx*Δx
    Ly = ncelly*Δy
    mx = Lx/2
    my = Ly/2
    rmat = zeros(ncellx,ncelly)
    for i in 1:ncellx
        for j in 1:ncellx
            xij = (i-1)*Δx + Δx/2
            yij = (j-1)*Δy + Δy/2
            rmat[i,j] = sqrt((xij-mx)^2 + (yij-my)^2)
        end
    end
    rvec = vec(rmat)
    rho = vec(ρ[2:ncellx+1,2:ncelly+1])
    r = LinRange(0.,maximum(rvec),K)
    dr = r[2]-r[1]
    rho_r = zeros(K)
    for k in 1:K
        rk = r[k]+dr
        rho_r[k] = mean(rho[abs.(rvec.-rk).<dr/2])
    end
    return r, rho_r
end
