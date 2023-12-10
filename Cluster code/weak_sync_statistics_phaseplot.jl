# Working code 31.03.23. This is the SDE version in which we also compute
# and plot the average trajectory of particles 


# Libraries
using DifferentialEquations, Plots, UnPack, Distributions, StaticArrays, CellListMap# libraries
using CellListMap.PeriodicSystems 
import CellListMap.wrap_relative_to
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())


# Function calculating and updating the potentials
function update_potential!(i,j,d2,potential,u,R)
    if sqrt(d2) < R
        dK = sin(u[j,3] - u[i,3])
        df = u[j,4]-u[i,4]

        potential[i] += @SVector[dK, df]
        potential[j] -= @SVector[dK, df]
    end
    return potential
end



# Function to compute how many neighbors each particles has
# IMPORTANT HINT: I check that actually the routine neighbourlist look at the distance and not at the squared distance
function update_neighbourlist!(position,neighbourlist_update, N,R,L)
    n =neighborlist(position,R; unitcell=[L,L])
    n1 = [ SVector{1,Int64}(n[i][1]) for i in 1:size(n,1)]
    n2 = [ SVector{1,Int64}(n[i][2]) for i in 1:size(n,1)]
    for j in 1:N
        count1 = count(i->(i == [j]), n1)
        count2 = count(i->(i == [j]), n2)
        neighbourlist_update[j]=  @SVector[count1 + count2]
    end 
    return neighbourlist_update
end



function Vicsek_model!(du,u,p,t)
    @unpack R, v₀,L,k_1, k_2, τ, N, system, cbias = p
    # This should trigger the update of the potential
    for i in 1:N
        system.xpositions[i] = @SVector[u[i,1], u[i,2]]
    end
    PeriodicSystems.map_pairwise!((x,y,i,j,d2,potential) -> update_potential!(i,j,d2,potential,u,R), system; show_progress = true)
    
    for i=1:N
        # dX_i/dt = V_i
        du[i,1] = v₀ * cos(u[i,3])
        du[i,2] = v₀ * sin(u[i,3])

        # dθ_i/dt
        du[i,3] = u[i,4] + (k_1*1/N) * ((L^2)/(pi*(R^2))) * system.potential[i][1]

        # dω_i/dt 
        du[i,4] = -u[i,4]*τ + (k_2*1/N) * ((L^2)/(pi*(R^2))) * system.potential[i][2] + cbias * sign(u[i,4])*exp(-abs(u[i,4])*20)*0.5
    end
    nothing
end


function σ_Vicsek!(du,u,p,t)
    @unpack noise1, noise2, N = p
    for i=1:N
        du[i, 1] = 0.0
        du[i, 2] = 0.0  # dX_i/dt = V_i (second component)
        du[i, 3] = noise1
        du[i, 4] = noise2
    end
end



#Parameters 
N=10_000 # Total number of individuals
k_1=0.4 # factor multiplying the interaction potential in the direction variable
k_2=2 # factor multiplying the interaction potential in the angular velocity variable
end_time = 300 # Final time for the simulation
noise1 = 0.1
noise2 = 0.1
R = 1   #Interaction radius
v₀ = 2    #constant velocity of particles 
L = 64  #dimension of the periodic domain
τ_interval = [0.01, 0.1, 1, 10, 100, 1000]
c_interval = [0.01, 0.1, 1, 10, 100, 1000]
k_interval = [0.1, 10, 50, 100]



# Initial data:
pos = rand(2,N) .* L
#the angle is in radiants
θ = rand(N)*2π
ω = 5*rand(N)
ω[Int64(N/4):end] = - ω[Int64(N/4):end]
u_init = [pos' θ  ω]
# This is the vector that the periodic system needs for the CellListMap algorithm
xpos = [ SVector{2,Float64}(pos[:,i]) for i in 1:N]
neighbours_init = zeros(1,N)
neighbours_upd = [ SVector{1,Int64}(neighbours_init[i]) for i in 1:N]
 




for τ ∈ τ_interval
    for k_1 ∈ k_interval
        for k_2 ∈ k_interval


            cbias = τ  
            param = (R=R, v₀ , L, N, k_1, k_2, τ, cbias)

            # Periodic system 
            # The cutoff should be the distance and not the distance squared
            system = PeriodicSystem(
                    xpositions = xpos,
                    unitcell = [param.L,param.L] , 
                    cutoff = 2*param.R, 
                    output = similar(xpos),
                    output_name = :potential
                )
            Threads.nthreads()
            system.parallel = true
            # Put the systems inside the param in order to trigger the update 
            param = (R=R, v₀, L, N, k_1, k_2, τ, system, noise1, noise2, cbias)


            # Solve the ODE with ODE solver 
            #probVK = ODEProblem(Vicsek_model!,u_init,(0.0,end_time),param)
            #time_execution = @elapsed solVK = solve(probVK, Tsit5())
            #@profview solve(probVK, Tsit5())
            prob_sde = SDEProblem(Vicsek_model!,σ_Vicsek!,u_init,(0.0,end_time), param)
            #focus_inds = [200,300]
            #Solution of the porblem with SDE solver 
            time_execution = @elapsed sol_sde= solve(prob_sde,LambaEM(),dt=0.1,progress=true,progress_steps=1)

            using BenchmarkTools
            #@benchmark solve(probVK, Tsit5())
            #@btime



            # Plotting averaged angular velocity and averaged velocity 
            average_angvel = []
            for time ∈ 1:1:size(sol_sde.t,1)
                append!(average_angvel, mean(sol_sde[:,4, time], dims=1))
            end 
            fig1= plot(sol_sde.t,average_angvel, title="Averaged angular velocity", linewidth=1, legend= false, xlabel= "Time")


            average_vel_x= []
            average_vel_y= []
            for time ∈ 1:1:size(sol_sde.t,1)
                append!(average_vel_x, mean(v₀*cos.(sol_sde[:,3, time]), dims=1))
                append!(average_vel_y, mean(v₀*sin.(sol_sde[:,3, time]), dims=1))
            end 
            fig2= plot(sol_sde.t,average_vel_x, label = "Vx")
            plot!(sol_sde.t,average_vel_y, title="Averaged velocity", label="Vy", linewidth=1)
            fig3 = plot(fig1,fig2, layout = (2,1))
            plot!(size=(1000,800))
            savefig(fig3,"Average_behavior_all_forces_N=$N,R=$(param[1]),cbias=$cbias,L=$(param[3]),v=$(param[2]),k_1=$(param[5]),k_2=$(param[6]),τ=$(param[7]),noise1=$(param[9]),noise2=$(param[10]).png") 



            # Plot  
            anim = @animate for time ∈ (end_time-30.0):0.2:end_time
                s = sol_sde(time)
                x=[]
                y=[]
                z=[]
                u=[]
                v=[]
                for i=1:N
                    append!(x,mod(s[i,1], param.L))
                    append!(y,mod(s[i,2], param.L))
                    append!(z, mod(s[i,3], 2π))
                    append!(u,cos(s[i,3]))
                    append!(v,sin(s[i,3]))
                end
                gr(size=(800,800),markerstrokewidth=0, markersize = 4, legend = false, colorbar = true, colorbar_scale = asinh)
                fig4 = scatter(x,y,marker_z=z, color=:rainbow, xlims=[0,L],ylims=[0,L])
                plot(fig4)
                title!("t=$time")
                end
            gif(anim, "animationVicsek_N=$N,R=$(param[1]),cbias = $cbias,L=$(param[3]),v=$(param[2]),k_1=$(param[5]),k_2=$(param[6]),R=$(param[1]),τ=$(param[7]),noise1=$(param[9]),noise2=$(param[10]).gif", fps = 10)





            #Plot of the fake particle 
            dt=0.2
            x_0 = 10
            y_0 = 10
            average_pos_x = []
            average_pos_y = []
            append!(average_pos_x,x_0)
            append!(average_pos_y,y_0)

            for time ∈ 2:1:size(sol_sde.t,1)
                append!(average_pos_x, average_pos_x[time-1] +dt*average_vel_x[time-1])
                append!(average_pos_y, average_pos_y[time-1] +dt*average_vel_y[time-1])
            end 
            n_color = size(average_pos_x,1)
            color = palette(:viridis, n_color)
            #Plot of the fake particle
            t = LinRange(0, end_time, n_color) 
            fig5= plot(average_pos_x,average_pos_y,line_z = t ,color=:rainbow, colorbar = true, linewidth = 2)
            plot!(fig5, size=(600,600))
            savefig(fig5,"Average_trajectory_all_forces_N=$N,R=$(param[1]),cbias = $cbias,L=$(param[3]),v=$(param[2]),k_1=$(param[5]),k_2=$(param[6]),τ=$(param[7]),noise1=$(param[9]),noise2=$(param[10]).png") 

        end
    end
end






######################################################################################################## OLD FUNCTIONS

# Compute average number of neighbors in a specific time  
s = sol_sde(1)
position = [ SVector{2,Float64}(pos[:,i]) for i in 1:N]
    for i=1:N
        position[i] = @SVector[mod(s[i,1],param.L), mod(s[i,2],param.L)]
    end
neighbours_upd = update_neighbourlist!(position,neighbours_upd, N,R,L)
mean(neighbours_upd)




#Plot of the fake particle 
dt=0.2
x_0 = 10
y_0 = 10
average_pos_x = []
average_pos_y = []
append!(average_pos_x,x_0)
append!(average_pos_y,y_0)
for time ∈ 2:1:size(sol_sde.t,1)
    append!(average_pos_x, average_pos_x[time-1] +dt*average_vel_x[time-1])
    append!(average_pos_y, average_pos_y[time-1] +dt*average_vel_y[time-1])
end 
fig3= plot(average_pos_x,average_pos_y)
plot!(fig3, size=(1000,1000))



# THIS IS WORKING 
n =neighborlist(system.xpositions,3.0; unitcell=[130,130])
n1 = [ SVector{1,Int64}(n[i][1]) for i in 1:size(n,1)]
n2 = [ SVector{1,Int64}(n[i][2]) for i in 1:size(n,1)]
for j in 1:N
    count1 = count(i->(i == [j]), n1)
    count2 = count(i->(i == [j]), n2)
    neighbours_upd[j]=  @SVector[count1 + count2]
end 
#show(stdout, "text/plain", neighbours_update)
mean(neighbours_upd)


system.xpositions
n =neighborlist(system.xpositions, 3.0; unitcell=[135,135])
n1 = []
n2 = []
for i in 1:size(n,1)
    append!(n1, n[i][1])
    append!(n2, n[i][2])
end 
#show(stdout, "text/plain", n1)
#show(stdout, "text/plain", n2)
show(stdout, "text/plain", n)
total_counter=[]
for j in 1:10000
    count1 = count(i->(i == j), n1)
    count2 = count(i->(i == j), n2)
    append!(total_counter, count1 + count2)
end 
mean(total_counter)





# Plot  
anim = @animate for time ∈ (end_time-30.0):0.2:end_time
    s = sol_sde(time)
    x=[]
    y=[]
    z=[]
    u=[]
    v=[]
    for i=1:N
        append!(x,mod(s[i,1], param.L))
        append!(y,mod(s[i,2], param.L))
        append!(z, mod(s[i,3], 2π))
        append!(u,cos(s[i,3]))
        append!(v,sin(s[i,3]))
    end
    gr(size=(800,800),markerstrokewidth=0, markersize = 4, legend = false, colorbar = true, colorbar_scale = asinh)
    #t_inds = sol_sde.t .< time
    #average_x = mean(sol_sde[:,1, t_inds], dims=1)
    #average_y = mean(sol_sde[:,2, t_inds], dims=1)
    #fig1 = plot(average_x', average_y',title = " N=$N v=$(param[2]) R=$(param[1]) L=$(param[3]) τ=$(param[7]) k_1=$(param[5]) k_2=$(param[6]) noise1=$(param[9]) noise2=$(param[10]) t=$time")
    fig2 = scatter(x,y,marker_z=z, color=:rainbow, xlims=[0,L],ylims=[0,L])
    #plot!( sol_sde[focus_inds, 1, t_inds]', sol_sde[focus_inds, 2, t_inds]', color = "red", linewidth=2) # I fixed the axis, otherwise, erase xlims and ylims
    #quiver!(x,y,quiver=(u,v))
    plot(fig2)
    end
gif(anim, "animationVicsek N=$N R=$(param[1]) L=$(param[3]) v=$(param[2])  k_1=$(param[5]) k_2=$(param[6]) R=$(param[1]) τ=$(param[7]) noise1=$(param[9]) noise2=$(param[10]).gif", fps = 10)




#Plot of the fake particle 
dt=0.2
x_0 = 10
y_0 = 10
average_pos_x = []
average_pos_y = []
append!(average_pos_x,x_0)
append!(average_pos_y,y_0)

for time ∈ 2:1:size(sol_sde.t,1)
    append!(average_pos_x, average_pos_x[time-1] +dt*average_vel_x[time-1])
    append!(average_pos_y, average_pos_y[time-1] +dt*average_vel_y[time-1])
end 
n_color = size(average_pos_x,1)
color = palette(:viridis, n_color)
#Plot of the fake particle
t = LinRange(0, end_time, n_color) 
fig5= plot(average_pos_x,average_pos_y,line_z = t ,color=:rainbow, colorbar = true, linewidth = 2)
plot!(fig5, size=(600,600))