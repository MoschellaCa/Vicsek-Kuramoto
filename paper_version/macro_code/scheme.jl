#-------- Add the package to the current path if not installed globally -------#
push!(LOAD_PATH,joinpath(pwd(),"src"))
#------------------------------------------------------------------------------#

# using SOH   # import the package
include("SOH.jl")


using DataFrames, Dates, CSV 
# Initializes the CSV file with headers if it doesn't exist.
# If the file exists, reads it to find the last simulation ID.
# Returns the next simulation ID to be used.
function initialize_csv_log(csv_path::String)::Int
    if !isfile(csv_path)
        # Create an empty DataFrame with the specified headers
        empty_df = DataFrame(simulation_id =[] , κ=[] , Lx=[] , Ly=[] , Δx = [], Δy=[], Δt=[], mean_ρ=[],
                             mean_w=[], mean_θ=[], range_ρ=[], range_w=[], range_θ=[], date = [])
        # Write the empty DataFrame to CSV (this creates the file with headers)
        CSV.write(csv_path, empty_df)
        return 1  # Start simulation IDs at 1
    else
        println("CSV file exists at ", csv_path, ". Reading last simulation ID...")
        # Read the existing CSV
        try
            existing_df = CSV.read(csv_path, DataFrame)
            if nrow(existing_df) == 0
                println("CSV file is empty. Starting simulation IDs at 1.")
                return 1
            else
                # Assuming simulation_id is sorted, find the maximum ID
                last_id = maximum(existing_df.simulation_id)
                println("Last simulation ID found: ", last_id)
                return last_id + 1
            end
        catch e
            println("Error reading CSV file: ", e)
            println("Starting simulation IDs at 1.")
            return 1
        end
    end
end

# Function to append a single simulation result to the CSV
function append_simulation_result(csv_path::String, simulation_id::Int, κ, Lx, Ly, Δx, Δy, Δt, mean_ρ, 
                                  mean_w, mean_θ, range_ρ, range_w, range_θ, date)
    # Create a DataFrame for the new row
    new_row = DataFrame(
        simulation_id =[simulation_id] ,
        κ=[κ] ,
        Lx=[Lx] ,
        Ly=[Ly] ,
        Δx=[Δx] ,
        Δy = [Δy], 
        Δt=[Δt], 
        mean_ρ=[mean_ρ], 
        mean_w=[mean_w], 
        mean_θ=[mean_θ],
        range_ρ=[range_ρ],
        range_w=[range_w],
        range_θ=[range_θ],
        date = [date]
    )
    
    # Append the new row to the CSV without writing headers
    CSV.write(csv_path, new_row, append=true)
end

output_csv = "data_macro_simulation.csv"
save_path = "/Users/moschellaca/Desktop/Vicsek-Kuramoto model/Macro_simulations/"
date = now()


#-------- Model parameters ----------------------------------------------------#
# The coefficients are computed using the function `coefficients_Vicsek` in the 
# script `toolbox.jl` for the Fokker-Planck and the BGK models. 
#for κ in 1.0:3.0:28.0
    κ = 50.0     # concentration parameter
    c1,c2,λ = coefficients_Vicsek(κ)

    #-------- Domain parameters ---------------------------------------------------#

    # Rectangular domain of size Lx*Ly
    Lx = 1.0   
    Ly = 1.0

    # Boundary conditions (possible choices "periodic", "Neumann", "reflecting")
    bcond_x = "periodic"
    bcond_y = "periodic"


    #-------- Numerical parameters ------------------------------------------------#

    # Numer of cells and spatial step
    ncellx = 200
    ncelly = 200
    Δx = Lx / ncellx
    Δy = Ly / ncelly

    # Time step 
    Δt = 0.001

    # Final time
    T = 30.

    # Method ("Roe" or "HLLE")
    method = "Roe"


    #-------- Exterior force ------------------------------------------------------#

    # If no exterior force:
    #Fx = nothing
    #Fy = nothing

    # Otherwise define the x and y components as (ncellx+2)*(ncelly+2) matrices.
    # See the examples in the script `init.jl`
    #Fx, Fy = flat_quadratic_potential_force(ncellx,ncelly,Δx,Δy,Lx/3,5.)

    #-------- Saving parameters ---------------------------------------------------#

    should_save = false
    simu_name = "simu"
    should_plot = false
    step_plot = 2   # A plot every `step_plot` iterations
    save_video = true   

    #-------- Initial conditions --------------------------------------------------#
    mean_rho = 1.0
    range_rho = 0.
    mean_omega = 5.
    range_omega = 1.
    mean_theta = 0.
    range_theta = 2*pi
    ρ,w,u,v = random_init(ncellx, ncelly, mean_rho, range_rho, mean_omega, range_omega, mean_theta, range_theta,bcond_x=bcond_x,bcond_y=bcond_y)


    #-------- Finally run the simulation ------------------------------------------#
    Fx = nothing
    Fy = nothing
    run!(
        ρ,w,u,v;
        Lx=Lx,Ly=Ly,Δt=Δt,
        c1=c1, c2=c2, λ=λ,
        final_time=T,
        bcond_x=bcond_x,bcond_y=bcond_y,
        Fx=Fx,Fy=Fy,
        method=method,
        simu_name="simu",
        should_save=should_save,save_step=1,
        should_plot=should_plot,plot_step=2,
        save_video=save_video,
        range=(0.,5.0),resolution=(800,600),theme="dark",fps=60
    );

    next_sim_id = initialize_csv_log(output_csv)
    # Append the results to the CSV file
    append_simulation_result(output_csv, next_sim_id, κ, Lx, Ly, Δx, Δy, Δt, mean_rho, mean_omega, mean_theta, range_rho, range_omega, range_theta, date)


#end 



