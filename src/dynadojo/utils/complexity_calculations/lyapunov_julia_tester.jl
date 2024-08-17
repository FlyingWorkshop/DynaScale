using DynamicalSystems

## LORENZ ##

@inline @inbounds function new_lorenz(u, p, t)
    σ = p[1]; ρ = p[2]; β = p[3]
    du1 = σ*(u[2]-u[1])
    du2 = u[1]*(ρ-u[3]) - u[2]
    du3 = u[1]*u[2] - β*u[3]
    return SVector{3}(du1, du2, du3)
end

lu0 = [-9.7869288, -15.03852, 20.533978]
lp = [10, 28, 8/3]
lds = ContinuousDynamicalSystem(new_lorenz, lu0, lp)
lλs = lyapunovspectrum(lds, 1000, Ttr = 0.0003002100350058257)

println("Lorenz:", lλs)

## ROSSLER ##

@inline @inbounds function new_rossler(u, p, t)
    a = p[1]; b = p[2]; c = p[3]
    du1 = -u[2] -u[3]
    du2 = u[1] + a*u[2]
    du3 = b + u[3]*(u[1] - c)
    return SVector{3}(du1, du2, du3)
end

ru0 = [6.5134412, 6.5134412, 0.34164294]
rp = [0.2, 0.2, 5.7]
rds = ContinuousDynamicalSystem(new_rossler, ru0, rp)
rλs = lyapunovspectrum(rds, 1000, Ttr = 0.001181916986164424)

println("Rossler:", rλs)