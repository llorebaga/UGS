#!/usr/bin/env julia
"""
Generate fixed target unitaries and save to targets.hdf5.
Run this ONCE. All notebooks then load from this file.

Targets are planted with 5 piecewise-constant controls + random perturbation.
"""

using LinearAlgebra
using Random
using HDF5

# ── Hamiltonians (must match the notebooks) ──
H0 = [
    0 0 0;
    0 0.515916 0;
    0 0 1
]
H0 ./= norm(H0, Inf)

V = [
    0 0.707107 0;
    0.707107 0 1;
    0 1 0
]
V ./= norm(V, Inf)

T = 0.5

# ── Target generation parameters ──
m_plant = 5                      # number of piecewise-constant control slices for planting
dt_plant = T / m_plant
n_samples = 100
control_bound = 1.0
target_random_strength = 0.10

Random.seed!(2801)

# ── Helper functions ──
function trace_matrix(M::AbstractMatrix)
    d = size(M, 1)
    return sum(M[i, i] for i in 1:d)
end

function random_hermitian_perturbation(strength)
    A = randn(ComplexF64, 3, 3)
    H = (A + A') / 2
    H -= trace_matrix(H) / 3 * Matrix{ComplexF64}(I, 3, 3)   # traceless
    strength * H / max(opnorm(H), eps())
end

function get_unitary_planted(ctrl)
    prod(exp(-im * dt_plant * (H0 + xx * V)) for xx in ctrl)
end

# ── Generate targets ──
U_targets = zeros(ComplexF64, n_samples, size(H0)...)
target_family = Vector{String}(undef, n_samples)
planted_controls = zeros(m_plant, n_samples)

for i in 1:n_samples
    # Random control values uniform in [-control_bound, control_bound]
    ctrl = control_bound * (2 * rand(m_plant) .- 1)
    planted_controls[:, i] = ctrl

    # Compute the actual propagator for this control
    U_planted = get_unitary_planted(ctrl)

    # Perturb with a random unitary: exp(i*R), ||R|| < target_random_strength
    R = random_hermitian_perturbation(target_random_strength * rand())
    U_targets[i, :, :] = U_planted * exp(im * R)
    target_family[i] = "planted_noise_$i"
end

# ── Verify unitarity ──
for i in 1:n_samples
    U = U_targets[i, :, :]
    err = norm(U' * U - I)
    @assert err < 1e-10 "Target $i is not unitary (error=$err)"
end
println("All $n_samples targets verified unitary ✓")

# ── Save ──
filename = "targets.hdf5"
h5open(filename, "w") do fid
    fid["U_targets"] = U_targets
    fid["target_family"] = join(target_family, "\n")
    fid["planted_controls"] = planted_controls
    fid["m_plant"] = m_plant
    fid["dt_plant"] = dt_plant
    fid["target_random_strength"] = target_random_strength
    fid["control_bound"] = control_bound
    fid["n_samples"] = n_samples
    fid["T"] = T

    # Also save Hamiltonians for reference
    fid["H0"] = H0
    fid["V"] = V
end
println("Saved $filename ($n_samples targets, m_plant=$m_plant, strength=$target_random_strength)")
