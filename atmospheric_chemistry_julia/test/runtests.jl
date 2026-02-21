#!/usr/bin/env julia
"""
Unit tests for the Atmospheric Chemistry Transport Model.
"""

using Test
using Dates

# Load the model
include(joinpath(@__DIR__, "..", "src", "AtmosphericChemistry.jl"))
using .AtmosphericChemistry

@testset "AtmosphericChemistry.jl" begin

    @testset "Grid" begin
        @testset "Horizontal grid construction" begin
            hg = HorizontalGrid(; resolution=2.0)
            @test hg.nlon == 180
            @test hg.nlat == 91
            @test hg.resolution == 2.0
            @test length(hg.lon) == 180
            @test length(hg.lat) == 91
            @test hg.lat[1] ≈ -90.0
            @test hg.lat[end] ≈ 90.0
            @test all(hg.area .> 0)
        end

        @testset "1-degree grid" begin
            hg = HorizontalGrid(; resolution=1.0)
            @test hg.nlon == 360
            @test hg.nlat == 181
            # Total area should approximately equal Earth's surface area
            total_area = sum(hg.area)
            earth_area = 4π * (6.371e6)^2
            @test abs(total_area - earth_area) / earth_area < 0.01
        end

        @testset "Vertical grid" begin
            vg = VerticalGrid(; nlevels=47)
            @test vg.nlevels == 47
            @test length(vg.a_half) == 48
            @test length(vg.b_half) == 48
            @test vg.b_half[1] ≈ 0.0
            @test vg.b_half[end] ≈ 1.0

            # Pressure should increase monotonically downward
            ps = fill(101325.0, 10, 10)
            p = pressure_at_levels(vg, ps)
            for i in 1:10, j in 1:10
                for k in 2:47
                    @test p[i, j, k] >= p[i, j, k-1]
                end
            end
        end

        @testset "Layer thickness" begin
            vg = VerticalGrid(; nlevels=20)
            ps = fill(101325.0, 5, 5)
            dp = layer_thickness_dp(vg, ps)
            @test all(dp .> 0)
            # Sum of dp should equal surface pressure minus top pressure
            for i in 1:5, j in 1:5
                total_dp = sum(dp[i, j, :])
                @test abs(total_dp - (ps[i, j] - vg.p_top)) / ps[i, j] < 0.01
            end
        end
    end

    @testset "Chemistry" begin
        @testset "Species" begin
            species = default_species()
            @test length(species) == 14
            @test species_index(species, :O3) == 1
            @test species_index(species, :NO) == 2
            @test species_index(species, :CO) == 4
        end

        @testset "Mechanism" begin
            mech = default_mechanism()
            @test mech.n_species == 14
            @test mech.n_reactions == 17
        end

        @testset "Rate constants" begin
            # NO + O3 reaction
            mech = default_mechanism()
            rxn = mech.reactions[2]  # NO + O3
            k = rate_constant(rxn, 298.0, 2.5e19)
            @test k > 0
            @test k < 1e-10  # Should be on the order of 1e-14

            # Temperature dependence
            k_cold = rate_constant(rxn, 220.0, 2.5e19)
            k_hot = rate_constant(rxn, 310.0, 2.5e19)
            @test k_hot > k_cold  # Positive activation energy
        end

        @testset "Unit conversions" begin
            n = ppb_to_number_density(100.0, 298.0, 101325.0)
            @test n > 0
            ppb = number_density_to_ppb(n, 298.0, 101325.0)
            @test abs(ppb - 100.0) / 100.0 < 0.001

            M = air_number_density(298.0, 101325.0)
            @test M > 2e19  # ~2.46e19 molec/cm³ at STP
            @test M < 3e19
        end

        @testset "Photolysis" begin
            # Daytime
            rates = compute_photolysis_rates(30.0, 0.0, 300.0, 0.0)
            @test rates.j_NO2 > 0
            @test rates.j_O3 > 0

            # Nighttime
            rates_night = compute_photolysis_rates(95.0, 0.0, 300.0, 0.0)
            @test rates_night.j_NO2 == 0.0
            @test rates_night.j_O3 == 0.0

            # SZA
            sza = solar_zenith_angle(45.0, 0.0, DateTime(2023, 6, 21, 12, 0, 0))
            @test sza >= 0
            @test sza < 90  # Should be daytime at noon on summer solstice in NH
        end

        @testset "Chemistry solver positivity" begin
            mech = default_mechanism()
            nspec = mech.n_species
            conc = zeros(Float64, 2, 2, 2, nspec)

            # Set initial concentrations
            for s in 1:nspec
                conc[:, :, :, s] .= max(mech.species[s].initial_ppb * 1e2, 1.0)
            end

            T = fill(280.0, 2, 2, 2)
            M = fill(2.5e19, 2, 2, 2)
            j = zeros(Float64, 2, 2, 2, 4)
            j[:, :, :, 1] .= 5e-3  # j_NO2

            solve_chemistry!(conc, mech, T, M, j, 600.0; n_substeps=4)

            # All concentrations should remain non-negative
            @test all(conc .>= 0.0)
        end
    end

    @testset "Transport" begin
        @testset "Advection positivity" begin
            hg = HorizontalGrid(; resolution=10.0)  # Coarse for speed
            vg = VerticalGrid(; nlevels=5)
            nlon, nlat = hg.nlon, hg.nlat
            nlevels = vg.nlevels
            nspec = 2

            conc = ones(Float64, nlon, nlat, nlevels, nspec) * 100.0
            u = ones(Float64, nlon, nlat, nlevels) * 5.0
            v = zeros(Float64, nlon, nlat, nlevels)
            w = zeros(Float64, nlon, nlat, nlevels)
            ps = fill(101325.0, nlon, nlat)

            advect!(conc, u, v, w, hg, vg, ps, 900.0)

            @test all(conc .>= 0.0)
            @test all(isfinite.(conc))
        end

        @testset "Diffusion stability" begin
            vg = VerticalGrid(; nlevels=10)
            nlon, nlat, nlevels = 4, 4, 10
            nspec = 1

            conc = zeros(Float64, nlon, nlat, nlevels, nspec)
            # Step function in vertical
            conc[:, :, 8:10, 1] .= 1000.0

            Kz = fill(10.0, nlon, nlat, nlevels)
            ps = fill(101325.0, nlon, nlat)

            diffuse_vertical!(conc, Kz, vg, ps, 900.0)

            @test all(conc .>= 0.0)
            @test all(isfinite.(conc))
            # Should have spread upward
            @test conc[1, 1, 7, 1] > 0.0
        end
    end

    @testset "Deposition" begin
        @testset "Dry deposition reduces surface concentration" begin
            species = default_species()
            nspec = length(species)
            hg = HorizontalGrid(; resolution=10.0)
            vg = VerticalGrid(; nlevels=5)
            nlon, nlat = hg.nlon, hg.nlat

            conc = ones(Float64, nlon, nlat, 5, nspec) * 1e10
            ps = fill(101325.0, nlon, nlat)
            u10 = fill(5.0, nlon, nlat)
            v10 = fill(2.0, nlon, nlat)
            blh = fill(1000.0, nlon, nlat)

            conc_before = copy(conc[:, :, 5, :])
            dry_deposition!(conc, species, hg, vg, ps, u10, v10, blh, 3600.0)

            # O3 (deposited) should decrease at surface
            o3_idx = species_index(species, :O3)
            @test all(conc[:, :, 5, o3_idx] .<= conc_before[:, :, o3_idx])
        end
    end

    @testset "EDGAR" begin
        @testset "Synthetic emissions" begin
            mgr = EDGARManager(; data_dir=tempdir(), resolution=2.0)
            hg = HorizontalGrid(; resolution=2.0)

            # This will generate synthetic data since no files exist
            emissions = load_emissions(mgr, :NOx; hgrid=hg)
            @test size(emissions) == (hg.nlon, hg.nlat)
            @test all(emissions .>= 0)
            @test any(emissions .> 0)  # Not all zero
        end
    end

    @testset "Model integration" begin
        @testset "Short simulation" begin
            config = ModelConfig(
                resolution = 10.0,   # Very coarse
                nlevels = 5,
                start_date = DateTime(2023, 1, 1),
                end_date = DateTime(2023, 1, 1, 1, 0, 0),  # 1 hour
                dt_advection = 1800.0,
                dt_chemistry = 1800.0,
                dt_diffusion = 1800.0,
                output_freq_h = 1,
                output_dir = joinpath(tempdir(), "actm_test"),
                do_convection = false,
                do_wet_dep = false,
            )

            state = ModelState(config)
            @test state.step_count == 0
            @test size(state.conc, 4) == 14  # 14 species

            run_simulation!(state)
            @test state.step_count > 0
            @test all(state.conc .>= 0.0)
        end
    end
end

println("\nAll tests passed!")
