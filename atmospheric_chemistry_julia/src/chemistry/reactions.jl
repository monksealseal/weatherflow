# ---------------------------------------------------------------------------
# Chemical reaction mechanism — simplified tropospheric chemistry
# ---------------------------------------------------------------------------

"""
    ReactionType

Enum for reaction types.
"""
@enum ReactionType begin
    THERMAL           # k(T) = A * exp(-Ea/RT)
    TERMOLECULAR      # k(T,M) — pressure-dependent
    PHOTOLYSIS        # j-value, computed from radiation
end

"""
    Reaction(type, reactants, products, stoich_r, stoich_p, A, Ea, n)

A single chemical reaction with Arrhenius-type rate constant.

For thermal reactions: k = A × (T/300)^n × exp(-Ea / (R×T))
For termolecular: uses Troe formalism
For photolysis: rate set externally by photolysis module
"""
struct Reaction
    rtype      :: ReactionType
    reactants  :: Vector{Symbol}     # Species names
    products   :: Vector{Symbol}
    stoich_r   :: Vector{Float64}    # Stoichiometric coefficients (reactants)
    stoich_p   :: Vector{Float64}    # Stoichiometric coefficients (products)
    A          :: Float64            # Pre-exponential factor [cm³ molec⁻¹ s⁻¹] or [s⁻¹]
    Ea         :: Float64            # Activation energy [J mol⁻¹]
    n          :: Float64            # Temperature exponent
    label      :: String             # Human-readable label
end

"""
    ChemicalMechanism(species, reactions)

Complete chemical mechanism: species list + reaction list.
"""
struct ChemicalMechanism
    species   :: Vector{Species}
    reactions :: Vector{Reaction}
    n_species :: Int
    n_reactions :: Int
end

function ChemicalMechanism(species::Vector{Species}, reactions::Vector{Reaction})
    return ChemicalMechanism(species, reactions, length(species), length(reactions))
end

"""
    default_mechanism()

Build the default simplified tropospheric chemistry mechanism.

Includes key reactions for O3-NOx-CO-CH4-VOC chemistry:
- NOx photochemistry (NO2 photolysis, NO+O3)
- HOx chemistry (OH+CO, OH+CH4, HO2+NO)
- Ozone production and loss
- SO2 oxidation
- PAN formation/decomposition
- Formaldehyde chemistry
"""
function default_mechanism()
    species = default_species()

    reactions = Reaction[
        # ----- NOx photochemistry -----
        # R1: NO2 + hν → NO + O3  (net: NO2 photolysis produces O3)
        Reaction(PHOTOLYSIS,
            [:NO2], [:NO, :O3],
            [1.0], [1.0, 1.0],
            0.0, 0.0, 0.0,
            "NO2 + hv -> NO + O3"),

        # R2: NO + O3 → NO2 + O2
        Reaction(THERMAL,
            [:NO, :O3], [:NO2],
            [1.0, 1.0], [1.0],
            2.0e-12, 1400.0 * Constants.R_GAS, 0.0,
            "NO + O3 -> NO2"),

        # ----- HOx chemistry -----
        # R3: O3 + hν → O(¹D) + O2  → 2OH (in presence of H2O, net)
        Reaction(PHOTOLYSIS,
            [:O3], [:OH, :OH],
            [1.0], [1.0, 1.0],
            0.0, 0.0, 0.0,
            "O3 + hv -> 2OH (net)"),

        # R4: OH + CO → HO2 + CO2
        Reaction(THERMAL,
            [:OH, :CO], [:HO2],
            [1.0, 1.0], [1.0],
            1.5e-13, 0.0, 0.0,
            "OH + CO -> HO2"),

        # R5: OH + CH4 → HCHO + HO2  (simplified)
        Reaction(THERMAL,
            [:OH, :CH4], [:HCHO, :HO2],
            [1.0, 1.0], [1.0, 1.0],
            2.45e-12, 1775.0 * Constants.R_GAS, 0.0,
            "OH + CH4 -> HCHO + HO2"),

        # R6: HO2 + NO → OH + NO2
        Reaction(THERMAL,
            [:HO2, :NO], [:OH, :NO2],
            [1.0, 1.0], [1.0, 1.0],
            3.3e-12, -270.0 * Constants.R_GAS, 0.0,
            "HO2 + NO -> OH + NO2"),

        # R7: HO2 + O3 → OH + 2O2
        Reaction(THERMAL,
            [:HO2, :O3], [:OH],
            [1.0, 1.0], [1.0],
            1.0e-14, 490.0 * Constants.R_GAS, 0.0,
            "HO2 + O3 -> OH"),

        # R8: HO2 + HO2 → H2O2 + O2
        Reaction(THERMAL,
            [:HO2, :HO2], [:H2O2],
            [1.0, 1.0], [1.0],
            3.0e-13, -460.0 * Constants.R_GAS, 0.0,
            "HO2 + HO2 -> H2O2"),

        # R9: H2O2 + hν → 2OH
        Reaction(PHOTOLYSIS,
            [:H2O2], [:OH, :OH],
            [1.0], [2.0],
            0.0, 0.0, 0.0,
            "H2O2 + hv -> 2OH"),

        # ----- HCHO chemistry -----
        # R10: HCHO + hν → CO + 2HO2  (radical channel)
        Reaction(PHOTOLYSIS,
            [:HCHO], [:CO, :HO2, :HO2],
            [1.0], [1.0, 1.0, 1.0],
            0.0, 0.0, 0.0,
            "HCHO + hv -> CO + 2HO2"),

        # R11: HCHO + OH → CO + HO2 + H2O
        Reaction(THERMAL,
            [:HCHO, :OH], [:CO, :HO2],
            [1.0, 1.0], [1.0, 1.0],
            1.0e-11, 0.0, 0.0,
            "HCHO + OH -> CO + HO2"),

        # ----- SO2 oxidation -----
        # R12: SO2 + OH → H2SO4 (→ PM2.5 as proxy)
        Reaction(THERMAL,
            [:SO2, :OH], [:PM25],
            [1.0, 1.0], [1.0],
            1.5e-12, 0.0, 0.0,
            "SO2 + OH -> PM2.5 (sulfate)"),

        # ----- PAN chemistry -----
        # R13: NO2 + CH3C(O)O2 → PAN  (simplified: use HO2 as proxy for acyl peroxy)
        Reaction(THERMAL,
            [:NO2, :HO2], [:PAN],
            [1.0, 1.0], [1.0],
            1.0e-11, 0.0, 0.0,
            "NO2 + HO2 -> PAN (simplified)"),

        # R14: PAN → NO2 + HO2  (thermal decomposition)
        Reaction(THERMAL,
            [:PAN], [:NO2, :HO2],
            [1.0], [1.0, 1.0],
            5.0e16, 13500.0 * Constants.R_GAS, 0.0,
            "PAN -> NO2 + HO2"),

        # ----- Isoprene chemistry (very simplified) -----
        # R15: ISOP + OH → HCHO + HO2
        Reaction(THERMAL,
            [:ISOP, :OH], [:HCHO, :HO2],
            [1.0, 1.0], [1.0, 1.0],
            1.0e-10, 0.0, 0.0,
            "ISOP + OH -> HCHO + HO2"),

        # R16: OH + HNO3 → NO3 + H2O  (OH sink)
        Reaction(THERMAL,
            [:OH, :HNO3], [],
            [1.0, 1.0], Float64[],
            1.3e-13, 0.0, 0.0,
            "OH + HNO3 -> products"),

        # R17: NO2 + OH → HNO3
        Reaction(TERMOLECULAR,
            [:NO2, :OH], [:HNO3],
            [1.0, 1.0], [1.0],
            2.0e-30, 0.0, 3.0,    # k0 for Troe formalism
            "NO2 + OH + M -> HNO3"),
    ]

    return ChemicalMechanism(species, reactions)
end

"""
    rate_constant(rxn, T, M)

Compute the rate constant for a reaction at temperature T [K] and
number density M [molec cm⁻³].
"""
function rate_constant(rxn::Reaction, T::Float64, M::Float64)
    if rxn.rtype == PHOTOLYSIS
        return 0.0  # Set externally
    elseif rxn.rtype == TERMOLECULAR
        # Troe formalism
        k0 = rxn.A * (T / 300.0)^(-rxn.n) * M
        kinf = 1.0e-11  # Default high-pressure limit
        kr = k0 / kinf
        f = 0.6
        k = (k0 / (1.0 + kr)) * f^(1.0 / (1.0 + (log10(kr))^2))
        return k
    else
        # Arrhenius: k = A × (T/300)^n × exp(-Ea/(R×T))
        return rxn.A * (T / 300.0)^rxn.n * exp(-rxn.Ea / (Constants.R_GAS * T))
    end
end
