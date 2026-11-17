"""Universal physical constants, each with its source.

Only values shared by more than one area live here. Property tables that a
single area uses (for example the soft-tissue acoustics of the ``head``
package) stay in that area.

Water and air are given at one reference state: 20 degC and 101.325 kPa, a
room-temperature water bath.
"""

# assumed, to verify: 20 degC (bench measurements) rather than 37 degC (body
# temperature) as the reference state of the water and air constants below.

# Sound speed assumed for beamforming and for the homogeneous reference medium [m/s].
# The conventional average for soft tissue used by clinical scanners; a
# convention, not a measurement. Szabo (2014), Diagnostic Ultrasound Imaging:
# Inside Out, 2nd ed., ch. 1.
C_REF = 1540.0

# Sound speed of pure water at 20 degC [m/s]. Del Grosso & Mader (1972),
# J. Acoust. Soc. Am. 52(5B):1442-1446, give 1482.34 m/s.
# assumed, to verify: the primary source was not re-read when this was written;
# the secondary sources that were checked give 1481-1482 m/s.
C_WATER = 1482.3

# Density of pure water at 20 degC [kg/m^3]. Tanaka et al. (2001), Metrologia
# 38(4):301-309, give 998.207 kg/m^3.
# assumed, to verify: the primary source was not re-read when this was written.
# NIST's X-ray tables list water at a nominal 1000 kg/m^3 instead.
RHO_WATER = 998.2

# Density of dry air near sea level at 20 degC [kg/m^3]. NIST X-Ray Mass
# Attenuation Coefficients (Hubbell & Seltzer), Table 2, "Air, Dry (near sea
# level)": 1.205e-3 g/cm^3.
RHO_AIR = 1.205

# Sound speed of dry air at 20 degC [m/s]. Ideal gas, sqrt(gamma * R * T / M)
# with gamma = 1.400, M = 28.9647 g/mol, T = 293.15 K (ISO 2533 standard
# atmosphere relation), which gives 343.2 m/s.
C_AIR = 343.2

# Basis materials of the CT model, in the channel order of material-fraction volumes.
BASIS_MATERIALS = ("air", "water", "soft_tissue", "cortical_bone")

# Mass density of soft tissue [kg/m^3]. NIST X-Ray Mass Attenuation Coefficients
# (Hubbell & Seltzer), Table 2, "Tissue, Soft (ICRU-44)": 1.060 g/cm^3
# (https://physics.nist.gov/PhysRefData/XrayMassCoef/tab2.html, checked 2026-11-17).
# Used as the soft-tissue basis material of the CT model and as the pore content
# of skull bone in the head's acoustic density, so both see the same mass.
RHO_SOFT_TISSUE_ICRU44 = 1060.0

# Mass density of pore-free cortical bone [kg/m^3]. NIST Table 2 (same URL),
# "Bone, Cortical (ICRU-44)": 1.920 g/cm^3. Used as the bone basis material of
# the CT model and as the solid phase of skull bone in the head's acoustic density.
RHO_CORTICAL_BONE_ICRU44 = 1920.0

# Constants of the porosity mixing law of Aubry et al. (2003), J. Acoust. Soc.
# Am. 113(1):84-93, Eqs. 5-6. They are the defaults of the head's ground-truth
# skull model (head.materials.SkullMixing) and the nominal constants of the
# clinical HU -> acoustics map (ct.config.HuMappingConfig).
# Sound speed at porosity 1, water-filled pores [m/s]: c_min = 1.5 mm/us.
AUBRY_2003_C_MIN = 1500.0
# Sound speed of pore-free cortical bone [m/s]: c_max = 2.9 mm/us (the value of
# Fry & Barger 1978 for both tables).
AUBRY_2003_C_MAX = 2900.0
# Exponent beta of porosity in the absorption law [-]: best match 0.5, tested
# range 0.3-0.7.
AUBRY_2003_ABSORPTION_EXPONENT = 0.5

# Default exponent y of the power-law acoustic absorption alpha0 * f**y [-],
# shared by the head's material mapping and the CT-derived acoustic map, which
# must use the same exponent. Soft tissue is close to 1, but k-Wave's power-law
# absorption has a dispersion term proportional to tan(pi y / 2), singular at
# y = 1, so 1.1 is used; assumed, to verify.
ALPHA_POWER_DEFAULT = 1.1
