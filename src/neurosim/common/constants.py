"""Universal physical constants, each with its source.

Only values shared by more than one area live here. Tissue and bone property
tables belong to the ``head`` package.

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
