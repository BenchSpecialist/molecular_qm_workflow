############ Property Keys ############
# Molecule-level properties
DFT_ENERGY_KEY = "energy_hartree"
HOMO_KEY = "homo_eV"
LUMO_KEY = "lumo_eV"
ESP_MIN_KEY = "esp_min_eV"
ESP_MAX_KEY = "esp_max_eV"
DIPOLE_X_KEY = "dipole_x_debye"
DIPOLE_Y_KEY = "dipole_y_debye"
DIPOLE_Z_KEY = "dipole_z_debye"  # Fixed typo in the key name

# Atom-level properties (plus elements, coordinates)
DFT_FORCES_KEY = "forces"  # used to generated per-axis label, e.g. "forces_x"
CHELPG_CHARGE_KEY = "chelpg_charge"
