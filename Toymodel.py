from pathlib import Path

import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt

output_dir = Path(__file__).resolve().parent

#c has to be in cm and masses in g
c = const.c * 100
sol_mass = 1.989e33
sol_radius = 6.957e10
day = 86400

beta = (3.0)
min_vel = (0.1 * c)
thres_vel = (0.3 * c)
total_mass = (0.01 * sol_mass)
opacity = (20.0)
t_elapsed = (3.0 * day)

def peak_lightcurve_time(total_mass, beta, min_vel, opacity):
    numerator = (3 * total_mass * opacity)
    denominator = (4 * const.pi * beta * min_vel * c)
    return ((numerator /denominator) ** (1/2)) / day


def fast_mass_ejected(total_mass, thres_vel, min_vel, beta):
    return (total_mass * (((min_vel) / thres_vel) ** beta)) / sol_mass

def rad_diffusion_tscale(total_mass, thres_vel, min_vel, beta, t_elapsed, opacity):
    #opacity needs to be in cm/g
    fast_mass = fast_mass_ejected(total_mass, thres_vel, min_vel, beta) * sol_mass

    numerator = (((fast_mass) ** (4/3)) * opacity)
    denominator = ((4 * const.pi) * (total_mass ** (1/3)) * min_vel * t_elapsed * c)

    return (numerator / denominator) / day


lc_tpeak = peak_lightcurve_time(total_mass, beta, min_vel, opacity)
print(f"{lc_tpeak:.3e} days")

lc_fastmass = fast_mass_ejected(total_mass, thres_vel, min_vel, beta)
print(f"{lc_fastmass:.3e} solar masses (g)")

lc_diffusiontime = rad_diffusion_tscale(total_mass, thres_vel, min_vel, beta, t_elapsed, opacity)
print(f"{lc_diffusiontime:.3e} days")#


t_plot = np.linspace(0, 3*lc_tpeak, 600)
M_v = np.where(t_plot < lc_tpeak, total_mass * (t_plot / lc_tpeak)**1.5, total_mass)
mass_solar = total_mass / sol_mass

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(t_plot, M_v / sol_mass, lw=2)
ax.axvline(lc_tpeak, ls="--", c="gray")
ax.text(lc_tpeak*1.03, 0.05*mass_solar, f"t_peak = {lc_tpeak:.2f} d", color="gray")
ax.axhline(mass_solar, ls=":", c="gray")
ax.set_xlabel("t (days)")
ax.set_ylabel(r"$M_v$ ($M_\odot$)")
ax.set_title("Eq. 12: mass layer the light escapes from vs time")
ax.set_xlim(0, t_plot.max()); ax.set_ylim(0, 1.1*mass_solar)
fig.tight_layout()
fig.savefig(output_dir / "mass_layer_v_time.png", dpi=180)

#----------------------------------------------------------------------------------------#

##Eq for Drv / dt = v

# Calculation below divides velocity into logarithmic bins stopping at max velocity
# The upper cutoff keeps this first Newtonian calculation below relativistic 
# (0.3c taken as in the paper it states the usual dynamical range)

max_vel = 0.2 * c
n_shells = 48
r_initial = 1.0e7  

velocity_edges = np.geomspace(min_vel, max_vel, n_shells + 1)
mass_above_edges = fast_mass_ejected(total_mass, velocity_edges, min_vel, beta) * sol_mass

shell_masses = mass_above_edges[:-1] - mass_above_edges[1:]
velocities = np.sqrt(velocity_edges[:-1] * velocity_edges[1:])

#time array to capture times, ranging to max of 1 day with 300 time steps
radius_times = np.linspace(0.0, max(3.0 * lc_tpeak * day, day), 301) 
shell_radii = np.empty((n_shells, len(radius_times))) #unfilled array with 1 row and column for each shell
shell_radii[:, 0] = r_initial #sets radius at first time point to the minimmum for all shells

# Backward Euler for each shell:
#   R[n+1] - R[n] - dt*v = 0.

for n, dt in enumerate(np.diff(radius_times)):
    r_old = shell_radii[:, n]                               #All shells radius at current time
    r_new = r_old.copy()                                    #Sets initial guess for next radii
    for _ in range(2):
        residual = r_new - r_old - dt * velocities          #how far off each guess is
        r_new -= residual                                   # derivative w.r.t r_new is 1 therefor gives exact new radius
        if np.max(np.abs(residual)) < 1.0e-10 * max(1.0, np.max(np.abs(r_new))):     #if even the worst shell’s residual is below the tolerance
            break
    shell_radii[:, n + 1] = r_new #save updated radii to next

fig, ax = plt.subplots(figsize=(7, 4.5))

color_map = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(min_vel / c, max_vel / c))
color_map.set_array([velocities / c])
colors = color_map.to_rgba(velocities / c)
colorbar = fig.colorbar(color_map, ax=ax)

for i in range(n_shells):
    ax.plot(radius_times / day, shell_radii[i] / sol_radius, color=colors[i], lw=1.1)

colorbar.set_label("Shell velocity (c)")
ax.set_xlabel("t (days)")
ax.set_ylabel(r"$R_v$ ($R_\odot$)")
ax.set_title("Implicit expansion of all ejecta mass shells")
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(output_dir / "shell_radii_v_time.png", dpi=180)

print(f"Number of velocity/mass shells: {n_shells}")
print(f"Mass represented: {np.sum(shell_masses) / sol_mass:.3e} solar masses")
print(f"Fraction of total mass represented: {np.sum(shell_masses) / total_mass:.3%}")
print("Radius plot saved as shell_radii_v_time.png")


 #---------------------------------------------------------------------------------#

# Thermal-energy evolution for every shell (Eqs. 14-16)
# Eq. (15): L_v = E_v / (t_diff,v + t_lc,v).
# Eq. (16): Qdot = Qdot_r,v + Qdot_mag + Qdot_fb.
# This first version includes r-process heating; magnetar and fallback terms are set to zero until those source models are added.

x_r = 1.0 # r-process mass {as a fraction}
eta_th = 0.5  # Fraction od decay that heats eject (simplified for now as can change with time)


def r_process_specific_heating(t):
    #Paper Eq. (22), thermalized heating rate in erg s^-1 g^-1 for red novae
    t0 = 1.3  # in s
    sigma = 0.11  # in s
    shape = 0.5 - np.arctan((t - t0) / sigma) / np.pi
    return 4.0e18 * eta_th * shape**1.3


def implicit_energy_step(energy_old, dt, velocity, radius_new,
                         t_diff_new, t_light_new, heating_power_new):
    # Implicit update of Eq. (14), applied to all shells

    loss_coefficient = velocity / radius_new + 1.0 / (t_diff_new + t_light_new)
    return (energy_old + dt * heating_power_new) / (1.0 + dt * loss_coefficient)


shell_energies = np.empty_like(shell_radii)
shell_energies[:, 0] = 0.5 * shell_masses * velocities**2                  #initial thermal energy approximated to kinetic in the paper
shell_luminosities = np.empty_like(shell_radii)


mass_above_shell = fast_mass_ejected(total_mass, velocities, min_vel, beta) * sol_mass # in g

# Evaluating the initial luminosity for completeness (plots omit t=0 on the logarithmic axis).
r_initial_all = shell_radii[:, 0]
t_diff_initial = 3.0 * mass_above_shell * opacity / (4.0 * const.pi * beta * r_initial_all * c)
t_light_initial = r_initial_all / c
shell_luminosities[:, 0] = shell_energies[:, 0] / (t_diff_initial + t_light_initial)          #applies luminosity to every shell based off initial E adn timescales

for n, dt in enumerate(np.diff(radius_times)):
    t_new = radius_times[n + 1]
    r_new = shell_radii[:, n + 1]                   #calculated in prior part

    # Diffusion and light-crossing times at the new time, (Eq 11).
    t_diff_new = 3.0 * mass_above_shell * opacity / (4.0 * const.pi * beta * r_new * c)
    t_light_new = r_new / c 

    # Eq 21): radioactive power  = shell mass * X_r * specific heating.
    qdot_radioactive = shell_masses * x_r * r_process_specific_heating(t_new)
    qdot_magnetar = np.zeros(n_shells)
    qdot_fallback = np.zeros(n_shells)
    heating_power_new = qdot_radioactive + qdot_magnetar + qdot_fallback

    #calculates energy and luminosity for each shell 
    shell_energies[:, n + 1] = implicit_energy_step(shell_energies[:, n], dt, velocities, r_new, t_diff_new, t_light_new, heating_power_new)
    shell_luminosities[:, n + 1] = shell_energies[:, n + 1] / (t_diff_new + t_light_new)

# Plot the energy stored in every shell, then the summed and sample shell luminosities. The radius plot above uses the same shell colors.
fig, ax = plt.subplots(figsize=(7, 4.5))
for i in range(n_shells):
    ax.loglog(radius_times[1:] / day, shell_energies[i, 1:], color=colors[i], lw=1.0)
ax.set_xlabel("t (days)")
ax.set_ylabel(r"Shell thermal energy $E_v$ (erg)")
ax.set_title("Implicit evolution of shell thermal energy")
ax.grid(True, which="both", alpha=0.25)
fig.tight_layout()
fig.savefig(output_dir / "shell_energy_v_time.png", dpi=180)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.loglog(radius_times[1:] / day, np.sum(shell_luminosities[:, 1:], axis=0), color="black", lw=2.2, label="Total luminosity")
for i in [0, n_shells // 4, n_shells // 2, 3 * n_shells // 4, n_shells - 1]:
    ax.loglog(radius_times[1:] / day, shell_luminosities[i, 1:], color=colors[i], lw=1.0)
ax.set_xlabel("t (days)")
ax.set_ylabel(r"Luminosity $L_v$ (erg s$^{-1}$)")
ax.set_title("Luminosity from the evolving ejecta shells")
ax.grid(True, which="both", alpha=0.25)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(output_dir / "bolometric_luminosity_v_time.png", dpi=180)

print("Energy plot saved as shell_energy_v_time.png")
print("Luminosity plot saved as bolometric_luminosity_v_time.png")


#----------------------------------------------------------------------#

# Calculation of effective temperature for the thermal emmision eq(18)

# L_tot already calculated in prior in luminosity plot

shell_tau_contributions = (opacity * shell_masses[:, None] / (4.0 * const.pi * shell_radii**2))
tau_outward = (np.cumsum(shell_tau_contributions[::-1, :], axis=0)[::-1, :] - 0.5 * shell_tau_contributions)

# Locate the tau=1 crossing between neighbouring shell centres at each time
photosphere_radius = np.full(len(radius_times), np.nan)
for j in range(len(radius_times)):
    tau_at_time = tau_outward[:, j]

    if tau_at_time[-1] >= 1.0:                 # The crossing is inside the fastest modeled shell, between its centre and the outer boundary, where the model has tau = 0.
        fraction = 1.0 - (1.0 / tau_at_time[-1])
        outer_boundary_radius = r_initial + max_vel * radius_times[j]
        photosphere_radius[j] = (shell_radii[-1, j] + fraction * (outer_boundary_radius - shell_radii[-1, j]))
    else:
        crossings = np.flatnonzero((tau_at_time[:-1] >= 1.0) & (tau_at_time[1:] < 1.0))
        if crossings.size > 0:
            i = crossings[0]
            tau_inner = tau_at_time[i]
            tau_outer = tau_at_time[i + 1]

            # Interpolate in log optical depth between the shell centres
            fraction = np.log(tau_inner) / np.log(tau_inner / tau_outer)
            photosphere_radius[j] = (shell_radii[i, j] + fraction * (shell_radii[i + 1, j] - shell_radii[i, j]))

# Eq. (18)
# Stefan-Boltzmann constant in CGS units: erg s^-1 cm^-2 K^-4

sigma_sb = 5.670e-5
total_luminosity = np.sum(shell_luminosities, axis=0)

effective_temperature = np.full(len(radius_times), np.nan)
valid_photosphere = np.isfinite(photosphere_radius) & (photosphere_radius > 0.0)
effective_temperature[valid_photosphere] = (total_luminosity[valid_photosphere] / (4.0 * const.pi * sigma_sb * photosphere_radius[valid_photosphere]**2)) ** 0.25

# A missing point means the tau=1 surface lies outside the modeled shell range or the modeled ejecta is optically thin throughout
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(radius_times / day, photosphere_radius / sol_radius, lw=2.0)
ax.set_xlabel("t (days)")
ax.set_ylabel(r"Photosphere radius $R_{ph}$ ($R_\odot$)")
ax.set_title("Photosphere location from outward optical depth")
ax.grid(True, alpha=0.25)
fig.tight_layout()
plt.savefig("photosphere_radius_v_time.png", dpi=180)

fig, ax = plt.subplots(figsize=(7, 4.5))
valid_temperature = valid_photosphere & (effective_temperature > 0.0) & (radius_times > 0.0)
ax.loglog(radius_times[valid_temperature] / day, effective_temperature[valid_temperature], lw=2.0)
ax.set_xlabel("t (days)")
ax.set_ylabel(r"Effective temperature $T_{eff}$ (K)")
ax.set_title("Blackbody effective temperature")
ax.grid(True, which="both", alpha=0.25)
fig.tight_layout()
plt.savefig("effective_temperature_v_time.png", dpi=180)

print("Photosphere plot saved as photosphere_radius_v_time.png")
print("Temperature plot saved as effective_temperature_v_time.png")

#------------------------------------------------#
# Frequency-dependent flux density from the blackbody photosphere (Eq. 19)
# Eq. (19) gives the flux received at Earth at one photon frequency 

# Repeating it for several representative band wavelengths gives light curves, monochromatic approximations to the broad photometric bands.
distance_mpc = 100.0  # Fig 9 value
mpc_to_cm = 3.0857e24
source_distance = distance_mpc * mpc_to_cm

# converting h and k from joules to ergs so they match the model's CGS distances
h_cgs = const.h * 1.0e7  # erg s
k_b_cgs = const.k * 1.0e7  # erg K^-1

# Approximate central wavelengths for common photometric bands, in nm.
band_wavelengths_nm = {"U": 365.0, "B": 445.0, "V": 551.0, "R": 658.0, "I": 806.0, "J": 1220.0, "H": 1630.0, "K": 2190.0,}

# Convert each representative wavelength to its corresponding frequency in cm
band_frequencies = {band: c / (wavelength_nm * 1.0e-7) for band, wavelength_nm in band_wavelengths_nm.items()}

# Only calculate the spectrum where the photosphere and temperature are valid; omit t=0 because the initial photosphere/temperature is not defined
valid_flux_times = ((radius_times > 0.0) & valid_photosphere & np.isfinite(effective_temperature) & (effective_temperature > 0.0))

flux_times_days = radius_times[valid_flux_times] / day            # keep only valid entries
temperature_for_flux = effective_temperature[valid_flux_times]
photosphere_for_flux = photosphere_radius[valid_flux_times]

# Rows correspond to bands; columns correspond to the valid time points.
flux_density = np.empty((len(band_frequencies), len(flux_times_days)))
band_names = list(band_frequencies)

for i, band in enumerate(band_names):
    nu = band_frequencies[band]
    exponent = h_cgs * nu / (k_b_cgs * temperature_for_flux)

    # Planck spectrum times the apparent photosphere area (Eq. 19).
    flux_density[i] = ((2.0 * const.pi * h_cgs * nu**3 / c**2) / np.expm1(exponent) * (photosphere_for_flux**2 / source_distance**2))

# Plot the received flux density at each representative band frequency.
fig, ax = plt.subplots(figsize=(7, 4.5))
for i, band in enumerate(band_names):
    positive_flux = flux_density[i] > 0.0
    ax.loglog(flux_times_days[positive_flux], flux_density[i, positive_flux], label=band)
ax.set_xlabel("t (days)")
ax.set_ylabel(r"Flux density $F_\nu$ (erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$)")
ax.set_title(f"Blackbody flux-density light curves at {distance_mpc:.0f} Mpc")
ax.grid(True, which="both", alpha=0.25)
ax.legend(title="Band", frameon=False, ncol=2)
fig.tight_layout()
plt.savefig("flux_density_light_curves.png", dpi=180)

# Convert F_nu to AB magnitudes for a plot comparable to paper Fig. 9.
# The conversion requires F_nu in erg s^-1 cm^-2 Hz^-1.
ab_magnitude = -2.5 * np.log10(flux_density) - 48.60

fig, ax = plt.subplots(figsize=(7, 4.5))
for i, band in enumerate(band_names):
    finite_magnitude = np.isfinite(ab_magnitude[i])
    ax.plot(flux_times_days[finite_magnitude], ab_magnitude[i, finite_magnitude], label=band)
ax.set_xlabel("t (days)")
ax.set_ylabel("AB magnitude")
ax.set_title(f"Blackbody light curves at {distance_mpc:.0f} Mpc")
ax.invert_yaxis()  # brighter magnitudes are plotted higher, as in astronomy
ax.grid(True, alpha=0.25)
ax.legend(title="Band", frameon=False, ncol=2)
fig.tight_layout()
plt.savefig("ab_magnitude_light_curves.png", dpi=180)

print("Flux-density plot saved as flux_density_light_curves.png")
print("AB-magnitude plot saved as ab_magnitude_light_curves.png")