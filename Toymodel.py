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
thres_vel = (0.2 * c)
total_mass = (0.02 * sol_mass)
opacity = (10.0)
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
