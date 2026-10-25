import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
#c has to be in cm and masses in g
c = const.c * 100
sol_mass = 1.989e33
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

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(t_plot, M_v, lw=2)
ax.axvline(lc_tpeak, ls="--", c="gray")
ax.text(lc_tpeak*1.03, 0.05*total_mass, f"t_peak = {lc_tpeak:.2f} d", color="gray")
ax.axhline(total_mass, ls=":", c="gray")
ax.set_xlabel("t (days)")
ax.set_ylabel(r"$M_v$ ($M_\odot$)")
ax.set_title("Eq. 12: mass layer the light escapes from vs time")
ax.set_xlim(0, t_plot.max()); ax.set_ylim(0, 1.1*total_mass)
fig.tight_layout()
plt.savefig("mass_layer_v_time")

