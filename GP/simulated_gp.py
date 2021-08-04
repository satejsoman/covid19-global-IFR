import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
import vapeplot as vp
from tqdm import trange

import pymc3 as pm
from pymc3 import gp
from pymc3.gp.util import plot_gp_dist
import arviz as az

# -----------------------------
# setup
# -----------------------------

# generate prevelance curves
def prevalence(x, seed = None, S0 = 0.2, lw = 0.01, nw = 0.5, ew = 0.00001):
    if seed is not None:
        np.random.seed(seed)
    return ( S0 
        + lw * 1/(1 + np.exp(-(x - 50)/10)) 
        + nw * scipy.stats.norm.pdf(x, loc = 50, scale = 5) 
        + ew * np.random.randn(x.size)
    )

# plotting options 
sns.set_theme(style = "white", font = "Univers LT Std")
palette = "sunset"
vp.set_palette(palette)

# example data
## "true" prevalence over ages 0 - 100
x_smooth = np.linspace(0, 100, 1001)
S_smooth = prevalence(x_smooth, seed = 0)

x        = x_smooth[::10]
S        = S_smooth[::10]

# age structure
age_weights_5y = np.sort(np.random.uniform(0, 1, 21))[::-1]
rho_5 = age_weights_5y / age_weights_5y.sum()
rho_1 = np.piecewise(x, [(l <= x) & (x < h) for (l, h) in (zip(range(0, 100, 5), range(5, 105, 5)))], rho_5 + [0]) / 5

# observed age bins
age_bin_breaks = [0, 18, 30, 40, 50, 60, 70, 100]
bins           = list(zip(age_bin_breaks[:-1], age_bin_breaks[1:]))
median_ages    = [int(np.median(_)) for _ in  bins]
R              = np.vstack([np.concatenate([np.zeros(l), rho_1[l:h]/rho_1[l:h].sum(), np.zeros(101 - h)]) for (l, h) in bins])
P_levels       = R @ S

##observed prevalence assuming representative samples
P = np.piecewise(x, [(l <= x) & (x < h) for (l, h) in bins], P_levels)

# -----------------------------
# initial plots
# -----------------------------

# plot true prevelance
# plt.plot(x_smooth, S_smooth)
# plt.xlim(0, 100)
# plt.gca().ticklabel_format(useOffset=False)
# plt.show()

fig, (ax1, ax2) = plt.subplots(2, 1, sharex = "col")

# plot population density
plt.sca(ax1)
plt.bar(range(0, 101, 5), rho_5, width = 4.5, label = "5 year age bins", align = "edge", color = vp.palette("cool")[-1])
plt.bar(x,                rho_1, width = 0.9, label = "1 year age bins", align = "edge", color = vp.palette("cool")[-2])
plt.ylabel("proportion of population\n", fontdict = {"size": 16})
plt.title("age distribution", loc = "left", fontdict = {"size": 20, "fontweight": "bold"})
plt.legend(handlelength = 1)
sns.despine()

# plot observed prevalence
plt.sca(ax2)
vp.set_palette("sunset")
plt.plot(x_smooth, S_smooth, label = "true prevalence", zorder = -1, alpha = 0.5)
c = vp.palette(palette)[1]
plt.scatter([l for (l, _) in bins], P_levels, edgecolors = "white", facecolors = "white")
plt.scatter([h for (_, h) in bins], P_levels, edgecolors = "white", facecolors = "none")
for (i, ((l, h), P)) in enumerate(zip(bins, P_levels)):
    plt.plot([l, h], [P, P], color = "white", linewidth = 3, zorder = 0)
    plt.scatter(l, P, edgecolors = c, facecolors = c,       zorder = 1)
    plt.plot([l, h], [P, P], color = c, label = "observed age-bin prevalence" if i == 0 else None, zorder = 2)
    plt.scatter(h, P, edgecolors = c, facecolors = "white", zorder = 3)
plt.xlim(-2, 102)
plt.ylim(.19, .25)
plt.legend(handlelength = 1)
sns.despine()
yticks = plt.yticks()[0]
plt.yticks(ticks = yticks, labels = [f"{int(100 * t)}%" for t in yticks])
plt.xlabel("\nage", fontdict = {"size": 16})
plt.ylabel("prevalence\n", fontdict = {"size": 16})
plt.title("underlying vs. observed prevalence, as functions of age", loc = "left", fontdict = {"size": 20, "fontweight": "bold"})
plt.show()



# -----------------------------
# modeling
# -----------------------------

μ0 = 0
X = x.reshape((101, 1))
with pm.Model() as model:
    S_gp = gp.Latent(
        gp.mean.Constant(μ0), 
        gp.cov.ExpQuad(input_dim = 1, ls = 100)
    )
    # S_prior = S_gp.prior("S_prior", X = np.array(median_ages).reshape(7, 1))
    # Ψ = pm.Normal("Ψ",
    #     mu = pm.math.dot(R.T, S_prior),
    #     sigma = 1, 
    #     observed = np.array(P_levels).reshape((7, 1))
    # )
    S_prior = S_gp.prior("S_prior", X = X)
    Ψ = pm.Normal("Ψ",
        mu = S_prior,
        sigma = 1, 
        observed = P
    )
    S_postr = S_gp.conditional("S_postr", X)
    trace = pm.sample(1000, return_inferencedata = True) 


# Sample from the GP conditional distribution
with model:
    prior_samples = pm.sample_prior_predictive    (var_names = ["S_prior"])
    postr_samples = pm.sample_posterior_predictive(var_names = ["S_postr"], trace = trace.posterior)

res = az.summary(trace)
plot_gp_dist(plt.gca(), trace.posterior["S_prior"][0, :, :], np.array(median_ages), plot_samples = False)
plt.show()
plot_gp_dist(plt.gca(), trace.posterior["S_postr"][0, :, :], X, plot_samples = False)
plt.show()