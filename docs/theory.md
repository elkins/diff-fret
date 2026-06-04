# FRET Theory

## Förster Resonance Energy Transfer

FRET efficiency $E$ depends on the distance $r$ between a donor and an acceptor fluorophore according to the Förster equation:

$$E = \frac{1}{1 + (r/R_0)^6}$$

where $R_0$ is the Förster distance at which the transfer efficiency is 50%.

## The Orientation Factor ($\kappa^2$)

The Förster distance $R_0$ depends on the relative orientation of the donor and acceptor transition dipoles, described by the orientation factor $\kappa^2$:

$$\kappa^2 = (\cos \theta_T - 3 \cos \theta_D \cos \theta_A)^2$$

While $\kappa^2 = 2/3$ is often assumed for randomly oriented dyes, **diff-fret** implements the **Dale-Eisinger-Blumberg (1979)** model to estimate the upper and lower bounds of $\kappa^2$ based on measured fluorescence anisotropy $r_{obs}$:

$$\langle \kappa^2 \rangle_{max} = \frac{2}{3} (1 + d_D + d_A + 3 d_D d_A)$$
$$\langle \kappa^2 \rangle_{min} = \frac{2}{3} \left[ 1 - \frac{1}{2}(d_D + d_A) \right]$$

where $d = \sqrt{r_{obs}/r_{0}}$ is the axial depolarization factor.
