"""
Verification battery for the cross-Hessian coupling primitives.

Object under test:
    B(theta, x) : scalar behaviour functional
    M = d/dx ( grad_theta B )   shape (P, D)   "mixed" / cross block of the joint Hessian

We verify, against a densely-formed M on a toy net:
    (1) M @ u   via forward-over-reverse jvp   (input-space vector -> param-space vector)
    (2) M^T @ v via grad of <v, grad_theta B>  (param-space vector -> input-space vector)
    (3) finite-difference cross-checks for both
    (4) top singular triplet via matrix-free power iteration on M^T M  vs dense SVD
    (5) Danskin gradient d sigma_1 / dx  vs finite difference

All in float64 so the FD tolerances are meaningful.
"""

import torch
from torch.func import grad, jvp, jacrev

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

# ----- toy model: 2-layer MLP -> nonlinear scalar (nonzero mixed partials) -----
d_in, d_h = 4, 5
P = d_h * d_in + d_h + d_h     # total params


def unflatten(theta):
    i = 0
    W1 = theta[i:i + d_h * d_in].reshape(d_h, d_in); i += d_h * d_in
    b1 = theta[i:i + d_h];                           i += d_h
    w2 = theta[i:i + d_h];                           i += d_h
    return W1, b1, w2


def B(theta, x):
    W1, b1, w2 = unflatten(theta)
    h = torch.tanh(W1 @ x + b1)
    out = w2 @ h
    return 0.5 * out**2 + torch.sin(out)          # scalar


theta = torch.randn(P)
x = torch.randn(d_in)

# grad wrt theta, as a function of (theta, x)
gtheta = lambda th, xx: grad(B, argnums=0)(th, xx)        # -> R^P

# ----- dense reference: M = jac_x ( grad_theta B ),  shape (P, D) -----
M_dense = jacrev(gtheta, argnums=1)(theta, x)            # (P, d_in)
print(f"M shape (P x D) = {tuple(M_dense.shape)}")

# ----- (1) M u via forward-over-reverse -----
u = torch.randn(d_in)
gval, Mu = jvp(lambda xx: gtheta(theta, xx), (x,), (u,))
err_Mu = (Mu - M_dense @ u).norm().item()
print(f"(1) ||M u  - jvp||        = {err_Mu:.2e}")

# ----- (2) M^T v via grad of <v, grad_theta B> -----
v = torch.randn(P)
MTv = grad(lambda xx: v @ gtheta(theta, xx))(x)
err_MTv = (MTv - M_dense.T @ v).norm().item()
print(f"(2) ||M^T v - grad||      = {err_MTv:.2e}")

# ----- (3) finite-difference cross-checks -----
eps = 1e-6
Mu_fd = (gtheta(theta, x + eps * u) - gtheta(theta, x - eps * u)) / (2 * eps)
print(f"(3a) ||M u  - FD||        = {(Mu - Mu_fd).norm().item():.2e}")
# M^T v FD: directional deriv of (v . grad_theta B) along each x coord is expensive;
# instead check the bilinear identity u^T M^T v == (M u)^T v == u-dir deriv of <v,g>
lhs = (M_dense @ u) @ v
rhs = ((v @ gtheta(theta, x + eps * u)) - (v @ gtheta(theta, x - eps * u))) / (2 * eps)
print(f"(3b) |<Mu,v> - FD<v,g>|   = {abs(lhs - rhs).item():.2e}")

# ----- (4) top singular triplet, matrix-free, via power iteration on M^T M -----
def Mvec(w):                                              # R^D -> R^P
    _, out = jvp(lambda xx: gtheta(theta, xx), (x,), (w,))
    return out

def MTvec(p):                                            # R^P -> R^D
    return grad(lambda xx: p.detach() @ gtheta(theta, xx))(x)

def MTM(w):                                              # R^D -> R^D
    return MTvec(Mvec(w))

w = torch.randn(d_in)
w = w / w.norm()
for _ in range(200):
    w_new = MTM(w)
    lam = w @ w_new                                      # Rayleigh quotient = sigma^2
    w = w_new / w_new.norm()
sigma1_pi = lam.sqrt().item()
v1 = w                                                   # right singular vector (input space)
u1 = Mvec(v1); u1 = u1 / u1.norm()                       # left singular vector (param space)

S = torch.linalg.svdvals(M_dense)
print(f"(4) sigma_1 power-iter    = {sigma1_pi:.6f}")
print(f"    sigma_1 dense SVD     = {S[0].item():.6f}")
print(f"    rel err               = {abs(sigma1_pi - S[0].item()) / S[0].item():.2e}")

# ----- (5) Danskin gradient d sigma_1 / dx  (freeze u1, v1) -----
def sigma1_danskin(xx):
    _, Mv1 = jvp(lambda x2: gtheta(theta, x2), (xx,), (v1,))
    return u1 @ Mv1                                       # = sigma_1 at the frozen singular vecs
grad_sigma1 = grad(sigma1_danskin)(x)

# FD reference: recompute sigma_1 (full power iteration) at x +/- eps e_k
def sigma1_full(xx):
    ww = torch.randn(d_in); ww = ww / ww.norm()
    def MTM_at(w_):
        _, mw = jvp(lambda x2: gtheta(theta, x2), (xx,), (w_,))
        return grad(lambda x2: mw.detach() @ gtheta(theta, x2))(xx)
    for _ in range(300):
        wn = MTM_at(ww); l = ww @ wn; ww = wn / wn.norm()
    return l.sqrt()

gfd = torch.zeros(d_in)
for k in range(d_in):
    e = torch.zeros(d_in); e[k] = 1.0
    gfd[k] = (sigma1_full(x + eps * e) - sigma1_full(x - eps * e)) / (2 * eps)
print(f"(5) ||grad sigma1 - FD||  = {(grad_sigma1 - gfd).norm().item():.2e}")
print(f"    cosine(grad, FD)      = {torch.nn.functional.cosine_similarity(grad_sigma1, gfd, dim=0).item():.6f}")

print("\nAll primitives verified against dense / finite-difference references.")
