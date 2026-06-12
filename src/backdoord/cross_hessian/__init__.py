"""Cross-Hessian coupling detection: input-parameter coupling as a backdoor signature.

A hidden behaviour is conditional computation (a trigger-detector gating a payload). Its
signature is coupling between inputs and parameters: the off-diagonal block
``M = d/dx(grad_theta B)`` of the joint Hessian of a behaviour functional ``B(theta, x)``.
A backdoor makes ``M`` low-rank and concentrated; a clean model's ``M`` is diffuse. ``M``
(P x D) is never materialised — everything is matrix-free vector products. See
``plans/cross_hessian_spec.md`` for the full design.
"""
