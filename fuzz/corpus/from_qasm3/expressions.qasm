OPENQASM 3.0;
include "stdgates.inc";
input float[64] alpha;
input float[64] beta;
qubit[1] q;
// Constants, functions and literal forms.
rx(π + τ - ℇ) q[0];
ry(sin(alpha) * cos(beta) + tan(0.5)) q[0];
rz(arcsin(0.5) + arccos(alpha/4.0) - arctan(beta)) q[0];
p(exp(-alpha) + log(2.0) + sqrt(beta**2 + 1)) q[0];
rx(0x1F + 0o17 + 0b101 + 1_000 + 1.5e-3 + .5 + 1.) q[0];
/* Power binds tighter than unary minus
   and is right-associative. */
ry(-alpha ** 2 ** 0.5 + (-beta) ** 2) q[0];
rz(((alpha - beta) - (alpha - (beta - 1.0))) / (alpha * beta + 1.0)) q[0];
