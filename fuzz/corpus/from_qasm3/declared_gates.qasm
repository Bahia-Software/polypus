OPENQASM 3.0;
include "stdgates.inc";
input float[64] θ;
input float β;
input float[64] unused;
gate rot(a, b) w {
  U(a, -pi/2 + b, pi/2 - b) w;
}
gate layer(g) u, v {
  rot(g, 2*g) u;
  cx u, v;
  rz(g**2/2.0) v;
  cx u, v;
}
gate empty _q {
}
qubit[2] q;
layer(θ*β) q[0], q[1];
layer(-θ) q[1], q[0];
empty q[0];
rot(β, pi/4) q[1];
