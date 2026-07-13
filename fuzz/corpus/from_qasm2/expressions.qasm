OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
rx(pi/2) q[0];
ry(-pi/4) q[0];
rz(2*pi) q[0];
u3(1.5e-3, (1+2)*pi, cos(0)) q[0];
rx(2^3) q[0];
