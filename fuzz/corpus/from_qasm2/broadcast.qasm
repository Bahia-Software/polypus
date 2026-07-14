OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q;
rx(pi/2) q;
measure q -> c;
