OPENQASM 2.0;
include "qelib1.inc";
gate ryy(param0) q0,q1 { sxdg q0; sxdg q1; cx q0,q1; rz(param0) q1; cx q0,q1; sx q0; sx q1; }
gate xx_plus_yy(param0,param1) q0,q1 { rz(param1) q0; sdg q1; sx q1; s q1; s q0; cx q1,q0; ry((-0.5)*param0) q1; ry((-0.5)*param0) q0; cx q1,q0; sdg q0; sdg q1; sxdg q1; s q1; rz(-param1) q0; }
gate inner(t) a,b { crz(2*t) a,b; barrier a,b; id b; }
gate outer(t,u) a,b,c { h a; inner(t/u) a,c; ccx a,b,c; ry(-t^2) b; }
gate nop a { }
qreg q[3];
qreg anc[1];
creg c[4];
ryy(0.3) q[2],q[0];
xx_plus_yy(pi/4,-0.1) q[1],anc[0];
outer(0.5,2) q[0],q[1],q[2];
outer(1,3) q[0],q[1],anc;
nop anc;
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure anc[0] -> c[3];
