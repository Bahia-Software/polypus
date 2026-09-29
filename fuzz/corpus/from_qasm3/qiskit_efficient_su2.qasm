OPENQASM 3.0;
include "stdgates.inc";
input float[64] _θ_0_;
input float[64] _θ_1_;
input float[64] _θ_2_;
input float[64] _θ_3_;
input float[64] _θ_4_;
input float[64] _θ_5_;
input float[64] _θ_6_;
input float[64] _θ_7_;
gate r(p0, p1) _gate_q_0 {
  U(p0, -pi/2 + p1, pi/2 - p1) _gate_q_0;
}
qubit[2] q;
r(_θ_0_, pi/2) q[0];
p(_θ_2_) q[0];
r(_θ_1_, pi/2) q[1];
p(_θ_3_) q[1];
cx q[0], q[1];
r(_θ_4_, pi/2) q[0];
p(_θ_6_) q[0];
r(_θ_5_, pi/2) q[1];
p(_θ_7_) q[1];
