OPENQASM 3.0;
include "stdgates.inc";
input float[64] theta_0;
input float[64] theta_1;
gate rzz(p0) a, b {
  cx a, b;
  rz(p0) b;
  cx a, b;
}
gate rxx(p0) a, b {
  h a;
  h b;
  cx a, b;
  rz(p0) b;
  cx a, b;
  h a;
  h b;
}
gate sxdg a {
  h a;
  sdg a;
  h a;
}
gate csx a, b {
  h b;
  cp(pi/2.0) a, b;
  h b;
}
gate cu1(p0) a, b {
  cp(p0) a, b;
}
gate cu3(p0, p1, p2) a, b {
  cu(p0, p1, p2, 0.0) a, b;
}
gate u0(p0) a {
}
gate rccx a, b, c_1 {
  h c_1;
  t c_1;
  cx b, c_1;
  tdg c_1;
  cx a, c_1;
  t c_1;
  cx b, c_1;
  tdg c_1;
  h c_1;
}
gate rc3x a, b, c_1, d {
  h d;
  t d;
  cx c_1, d;
  tdg d;
  h d;
  cx a, d;
  t d;
  cx b, d;
  tdg d;
  cx a, d;
  t d;
  cx b, d;
  tdg d;
  h d;
  t d;
  cx c_1, d;
  tdg d;
  h d;
}
gate c3x a, b, c_1, d {
  h d;
  p(pi/8.0) a;
  p(pi/8.0) b;
  p(pi/8.0) c_1;
  p(pi/8.0) d;
  cx a, b;
  p(-pi/8.0) b;
  cx a, b;
  cx b, c_1;
  p(-pi/8.0) c_1;
  cx a, c_1;
  p(pi/8.0) c_1;
  cx b, c_1;
  p(-pi/8.0) c_1;
  cx a, c_1;
  cx c_1, d;
  p(-pi/8.0) d;
  cx b, d;
  p(pi/8.0) d;
  cx c_1, d;
  p(-pi/8.0) d;
  cx a, d;
  p(pi/8.0) d;
  cx c_1, d;
  p(-pi/8.0) d;
  cx b, d;
  p(pi/8.0) d;
  cx c_1, d;
  p(-pi/8.0) d;
  cx a, d;
  h d;
}
gate c3sqrtx a, b, c_1, d {
  h d;
  cp(pi/8.0) a, d;
  h d;
  cx a, b;
  h d;
  cp(-pi/8.0) b, d;
  h d;
  cx a, b;
  h d;
  cp(pi/8.0) b, d;
  h d;
  cx b, c_1;
  h d;
  cp(-pi/8.0) c_1, d;
  h d;
  cx a, c_1;
  h d;
  cp(pi/8.0) c_1, d;
  h d;
  cx b, c_1;
  h d;
  cp(-pi/8.0) c_1, d;
  h d;
  cx a, c_1;
  h d;
  cp(pi/8.0) c_1, d;
  h d;
}
gate c4x a, b, c_1, d, e {
  h e;
  cp(pi/2.0) d, e;
  h e;
  c3x a, b, c_1, d;
  h e;
  cp(-pi/2.0) d, e;
  h e;
  c3x a, b, c_1, d;
  c3sqrtx a, b, c_1, e;
}
qubit[5] q;
bit[2] c;
rzz(theta_0) q[0], q[1];
rxx(0.5) q[1], q[2];
sxdg q[2];
csx q[3], q[0];
cu1(theta_1) q[0], q[4];
cu3(0.1, theta_0, -0.3) q[4], q[3];
u0(1.0) q[2];
rccx q[0], q[1], q[2];
rc3x q[3], q[2], q[1], q[0];
c3x q[1], q[2], q[3], q[4];
c3sqrtx q[4], q[3], q[2], q[1];
c4x q[0], q[1], q[2], q[3], q[4];
c[0] = measure q[0];
c[1] = measure q[4];
