from sympy import *

q1, q2, q3, q4, q5, q6 = symbols('q1 q2 q3 q4 q5 q6')

d1, a2, a3, d4, d5, d6 = symbols('d1 a2 a3 d4 d5 d6')

s1 = sin(q1) 
c1 = cos(q1)

q234 = q2+q3+q4 
s2 = sin(q2)
c2 = cos(q2)
s3 = sin(q3)
c3 = cos(q3)
s5 = sin(q5)
c5 = cos(q5)
s6 = sin(q6)
c6 = cos(q6) 
s234 = sin(q234)
c234 = cos(q234)

nx = ((c1*c234-s1*s234)*s5)/2.0 - c5*s1 + ((c1*c234+s1*s234)*s5)/2.0

ox = (c6*(s1*s5 + ((c1*c234-s1*s234)*c5)/2.0 + ((c1*c234+s1*s234)*c5)/2.0) - 
          (s6*((s1*c234+c1*s234) - (s1*c234-c1*s234)))/2.0)
          
ax = (-(c6*((s1*c234+c1*s234) - (s1*c234-c1*s234)))/2.0 - 
          s6*(s1*s5 + ((c1*c234-s1*s234)*c5)/2.0 + ((c1*c234+s1*s234)*c5)/2.0))
          
px = ((d5*(s1*c234-c1*s234))/2.0 - (d5*(s1*c234+c1*s234))/2.0 - 
          d4*s1 + (d6*(c1*c234-s1*s234)*s5)/2.0 + (d6*(c1*c234+s1*s234)*s5)/2.0 - 
          a2*c1*c2 - d6*c5*s1 - a3*c1*c2*c3 + a3*c1*s2*s3)        
          
ny = c1*c5 + ((s1*c234+c1*s234)*s5)/2.0 + ((s1*c234-c1*s234)*s5)/2.0

oy = (c6*(((s1*c234+c1*s234)*c5)/2.0 - c1*s5 + ((s1*c234-c1*s234)*c5)/2.0) + 
          s6*((c1*c234-s1*s234)/2.0 - (c1*c234+s1*s234)/2.0))
          
ay = (c6*((c1*c234-s1*s234)/2.0 - (c1*c234+s1*s234)/2.0) - 
          s6*(((s1*c234+c1*s234)*c5)/2.0 - c1*s5 + ((s1*c234-c1*s234)*c5)/2.0))          
          
py = ((d5*(c1*c234-s1*s234))/2.0 - (d5*(c1*c234+s1*s234))/2.0 + d4*c1 + 
          (d6*(s1*c234+c1*s234)*s5)/2.0 + (d6*(s1*c234-c1*s234)*s5)/2.0 + d6*c1*c5 - 
          a2*c2*s1 - a3*c2*c3*s1 + a3*s1*s2*s3)
          
nz = ((c234*c5-s234*s5)/2.0 - (c234*c5+s234*s5)/2.0)

oz = ((s234*c6-c234*s6)/2.0 - (s234*c6+c234*s6)/2.0 - s234*c5*c6)

az = (s234*c5*s6 - (c234*c6+s234*s6)/2.0 - (c234*c6-s234*s6)/2.0)

pz = (d1 + (d6*(c234*c5-s234*s5))/2.0 + a3*(s2*c3+c2*s3) + a2*s2 - 
         (d6*(c234*c5+s234*s5))/2.0 - d5*c234)
         
##########
# [nx, ox, ax, px],
# [ny, oy, ay, py],
# [nz, oz, az, pz],						
# [0 , 0 , 0 , 1 ]         



tx1 = diff(px, q1)
tx2 = diff(px, q2)
tx3 = diff(px, q3)
tx4 = diff(px, q4)
tx5 = diff(px, q5)
tx6 = diff(px, q6)

ty1 = diff(py, q1)
ty2 = diff(py, q2)
ty3 = diff(py, q3)
ty4 = diff(py, q4)
ty5 = diff(py, q5)
ty6 = diff(py, q6)

tz1 = diff(pz, q1)
tz2 = diff(pz, q2)
tz3 = diff(pz, q3)
tz4 = diff(pz, q4)
tz5 = diff(pz, q5)
tz6 = diff(pz, q6)

#####################

nx1 = diff(nx, q1)
nx2 = diff(nx, q2)
nx3 = diff(nx, q3)
nx4 = diff(nx, q4)
nx5 = diff(nx, q5)
nx6 = diff(nx, q6)

ox1 = diff(ox, q1)
ox2 = diff(ox, q2)
ox3 = diff(ox, q3)
ox4 = diff(ox, q4)
ox5 = diff(ox, q5)
ox6 = diff(ox, q6)

ax1 = diff(ax, q1)
ax2 = diff(ax, q2)
ax3 = diff(ax, q3)
ax4 = diff(ax, q4)
ax5 = diff(ax, q5)
ax6 = diff(ax, q6)

ny1 = diff(ny, q1)
ny2 = diff(ny, q2)
ny3 = diff(ny, q3)
ny4 = diff(ny, q4)
ny5 = diff(ny, q5)
ny6 = diff(ny, q6)

oy1 = diff(oy, q1)
oy2 = diff(oy, q2)
oy3 = diff(oy, q3)
oy4 = diff(oy, q4)
oy5 = diff(oy, q5)
oy6 = diff(oy, q6)

ay1 = diff(ay, q1)
ay2 = diff(ay, q2)
ay3 = diff(ay, q3)
ay4 = diff(ay, q4)
ay5 = diff(ay, q5)
ay6 = diff(ay, q6)

nz1 = diff(nz, q1)
nz2 = diff(nz, q2)
nz3 = diff(nz, q3)
nz4 = diff(nz, q4)
nz5 = diff(nz, q5)
nz6 = diff(nz, q6)

oz1 = diff(oz, q1)
oz2 = diff(oz, q2)
oz3 = diff(oz, q3)
oz4 = diff(oz, q4)
oz5 = diff(oz, q5)
oz6 = diff(oz, q6)

az1 = diff(az, q1)
az2 = diff(az, q2)
az3 = diff(az, q3)
az4 = diff(az, q4)
az5 = diff(az, q5)
az6 = diff(az, q6)

#######################


print("tx1 =", tx1, "\n")
print("tx2 =", tx2, "\n")
print("tx3 =", tx3, "\n")
print("tx4 =", tx4, "\n")
print("tx5 =", tx5, "\n")
print("tx6 =", tx6, "\n")

print("ty1 =", ty1, "\n")
print("ty2 =", ty2, "\n")
print("ty3 =", ty3, "\n")
print("ty4 =", ty4, "\n")
print("ty5 =", ty5, "\n")
print("ty6 =", ty6, "\n")

print("tz1 =", tz1, "\n")
print("tz2 =", tz2, "\n")
print("tz3 =", tz3, "\n")
print("tz4 =", tz4, "\n")
print("tz5 =", tz5, "\n")
print("tz6 =", tz6, "\n")

#####################

print("nx1 =", nx1, "\n")
print("nx2 =", nx2, "\n")
print("nx3 =", nx3, "\n")
print("nx4 =", nx4, "\n")
print("nx5 =", nx5, "\n")
print("nx6 =", nx6, "\n")

print("ox1 =", ox1, "\n")
print("ox2 =", ox2, "\n")
print("ox3 =", ox3, "\n")
print("ox4 =", ox4, "\n")
print("ox5 =", ox5, "\n")
print("ox6 =", ox6, "\n")

print("ax1 =", ax1, "\n")
print("ax2 =", ax2, "\n")
print("ax3 =", ax3, "\n")
print("ax4 =", ax4, "\n")
print("ax5 =", ax5, "\n")
print("ax6 =", ax6, "\n")

print("ny1 =", ny1, "\n")
print("ny2 =", ny2, "\n")
print("ny3 =", ny3, "\n")
print("ny4 =", ny4, "\n")
print("ny5 =", ny5, "\n")
print("ny6 =", ny6, "\n")

print("oy1 =", oy1, "\n")
print("oy2 =", oy2, "\n")
print("oy3 =", oy3, "\n")
print("oy4 =", oy4, "\n")
print("oy5 =", oy5, "\n")
print("oy6 =", oy6, "\n")

print("ay1 =", ay1, "\n")
print("ay2 =", ay2, "\n")
print("ay3 =", ay3, "\n")
print("ay4 =", ay4, "\n")
print("ay5 =", ay5, "\n")
print("ay6 =", ay6, "\n")

print("nz1 =", nz1, "\n")
print("nz2 =", nz2, "\n")
print("nz3 =", nz3, "\n")
print("nz4 =", nz4, "\n")
print("nz5 =", nz5, "\n")
print("nz6 =", nz6, "\n")

print("oz1 =", oz1, "\n")
print("oz2 =", oz2, "\n")
print("oz3 =", oz3, "\n")
print("oz4 =", oz4, "\n")
print("oz5 =", oz5, "\n")
print("oz6 =", oz6, "\n")

print("az1 =", az1, "\n")
print("az2 =", az2, "\n")
print("az3 =", az3, "\n")
print("az4 =", az4, "\n")
print("az5 =", az5, "\n")
print("az6 =", az6, "\n")

#######################