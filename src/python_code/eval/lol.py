import numpy as np
import scipy.linalg as null_space

jacobian_c1 = np.array([[-1, -1, 0],
                        [0, -1, -1]])

ns = null_space.null_space(jacobian_c1)
print("The null space of case 1 jacobian is")
print(ns)

q,r = np.linalg.qr(jacobian_c1)
print("The range space of case 1 jacobian is")
print(q)

jacobian_c1_T = np.array([[-1, -1, 0],
                          [0, -1, -1]]).T

ns = null_space.null_space(jacobian_c1_T)
print("The null space of case 1 jacobian transpose is")
print(ns)

q,r = np.linalg.qr(jacobian_c1_T)

print("The range space of case 1 jacobian transpose is")
print(q)

jacobian_c2 = np.array([[-1, 0, 1],
                        [0, 0, 0]])

ns = null_space.null_space(jacobian_c2)
print("The null space of case 2 jacobian is")
print(ns)

q,r = np.linalg.qr(jacobian_c2)

print("The range space of case 2 jacobian is")
print(q)

jacobian_c2_T = np.array([[-1, 0, 1],
                          [0, 0, 0]]).T

ns = null_space.null_space(jacobian_c2_T)
print("The null space of case 2 jacobian transpose is")
print(ns)

q,r = np.linalg.qr(jacobian_c2_T)

print("The range space of case 2 jacobian transpose is")
print(q)