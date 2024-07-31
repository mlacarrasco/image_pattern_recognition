import numpy as np

def mrs(r, s, I, J):
    i = I**r
    j = J**s
    return np.sum(i*j)


#******************************
#      PROGRAMA PRINCIPAL     *
#******************************

BW=[[0, 0, 0,  0, 0, 0],
    [0, 0, 1,  0, 0, 0],
    [0, 1, 1,  1, 1, 0],
    [0, 0, 1,  0, 1, 0],
    [0, 0, 0,  0, 0, 0],
    [0, 0, 0,  0, 0, 0]]

BW= np.array(BW)

coords = np.argwhere(BW==1)
m00 = mrs(0,0,coords[:,0], coords[:,1])
m10 = mrs(1,0,coords[:,0], coords[:,1])
m01 = mrs(0,1,coords[:,0], coords[:,1])

ci = m10/m00
cj = m01/m00

print(ci, cj)


