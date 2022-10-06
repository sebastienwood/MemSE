import torch
import math

ch = 2
wh = 2

def empty():
    return torch.zeros(ch, wh, wh, ch, wh, wh)

res = empty()
for c in range(res.shape[0]):
    for i in range(res.shape[1]):
        for j in range(res.shape[2]):
            for cc in range(res.shape[3]):
                for ii in range(res.shape[4]):
                    for jj in range(res.shape[5]):
                        res[c,i,j,cc,ii,jj] += 1

assert torch.all(res == 1)

# res = empty()
# for c in range(res.shape[0]):
#     for i in range(res.shape[1]):
#         for j in range(res.shape[2]):
#             for cc in range(c+1):
#                 for ii in range(i+1):
#                     for jj in range(j+1):
#                         print(f'{c,i,j,cc,ii,jj} and {cc,ii,jj, c,i,j}')
#                         res[c,i,j,cc,ii,jj] += 1
#                         res[cc,ii,jj, c,i,j] += 1

# assert torch.all(res == 1), res.view(int(math.sqrt(res.numel())), -1)


# res = empty()
# for c in range(res.shape[0]):
#     for i in range(res.shape[1]):
#         for j in range(res.shape[2]):
#             for cc in range(c+1, res.shape[3]):
#                 for ii in range(i+1, res.shape[4]):
#                     for jj in range(j+1, res.shape[5]):
#                         res[c,i,j,cc,ii,jj] += 1
#                         res[cc,ii,jj, c,i,j] += 1

# assert torch.all(res == 1), res

# res = empty()
# for c in range(res.shape[0]):
#     for i in range(res.shape[1]):
#         for j in range(res.shape[2]):
#             for cc in range(c+1):
#                 for ii in range(i+1):
#                     for jj in range(j+1):
#                         res[c,i,j,cc,ii,jj] += 1
#                         if cc != c or ii != i or jj != j:
#                             print(f'{c,i,j,cc,ii,jj} and {cc,ii,jj, c,i,j}')
#                             res[cc,ii,jj, c,i,j] += 1

# print((res==0).nonzero(as_tuple=False))
# assert torch.all(res == 1), res.view(int(math.sqrt(res.numel())), -1)


# res = empty()
# for n in range(res.shape[0] * res.shape[1] * res.shape[2]):
#     c = n // res.shape[1] * res.shape[2]
#     ij = n % res.shape[1] * res.shape[2]
#     i = ij // res.shape[2]
#     j = ij % res.shape[2]
#     for n2 in range(n+1, res.shape[0] * res.shape[1] * res.shape[2]):
#         cc = n2 // res.shape[1] * res.shape[2]
#         ij2 = n2 % res.shape[1] * res.shape[2]
#         ii = ij2 // res.shape[2]
#         jj = ij2 % res.shape[2]
#         print(f'{n2, res.shape[1] * res.shape[2]}')
#         print(f'({n,n2}) {c,i,j,cc,ii,jj} and {cc,ii,jj, c,i,j}')
#         res[c,i,j,cc,ii,jj] += 1
#         if cc != c or ii != i or jj != j:
#             res[cc,ii,jj, c,i,j] += 1

# print((res==0).nonzero(as_tuple=False))
# assert torch.all(res == 1), res.view(int(math.sqrt(res.numel())), -1)