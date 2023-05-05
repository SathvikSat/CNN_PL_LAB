import torch, numpy as np


#tensor dim 1, 2, n dim

#tensors
x = torch.empty(1) #scalar
y = torch.empty(3) #1D vector with 3 elements
z = torch.empty(2,2,3) # 3Dimensional


x = torch.rand(2,2, dtype=torch.float16)
#torch.zero(2,2)
#torch.ones(2,3)
print(x.dtype)
print(x.size())


#tensor from a data
x = torch.tensor([2.5, 0.1])

x = torch.ones(2,2)
y = torch.ones(2,2)


z = x +y #element wise addition
z = torch.add(x,y) #same as above 

#inplace addition
y.add_(x) #trailing_ does inplace, modifies the variable it is applied on


z = x-y
z = torch.sub(x,y)
y.sub_(x)


z = torch.mul(x,y)
z = x/y
z = torch.div(x,y)


#slicing 
x = torch.rand(5,3)
print(x[:, 0]) #all rows and only col 0
print(x[1,:]) #row 1(second row) but all columns
print(x[1,1]) # element at certain pos, gives tensor not actual value
print(x[1,1].item()) #actual value when only element


#reshape
x = torch.rand(
    4,4
)
y = x.view(16) # converted to 1D, numbers should match
y = x.view(-1, 8) #auto determines for -1
#size will be 2 by 8


#converting np to torch tensor
a = torch.ones(5)
print(a)
b = a.numpy()
print(type(b)) # numpy array type

#on GPU both obj will share same mem loc, ie., a and b
a.add_(1)

#numpy to tensor

a = np.ones(5)
b = torch.from_numpy(a)
print(b) #default float64


a += 1 #inc each val, b would also be modified, when tensor is on GPU

print(y)

x = torch.ones(5, requires_grad=True)
print(x) #required_grad=True need to cal gradient later in optimisation steps

if torch.cuda.is_available():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #tensor on gpu
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x+y # op performed on on GPU

    #z.numpy() # err numpy can only handle CPU tensors
    z = z.to("cpu")

#print(y)
#print(z)
