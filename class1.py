import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]


def forward(x):
	return w * x

def loss(x,y):
	return (y - forward(x))*(y - forward(x))

l_list = []
w_list = []
for w in np.arange(0.0,10.0,0.1):

	print("w=" , w)
	l_sum = 0
	for x_val,y_val in zip(x_data,y_data):
		l_val = loss(x_val,y_val)
		l_sum = l_sum + l_val
	l_list.append(l_sum/len(x_data))
	w_list.append(w)
print(l_list,w_list)

