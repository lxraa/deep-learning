x_data = [1.0,2.0,3.0]

y_data = [2.0,4.0,6.0]
w = 1.0
def forward(x):
	return w*x

def cost(xs,ys):
	c = 0
	for x,y in zip(xs,ys):
		c = c + (y - forward(x)) ** 2
	return c/len(xs)

def gradiend(xs,ys):
	grad = 0
	for x,y in zip(xs,ys):
		grad = grad + (2 * x * (x * w - y))
	return grad / len(xs)

for epoch in range(100):
	cost_val = cost(x_data,y_data)
	grad_val = gradiend(x_data,y_data)
	w = w - 0.01 * grad_val
	print(w)


