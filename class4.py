import torch
import numpy
#1、准备数据集
#2、设计模型，计算y-hat
#3、构造损失函数
#4、forward-backward-update

x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[2.0],[4.0],[6.0]])


class LinearModel(torch.nn.Module):
	def __init__(self):
		super(LinearModel,self).__init__()
		self.linear = torch.nn.Linear(1,1)
	def forward(self,x):
		return self.linear(x)

model = LinearModel()	#计算模型

criterion = torch.nn.MSELoss(size_average = False) #损失函数计算器
opt = torch.optim.SGD(model.parameters(),lr = 0.01) #最值计算器（优化器）

for epoch in range(1000):
	y_hat = model(x_data)
	loss = criterion(y_hat,y_data)
	opt.zero_grad()
	loss.backward()
	opt.step()
print("w = ",model.linear.weight.item())
print("b = ",model.linear.bias.item())


