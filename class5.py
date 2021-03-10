import torchvision
import torch
import torch.nn.functional as F
#train_set = torchvision.datasets.MNIST(root="../dataset/mnist",train=True,download=True)
#test_set = torchvision.datasets.MNIST(root="../dataset/mnist",train=False,download=True)

#print(train_set)
# 1、准备dataset----------------------------------------------------------------
x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[0],[0],[1]])


# 2、设计模型类-----------------------------------------------------------------
class LogisticRegressionModel(torch.nn.Module):
	def __init__(self):
		super(LogisticRegressionModel,self).__init__()
		self.linear = torch.nn.Linear(1,1)
	def forward(self,x):
		return F.sigmoid(self.linear(x))

model = LogisticRegressionModel()

# 3、构造损失函数和优化器（使用torch api）---------------------------------------
criterion = torch.nn.BCELoss(size_average = False)
opt = torch.optim.SGD(model.parameters(),lr = 0.01)
# 4、循环训练---------------------------------------------------------------------
for ep in range(0,100):
	y_hat = model(x_data)
	loss = criterion(y_hat,y_data)
	opt.zero_grad()
	loss.backward()
	opt.step()
print(model.linear.weight.item())
print(model.linear.bias.item())



