import torch
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import json

# CAPACITIES = {10: 20., 20: 30., 50: 40., 100: 50.}
# CAPACITIES = {5: 10., 10: 20., 20: 30., 50: 40., 100: 50.}
CAPACITIES = {20: 3, 50: 7, 80: 11, 100: 15, 120: 18}
max_demand = 9

def generate_data(device, batch = 10, n_car_each_depot = 3, n_depot = 3, n_customer = 50, capa = 1., seed = None):
	if seed is not None:
		torch.manual_seed(seed)
	n_node = n_depot + n_customer
	n_car = n_car_each_depot * n_depot
	# assert (9. / CAPACITIES[n_customer]) * n_customer <= capa * n_car, 'infeasible; Customer Demand should be smaller than Vechile Capacity' 
	#assert (max_demand / CAPACITIES[n_customer]) * n_customer <= capa * n_car, 'infeasible; Customer Demand should be smaller than Vechile Capacity'
	return {'depot_xy': 10*torch.rand((batch, n_depot, 2), device = device)
			,'customer_xy': 10*torch.rand((batch, n_customer, 2), device = device)
			# ,'demand': torch.randint(low = 1, high = 10, size = (batch, n_customer), device = device) / CAPACITIES[n_customer]
			,'demand': torch.ones(size = (batch, n_customer),dtype=torch.float32, device = device)
			# ,'car_start_node': torch.randint(low = 0, high = n_depot, size = (batch, n_car), device = device)
			,'car_start_node': torch.arange(n_depot, device = device)[None,:].repeat(batch, n_car_each_depot)
			 #,'car_capacity': torch.ones((batch, n_car), device = device)
			,'car_capacity': CAPACITIES[n_customer] * torch.ones((batch, n_car), device = device)
			,'car_level' : torch.arange(n_car_each_depot, device = device)[None,:].repeat(batch, n_depot)
			,'demand_level':torch.randint(low = 0, high = n_car_each_depot, size = (batch, n_customer), device = device)
			}

class Generator(Dataset):
	""" https://github.com/utkuozbulak/pytorch-custom-dataset-examples
		 https://github.com/wouterkool/attention-learn-to-route/blob/master/problems/vrp/problem_vrp.py
		 https://github.com/nperlmut31/Vehicle-Routing-Problem/blob/master/dataloader.py
	"""
	def __init__(self, device, n_samples = 5120, n_car_each_depot = 3, n_depot = 3, n_customer = 50, capa = 1., seed = None):
		if seed is not None:
			self.data = generate_data(device, n_samples, n_car_each_depot, n_depot, n_customer, capa, seed)
		self.data = generate_data(device, n_samples, n_car_each_depot, n_depot, n_customer, capa, seed)
		
	def __getitem__(self, idx):
		dic = {}
		for k, v in self.data.items():
			# e.g., dic['depot_xy'] = self.data['depot_xy'][idx]
			dic[k] = v[idx]
		#print(dic)
		return dic

	def __len__(self):
		return self.data['depot_xy'].size(0)

	
if __name__ == '__main__':
	import torch

	print(torch.__version__)
	print(torch.cuda.is_available())
	'''
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print(device)
	batch, batch_steps, n_customer = 128, 10, 20
	dataset = Generator(device, n_samples = 3,
		n_car_each_depot = 3, n_depot = 3, n_customer = 50, capa = 2.)
	data = next(iter(dataset))	
	print(data)
'''
	# generate_data(device, batch = 10, n_car = 15, n_depot = 2, n_customer = 20, seed = )
	# data = {}
	# seed = 123
	# for k in ['depot_xy', 'customer_xy', 'demand', 'car_start_node', 'car_capacity']:
	# 	elem = generate_data(device, batch = 1, n_car = 15, n_depot = 2, n_customer = n_customer, seed = seed)[k].squeeze(0)
	# 	data[k] = elem
	
	"""
	dataloader = DataLoader(dataset, batch_size = 8, shuffle = True)
	for i, data in enumerate(dataloader):
		for k, v in data.items():
			print(k, v.size())
			if k == 'demand': print(v[0])
		if i == 0:
			break
	"""
