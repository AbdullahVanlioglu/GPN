import numpy as np

GameDescription = {}
GameDescription['aliens'] = {'ascii':[".","0","1","2","A"],
		             'mapping':[13, 3, 11, 12, 1],
			     'state_shape':(14, 12, 32),
			     'model_shape':[(3, 4),(6, 8),(12, 16),(12, 32)],
			     'requirements':["A"]}
"""
GameDescription['zelda'] = {'ascii':[".","w","g","+","1","2","3","A"],
                             'mapping':[13, 0, 3, 4, 10, 11, 12, 7],
                             'state_shape':(14, 12, 16),
                             'model_shape':[(3, 4),(6, 8),(12, 16)],
			     'requirements':["A","g","+"]}
"""
GameDescription['gpn'] = {'ascii':["-","W","X"],
                             'mapping':[1, 2, 3],
                             'state_shape':(14, 12, 16),
                             'model_shape':[(5,5),(10,10),(20,20),(80,80)],
			     'requirements':["-","X","W"]}

class Env:
	def __init__(self, name, length):
		self.name = name
		self.length = length
		try:
			self.ascii = GameDescription[name]['ascii']
			self.mapping = GameDescription[name]['mapping']
			self.state_shape = GameDescription[name]['state_shape']
			self.model_shape = GameDescription[name]['model_shape']
			self.requirements = GameDescription[name]['requirements']
		except:
			raise Exception(name + " data not implemented in env.py")

		self.map_level = np.vectorize(lambda x: self.ascii[x])

	def create_levels(self, tensor): # 32x8x12x16 our model: 32x3x80x80
		# print("tensor",tensor.shape)
		lvl_array = tensor.argmax(dim=1).cpu().numpy() # 32x12x16 our model: 32x80x80
		lvls = self.map_level(lvl_array).tolist()
		#lvl_strs = ['\n'.join([''.join(row) for row in lvl]) for lvl in lvls]
		lvl_strs = [[''.join(row) for row in lvl] for lvl in lvls]
		return lvl_strs

	def pass_requirements(self, lvl_str):
		num_failed = sum(lvl_str.find(i) == -1 for i in self.requirements)
		return num_failed == 0
