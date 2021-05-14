from stable_baselines.common.policies import FeedForwardPolicy, register_policy
import tensorflow as tf

# Custom MLP policy of three layers of size 128 each
class MlpPolicy128(FeedForwardPolicy):
	def __init__(self, *args, **kwargs):
		super(MlpPolicy128, self).__init__(*args, **kwargs,
										   net_arch=[dict(pi=[128, 128, 128],
														  vf=[128, 128, 128])],
										   feature_extraction="mlp")
# Register the policy, it will check that the name is not already taken
register_policy('MlpPolicy128', MlpPolicy128)




# Custom MLP policy using relu
class MlpPolicyReLU128(FeedForwardPolicy):
	def __init__(self, *args, **kwargs):
		super(MlpPolicyReLU128, self).__init__(*args, **kwargs,
											net_arch=[dict(pi=[128,128],
														  vf=[128,128])],
											act_fun=tf.nn.relu,
											feature_extraction="mlp")
# Register the policy, it will check that the name is not already taken
register_policy('MlpPolicyReLU128', MlpPolicyReLU128)


# Custom MLP policy using relu
class MlpPolicyReLU10000(FeedForwardPolicy):
	def __init__(self, *args, **kwargs):
		super(MlpPolicyReLU10000, self).__init__(*args, **kwargs,
											net_arch=[dict(pi=[10000],
														  vf=[10000])],
											act_fun=tf.nn.relu,
											feature_extraction="mlp")
# Register the policy, it will check that the name is not already taken
register_policy('MlpPolicyReLU10000', MlpPolicyReLU10000)



# Custom MLP policy 
class MlpPolicyRelu64_2(FeedForwardPolicy):
	def __init__(self, *args, **kwargs):
		super(MlpPolicyRelu64_2, self).__init__(*args, **kwargs,
											net_arch=[dict(pi=[64, 64],
														  vf=[64, 64])],
											act_fun=tf.nn.relu,
											feature_extraction="mlp")
# Register the policy, it will check that the name is not already taken
register_policy('MlpPolicyRelu64_2', MlpPolicyRelu64_2)

# Custom MLP policy 
class MlpPolicyRelu64_3(FeedForwardPolicy):
	def __init__(self, *args, **kwargs):
		super(MlpPolicyRelu64_3, self).__init__(*args, **kwargs,
											net_arch=[dict(pi=[64, 64, 64],
														  vf=[64, 64, 64])],
											act_fun=tf.nn.relu,
											feature_extraction="mlp")
# Register the policy, it will check that the name is not already taken
register_policy('MlpPolicyRelu64_3', MlpPolicyRelu64_3)

# Custom MLP policy 
class MlpPolicyRelu64_4(FeedForwardPolicy):
	def __init__(self, *args, **kwargs):
		super(MlpPolicyRelu64_4, self).__init__(*args, **kwargs,
											net_arch=[dict(pi=[64, 64, 64, 64],
														  vf=[64, 64, 64, 64])],
											act_fun=tf.nn.relu,
											feature_extraction="mlp")
# Register the policy, it will check that the name is not already taken
register_policy('MlpPolicyRelu64_4', MlpPolicyRelu64_4)

# Custom MLP policy 
class MlpPolicyRelu128_2(FeedForwardPolicy):
	def __init__(self, *args, **kwargs):
		super(MlpPolicyRelu128_2, self).__init__(*args, **kwargs,
											net_arch=[dict(pi=[128,128],
														  vf=[128,128])],
											act_fun=tf.nn.relu,
											feature_extraction="mlp")
# Register the policy, it will check that the name is not already taken
register_policy('MlpPolicyRelu128_2', MlpPolicyRelu128_2)

# Custom MLP policy 
class MlpPolicyRelu128_3(FeedForwardPolicy):
	def __init__(self, *args, **kwargs):
		super(MlpPolicyRelu128_3, self).__init__(*args, **kwargs,
											net_arch=[dict(pi=[128,128,128],
														  vf=[128,128,128])],
											act_fun=tf.nn.relu,
											feature_extraction="mlp")
# Register the policy, it will check that the name is not already taken
register_policy('MlpPolicyRelu128_3', MlpPolicyRelu128_3)

# Custom MLP policy 
class MlpPolicyRelu128_4(FeedForwardPolicy):
	def __init__(self, *args, **kwargs):
		super(MlpPolicyRelu128_4, self).__init__(*args, **kwargs,
											net_arch=[dict(pi=[128,128,128,128],
														  vf=[128,128,128,128])],
											act_fun=tf.nn.relu,
											feature_extraction="mlp")
# Register the policy, it will check that the name is not already taken
register_policy('MlpPolicyRelu128_4', MlpPolicyRelu128_4)

# Custom MLP policy 
class MlpPolicyTanh64_2(FeedForwardPolicy):
	def __init__(self, *args, **kwargs):
		super(MlpPolicyTanh64_2, self).__init__(*args, **kwargs,
											net_arch=[dict(pi=[64, 64],
														  vf=[64, 64])],
											act_fun=tf.nn.tanh,
											feature_extraction="mlp")
# Register the policy, it will check that the name is not already taken
register_policy('MlpPolicyTanh64_2', MlpPolicyTanh64_2)

# Custom MLP policy 
class MlpPolicyTanh64_3(FeedForwardPolicy):
	def __init__(self, *args, **kwargs):
		super(MlpPolicyTanh64_3, self).__init__(*args, **kwargs,
											net_arch=[dict(pi=[64, 64, 64],
														  vf=[64, 64, 64])],
											act_fun=tf.nn.tanh,
											feature_extraction="mlp")
# Register the policy, it will check that the name is not already taken
register_policy('MlpPolicyTanh64_3', MlpPolicyTanh64_3)

# Custom MLP policy 
class MlpPolicyTanh64_4(FeedForwardPolicy):
	def __init__(self, *args, **kwargs):
		super(MlpPolicyTanh64_4, self).__init__(*args, **kwargs,
											net_arch=[dict(pi=[64, 64, 64, 64],
														  vf=[64, 64, 64, 64])],
											act_fun=tf.nn.tanh,
											feature_extraction="mlp")
# Register the policy, it will check that the name is not already taken
register_policy('MlpPolicyTanh64_4', MlpPolicyTanh64_4)

# Custom MLP policy 
class MlpPolicyTanh128_2(FeedForwardPolicy):
	def __init__(self, *args, **kwargs):
		super(MlpPolicyTanh128_2, self).__init__(*args, **kwargs,
											net_arch=[dict(pi=[128,128],
														  vf=[128,128])],
											act_fun=tf.nn.tanh,
											feature_extraction="mlp")
# Register the policy, it will check that the name is not already taken
register_policy('MlpPolicyTanh128_2', MlpPolicyTanh128_2)

# Custom MLP policy 
class MlpPolicyTanh128_3(FeedForwardPolicy):
	def __init__(self, *args, **kwargs):
		super(MlpPolicyTanh128_3, self).__init__(*args, **kwargs,
											net_arch=[dict(pi=[128,128,128],
														  vf=[128,128,128])],
											act_fun=tf.nn.tanh,
											feature_extraction="mlp")
# Register the policy, it will check that the name is not already taken
register_policy('MlpPolicyTanh128_3', MlpPolicyTanh128_3)

# Custom MLP policy 
class MlpPolicyTanh128_4(FeedForwardPolicy):
	def __init__(self, *args, **kwargs):
		super(MlpPolicyTanh128_4, self).__init__(*args, **kwargs,
											net_arch=[dict(pi=[128,128,128,128],
														  vf=[128,128,128,128])],
											act_fun=tf.nn.tanh,
											feature_extraction="mlp")
# Register the policy, it will check that the name is not already taken
register_policy('MlpPolicyTanh128_4', MlpPolicyTanh128_4)


# Custom MLP policy 
class MlpPolicyRelu512_3(FeedForwardPolicy):
	def __init__(self, *args, **kwargs):
		super(MlpPolicyRelu512_3, self).__init__(*args, **kwargs,
											net_arch=[dict(pi=[512,512,512],
														  vf=[512,512,512])],
											act_fun=tf.nn.relu,
											feature_extraction="mlp")
# Register the policy, it will check that the name is not already taken
register_policy('MlpPolicyRelu512_3', MlpPolicyRelu512_3)