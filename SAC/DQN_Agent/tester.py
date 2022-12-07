import sys
sys.dont_write_bytecode = True

import sys
import agent_torch as agent
import environment as env
#import tensorflow as tf
import os 
import parameters.setup as setup

environment = env.CarRacing(**setup.setup_dict['car racing'])
control = agent.DQN_Agent(environment=environment, model_name=sys.argv[1], **setup.setup_dict['agent'])

DIR = './models/best'
model_name = sys.argv[1]
base = 2700
save_frequency = 100
num_test_episodes = 100
number_of_checkpoints = 3#(len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]) - 2) 
file = open('./models/best/rewards.txt','w')
file.write("mean rewards, stds" + '\n')
file.close()
for n in range(number_of_checkpoints):
	chkp = n*save_frequency + base
	control.load("./models/best/"+model_name+"_{0}.pt".format(chkp))
	print('Currently testing checkpoint {0}'.format(chkp))
	mean, std, rewards = control.test(num_test_episodes, True)
	print('Checkpoint {0} got a score of {1} +- {2}'.format(chkp, mean, std))
	file = open('./models/best/rewards.txt','a')
	file.write(','.join(map(str, [mean, std])) + '\n')
	file.close()