readme: Test Code
1. The code file contains : 
	SQTraining.py
	DQLTraining.py 
	P3Testcode.py
2. Additional files: 
	SingleQL.hdf5,
	Memory.npy,
	frames.npy
	flags.npy
	rewards.npy
	actions.npy
3. P3Testcode python uses all the additional files and the trained model. 
4. In the test code Agent plays new game with the memory,actions, frames,rewards and flags obtained from the training.
5. env() creates environment in the test code and when Debug=True in test code, game environment animation is rendered with env.render().
6. In agent dont learn in the game like in Training.
7. Better results were observed in Single Qlearning.


References:

1. Basic understanding of RL
https://towardsdatascience.com/what-is-tabulated-reinforcement-learning-81eb8636f478

2. what is Deep RL
https://towardsdatascience.com/what-is-deep-reinforcement-learning-de5b7aa65778


3. AI playing Atari Pong using Tensor flow and Python
https://towardsdatascience.com/getting-an-ai-to-play-atari-pong-with-deep-reinforcement-learning-47b0c56e78ae

4.  Playing Atari with Deep Reinforcement Learning arXiv:1312.5602v1 

5. Deep Reinforcement Learning with Double Q-learning arXiv:1509.06461v3

 
	
	  
