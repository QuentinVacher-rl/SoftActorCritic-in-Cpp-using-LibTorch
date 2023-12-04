# Soft Actor-Critic in C++ using LibTorch

I needed an implementation of Soft Actor-Critic in C++ for my work but couldn't find a suitable one, so I decided to create my own. 
The code is provided with a simple GridWorld environment and a main file to test the algorithm. 
I didn't have time to create a wrapper for Gym to test it.

JsonCpp and Libtorch are required. JsonCpp is already in the 'lib' folder.
It seems there is a compatibility error between JsonCpp and Torch when using Cuda, so the Libtorch version I'm using is for CPU only. If you find out why, I would be happy to know :)

This code is based on the python implementation of philtabor, which you can find here: [GitHub](https://github.com/philtabor/Youtube-Code-Repository/tree/eb3aa9733158a4f7c4ba1fefaa812b27ffd889b6/ReinforcementLearning/PolicyGradient/SAC)

Soft Actor-Critic Paper: [Paper](https://arxiv.org/abs/1801.01290)
