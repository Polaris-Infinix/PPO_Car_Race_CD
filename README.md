# PPO_Car_Race_CD
This is the repository that contains the the PPO code and model for continuous and Discrete action spaces for the environment Car-Racing 

https://github.com/user-attachments/assets/6bfa03b2-50ea-424c-92eb-93594da556f7

HOW IT WORKS

The agent receives input as 4 stacked RGB frames, giving it a short-term memory of motion (like a mini video clip).

A Convolutional Neural Network (CNN) processes the visual input to extract spatial and temporal features.

The CNN is followed by deep fully connected layers (MLP) that output both the action policy and the value estimate.

The model is trained using Proximal Policy Optimization (PPO) — a stable and widely-used policy gradient algorithm.

Works in a discrete action space, with reward shaping and frame skipping to improve learning stability.



![image](https://github.com/user-attachments/assets/6460c77d-d913-4d96-bf26-3dab8fe0e075)
The above image represents the training of model, which is as par with many already existing PPO codes 


