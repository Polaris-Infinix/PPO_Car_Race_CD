# PPO_Car_Race_CD
This project trains an AI agent to autonomously drive in the **CarRacing-v3** environment using **Proximal Policy Optimization (PPO)** — implemented completely from scratch with **PyTorch** and **NumPy**.

No external reinforcement learning libraries like Stable Baselines or RLlib were used.

Watch the trained agent in action:
https://github.com/user-attachments/assets/6bfa03b2-50ea-424c-92eb-93594da556f7
##  How It Works

- The agent observes the environment through **4 stacked RGB frames** (shape: `4 x 96 x 96 x 3`), enabling motion detection from pixel changes over time.
- A **Convolutional Neural Network (CNN)** extracts spatial-temporal features from the visual input.
- These features are passed to a **deep Multi-Layer Perceptron (MLP)** that outputs:
  - An **action policy** (probabilities over discrete actions)
  - A **value estimate** for advantage calculation
- Trained using the **PPO algorithm**, carefully tuned for stable learning and generalization.
- Operates in a **discrete action space** with:
  - **Frame skipping**
  - **Reward shaping**
  - **Entropy regularization**

---




![image](https://github.com/user-attachments/assets/6460c77d-d913-4d96-bf26-3dab8fe0e075)
The above image represents the training of model, which is as par with many already existing PPO codes 


---




## 🛠️ Features

✅ PPO implemented from scratch  
✅ CNN for visual input  
✅ MLP for policy & value networks  
✅ Frame stacking (4 frames) for motion perception  
✅ Discrete action space  
✅ Tested and fine-tuned for stability  

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install numpy torch gym[box2d]


