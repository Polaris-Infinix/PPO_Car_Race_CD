import gymnasium as gym 
import numpy as np 
import torch
# import cv2

# This handles all the nuances in the environment and gives a proper input to the PPO to work with 

class Environment():
    def __init__(self):
        pass

    def create_env(self):
        self.env = gym.make("CarRacing-v3", lap_complete_percent=1,render_mode ="human", domain_randomize=False, continuous=False)
   
    
    #Run first 50 frames without action or obeservation  as this is the intro 
    def dry_run(self,dry_frames=52):
        self.create_env()
        obs, info = self.env.reset()
        for i in range(dry_frames):
            arr=np.array([0,0,0])
            obs, reward, done, truncated, info = self.env.step(0)
            if i==dry_frames-3:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1)
                self.obs_l=obs_tensor
            elif i==dry_frames-2:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1)
                self.obs_m=obs_tensor
            elif i==dry_frames-1:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1)
                self.obs_rm=obs_tensor

     
    # Ths handles frame stacking and deals with the nuainces of the environment observation 
    def input(self,action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.obs_r = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1)
        state=torch.cat((self.obs_r,self.obs_rm,self.obs_m,self.obs_l),dim=0)
        self.obs_l=self.obs_m
        self.obs_m=self.obs_rm
        self.obs_rm=self.obs_r
        return state/255,reward,done,truncated
    
    def reset(self):
        self.dry_run()
        state,reward,done,truncated=self.input(0)
        return state

        

        


            



# env=Environment()
# env.dry_run()
# while True:
#     env.input()


        



#   b=state.numpy()
#         image_bgr = (b[1] * 255).astype(np.uint8)
#         image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)

#     # Step 3: Show the image using OpenCV
#         cv2.imshow("Tensor Image", image_bgr)
#         cv2.waitKey(0)  # Wait for a key press to close
#         cv2.destroyAllWindows()
    
