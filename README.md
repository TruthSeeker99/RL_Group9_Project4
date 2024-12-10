# RL_Group9_Project4

This repository contains the implementation for **WPI-DS551 Reinforcement Learning Project 4**.

## Resources
- **Demo Video**: [Download here](https://drive.google.com/file/d/1SX8rwjAEx3dLwKO7aByH1MT2DFZ4lQgL/)  
- **Model Checkpoints**: [Download here](https://drive.google.com/file/d/1BUSAON2ULaxpbSSDfPr5Vu5ewUKGcFEA/) and place them into the `./ckpts/` directory.  
- **ROM File**: [Download here](https://wowroms.com/en/roms/mame/street-fighter-iii-3rd-strike-fight-for-the-futur-japan-clone/106255.html).  

## Installation
To install the required dependencies, run:
```sh
pip install -r requirements.txt
```

## Training
```sh
# DPO
diambra run -r [game_rom_absolute_path] -s=4 python DQN.py 
# PPO
diambra run -r [game_rom_absolute_path] -s=4 python PPO.py 
```

## Testing
```sh
diambra run -r [game_rom_absolute_path] python test_agent.py --model PPO_final 
```

## Recording
```sh
diambra run -r [game_rom_absolute_path] python record_video.py --model PPO_final 
```

## Notes

- Replace `[game_rom_absolute_path]` with the absolute path to the ROM file on your system.  
- Ensure that the pre-trained model is located in the `./ckpts/` directory before running testing or recording scripts.





