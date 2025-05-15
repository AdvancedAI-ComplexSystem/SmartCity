import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from .src.config import Config
from .src.HINT import HINT
import wandb
import time 
class H_I_N_T:
    
    def __init__(self, mode=2):
        r"""starts the model

        Args:
            mode (int): 1: train, 2: test, reads from config file if not specified
        """
        config = self.load_config(mode)
        
        config.DEVICE = torch.device("cuda:5")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
        


        # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
        cv2.setNumThreads(0)


        # initialize random seed
        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed_all(config.SEED)
        np.random.seed(config.SEED)
        random.seed(config.SEED)



        # build the model and initialize
        model = HINT(config)
        model.load()

        

        self.inpaint_model = model.inpaint_model
        

    def demask_state(self, masked_next_state, action, states, mask, reverse_step=2):    

        self.inpaint_model.eval()
        device = masked_next_state.device
        batch_size = masked_next_state.shape[0]
        image = torch.zeros(batch_size, 2, 12).to(device)
        masks = torch.zeros(batch_size, 2, 12).to(device)
        for i in range(batch_size):
            image[i, 0, :] = masked_next_state[i, 8:20]
            image[i, 1, :] = masked_next_state[i, 20:]
            masks[i, 0, :] = mask[i, 8:20]
            masks[i, 1, :] = mask[i, 20:]
        
        padded_arr = torch.zeros((batch_size,1,32,32), dtype=image.dtype)
        padded_arr[:, 0, :2, :12] = image
        images = padded_arr.to(device)
        
        padded_arr = torch.zeros((batch_size,1,32,32), dtype=images.dtype)
        padded_arr[:, 0, :2, :12] = masks
        masks = padded_arr.to(device)
        with torch.no_grad():
            tsince = int(round(time.time()*1000))
            outputs_img = self.inpaint_model(images, masks)
            ttime_elapsed = int(round(time.time()*1000))-tsince
            print('test time elaspsed {}ms'.format(ttime_elapsed))
        outputs_merged = (images* masks) + (outputs_img * (1 - masks))
        outputs_merged = outputs_merged[:, 0, :2, :12]
        masked_next_state[:, 8:20] = outputs_merged[:,0,:]
        masked_next_state[:, 20:] = outputs_merged[:,1,:]
        return masked_next_state.cpu().numpy()




    def load_config(self, mode=2):
        r"""loads model config

        Args:
            mode (int): 1: train, 2: test, reads from config file if not specified
        """

        parser = argparse.ArgumentParser()
        parser.add_argument('--path', '--checkpoints', type=str, default='/home/myli/RL_Optimizer/RobustLight/inferences/checkpoints',
                            help='model checkpoints path (default: ./checkpoints)')

        parser.add_argument('--model', type=int, default='2', choices=[2])

        # test mode
        if mode == 2:
            parser.add_argument('--input', type=str, help='path to the input images directory or an input image')
            parser.add_argument('--mask', type=str, help='path to the masks directory or a mask file')
            parser.add_argument('--output', type=str, help='path to the output directory')

        args = parser.parse_args()
        config_path = os.path.join(args.path, 'config.yml')
        # config_path = os.path.join(args.path, 'config_test.yml')

        # create checkpoints path if does't exist
        if not os.path.exists(args.path):
            os.makedirs(args.path)

        # copy config template if does't exist
        if not os.path.exists(config_path):
            copyfile('./config.yml.example', config_path)

        # load config file
        config = Config(config_path)
        print(config_path)

        # train mode
        if mode == 1:
            config.MODE = 1
            if args.model:
                config.MODEL = args.model

        # test mode
        elif mode == 2:
            config.MODE = 2
            config.MODEL = args.model if args.model is not None else 3

            if args.input is not None:
                config.TEST_INPAINT_IMAGE_FLIST = args.input

            if args.mask is not None:
                config.TEST_MASK_FLIST = args.mask

            if args.output is not None:
                config.RESULTS = args.output


        return config


if __name__ == "__main__":
    main()
