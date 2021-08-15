import argparse
from datasets.datasets import load_dataset
from models import *
from eval import *
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import yaml
import neptune.new as neptune
from neptune.new.types import File

class Logger:
    def __init__(self,args):
        self.var_names = []
        self.data = {}
        self.args = args

        # Prepare Logging Tools
        if args.log_neptune:
            api_token =  "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1MzUwNWY0Ni04MjA3LTQzNzQtYTQyMi1kNGJlMWQ4ZTQ1ZGUifQ=="
            self.run = neptune.init(args.neptune_proj_name, api_token=api_token)
            #neptune.create_experiment(args.experiment_name, params=args.params, tags=args.neptune_tags)
            self.run['parameters'] = args.params
            self.run['experiment_name'] = args.experiment_name
            self.run['tags'] = args.neptune_tags
            self.samples_count = 0
        
        if args.log_tensorboard:
            pass

    def log(self,name:str, val, type=None):
        if name not in self.var_names:
            self.var_names.append(name)
            self.data[name] = []
        
        self.data[name].append(val)

        if self.args.log_neptune:
            if 'samples' in name:
                samples_name = name+'/'+str(self.samples_count)
                for sample in val:
                    self.run[samples_name].log(File.as_image(sample.transpose(1,2,0)))
                self.samples_count += 1
            else:
                self.run[name].log(val)
        
        if self.args.log_tensorboard:
            pass
    
        
def train(model, args, train_loader, val_loader, optimizer, clip_grad=None):
    '''
    
    '''
    device = next(model.parameters()).device
    logger = Logger(args)

    val_loss = eval_model(model, val_loader)
    logger.log('metrics/val/loss',val_loss)

    model.train()
    for idx_iter in tqdm(range(1, args.num_iters + 1),position=0,leave=True, disable=not args.log_tqdm):
        inps, targets = next(train_loader)
        inps, targets = inps.to(device), targets.to(device)
        
        optimizer.zero_grad()
        loss = model.loss(inps)
        loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        logger.log('metrics/train/loss',loss.item())
        
        if idx_iter%args.eval_every == 0:
            val_loss = eval_model(model, val_loader)
            logger.log('metrics/val/loss',val_loss)
        
        if idx_iter%args.generate_every == 0:
            if args.is_AR:
                generated_samples = generate_using_AR(model,args.final_num_gen)
            else:
                generated_samples = generate_using_flow(model,args.final_num_gen, args.floor_gen)

            logger.log('metrics/train/samples',generated_samples)
    model.eval()
    return logger

def main(args):
    print("Loading Dataset")
    args.is_AR = args.model in ['MADE', 'PixelCNN']
    img_size, train_loader, val_loader, test_loader = load_dataset(args.dataset, args.batch_size, args.is_AR)

    print("Loading Model")
    if args.model == 'MADE':
        assert img_size[0] == 1
        model = MADE(img_size[1], img_size[2])
    elif args.model == 'PixelCNN':
        model = PixelCNN(img_size[1], img_size[2])
    elif args.model == 'RealNVP':
        z_dist = torch.distributions.normal.Normal(0,1)
        model = RealNVP(img_size, z_dist, args.max_val, args.large_model)
    elif args.model == 'Glow':
        z_dist = torch.distributions.normal.Normal(0,1)
        model = Glow(img_size, z_dist, args.n_blocks, args.flows_per_block, args.max_val, args.large_model)
    else:
        raise NotImplementedError("Model {} not recognized".format(args.model))

    if args.use_cuda:
        model = model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(),lr=float(args.lr))

    print("Starting Training")
    logger = train(model, args, train_loader, val_loader, optimizer, args.clip_grad)
    
    final_test_loss = eval_model(model,test_loader)
    print("Final Test Loss: {:.5f}".format(final_test_loss))

    if args.is_AR:
        final_generated_samples = generate_using_AR(model,args.final_num_gen)
    else:
        final_generated_samples = generate_using_flow(model,args.final_num_gen, args.floor_gen)

    logger.log('metrics/train/samples',final_generated_samples)
    logger.log('metrics/test/loss',final_test_loss)

    # Save experiment results manually
    if args.log_file:
        to_save = logger.data
        to_save['params'] = args
        np.save(args.experiment_name,to_save,allow_pickle=True)
    
    # Save model
    if args.save_model_path is not None:
        torch.save(model, args.save_model_path)


class Args():
    def __init__(self,file_path):
        with open(file_path,"r") as f:
            params = yaml.safe_load(f)
        self.params = params
        for key in params.keys():
            setattr(self,key,params[key])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Autoregressive trainer")
    parser.add_argument("--config", default='./configs/MADE_shapes.yaml', type=str, help="Path to the config file")
    file_path = parser.parse_args().config
    args = Args(file_path)

    main(args)