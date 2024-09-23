import functools
import torch
import torch.nn as nn
from time import time
from networks.resnet import resnet50
from networks.base_model import BaseModel, init_weights
import os


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        if opt.num_classes > 1:
            print(f"Attention! Train on {opt.num_classes} classes!")

        if self.isTrain and not opt.continue_train:
            self.model = resnet50(pretrained=False)
            self.model.fc = nn.Linear(2048, opt.num_classes)
            init_weights(self.model, init_type=opt.init_type, gain=opt.init_gain)
            
        if not self.isTrain or opt.continue_train:
            self.model = resnet50(num_classes=opt.num_classes)

        if self.isTrain:
            if opt.num_classes > 1:
                self.loss_fn = nn.CrossEntropyLoss()
            else:
                self.loss_fn = nn.BCEWithLogitsLoss()

            if opt.optim == 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                 lr=opt.lr, momentum=0.0, weight_decay=0)
            else:
                raise ValueError("optim should be [adam, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)
        self.model.to(opt.gpu_ids[0])
        self.opt = opt
        self.current_lr = opt.lr


    def adjust_learning_rate(self, min_lr=1e-6):
        for i, param_group in enumerate(self.optimizer.param_groups):
            lr = param_group['lr']
            # print(f'Before update: param_group {i} lr = {lr}')
            
            self.current_lr = lr / 10.0
            param_group['lr'] = self.current_lr  
            
            # print(f'After update: param_group {i} lr = {param_group["lr"]}')
            
            if param_group['lr'] < min_lr:
                print(f"Learning rate for param_group {i} is below min_lr ({min_lr}). Stop updating.")
                return False
        return True


    def set_input(self, input):
        self.input = input[0].to(self.device)
        if self.opt.num_classes > 1:
            self.label = input[1].to(self.device).to(torch.int64)
        else:
            self.label = input[1].to(self.device).float()


    def forward(self):
        self.output = self.model(self.input)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        
    def save_networks(self, epoch):
        """Save both model and optimizer state."""
        super().save_networks(epoch)  
        optimizer_path = os.path.join(self.save_dir, f'optimizer_{epoch}.pth')
        torch.save(self.optimizer.state_dict(), optimizer_path)  
        print(f"Saved optimizer state at epoch {epoch}.")

    def load_networks(self, epoch):
        """Load both model and optimizer state."""
        super().load_networks(epoch)  
        optimizer_path = os.path.join(self.save_dir, f'optimizer_{epoch}.pth')
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path))  
            # print(f"Loaded optimizer state from epoch {epoch}.")
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.current_lr  
                # print(f"Restored learning rate to {self.current_lr} after loading best model")

