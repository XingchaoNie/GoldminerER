import torch.optim as optim
from torch.nn.utils import clip_grad_value_
from torch.optim import lr_scheduler


class Optim:
    def __init__(self, lr, max_grad_value, weight_decay):
        self.lr = lr
        self.max_grad_value = max_grad_value
        self.weight_decay = weight_decay
        self.params = None
        self.optimizer = None

    def set_parameters(self, params, name):
        self.params = list(params)
        if name == "sgd":
            self.optimizer = optim.SGD(
                self.params, lr=self.lr, weight_decay=self.weight_decay
            )
        elif name == "rmsprop":
            self.optimizer = optim.RMSprop(
                self.params, lr=self.lr, weight_decay=self.weight_decay
            )
        elif name == "adam":
            self.optimizer = optim.Adam(
                self.params, lr=self.lr, weight_decay=self.weight_decay
            )
        elif name == "adamw":
            self.optimizer = optim.AdamW(
                self.params, lr=self.lr, weight_decay=self.weight_decay
            )

    def get_scheduler(self, sch):
        print("Using Scheduler")
        if sch == "reduceLR":
            sched = lr_scheduler.ReduceLROnPlateau(self.optimizer, "min")
        elif sch == "expLR":
            sched = ExponentialLR(self.optimizer, gamma=0.9)
        return sched

    def step(self):
        if self.max_grad_value != -1:
            clip_grad_value_(self.params, self.max_grad_value)
        self.optimizer.step()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
