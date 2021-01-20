"""Shared models.
"""
import torch.nn as nn


class Model1(nn.Module):
    def __init__(self):
        super().__init__()


class Model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)


class Model3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 1)
        self.fc2 = nn.Linear(1, 2)


class Model4(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)
        self.seq = nn.Sequential(self.fc, self.fc)
