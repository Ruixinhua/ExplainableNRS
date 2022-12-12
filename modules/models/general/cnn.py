import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1D(nn.Module):
    def __init__(self, in_channels: int, kernel_num: int, window_size: int, cnn_method: str = "naive"):
        super(Conv1D, self).__init__()
        assert cnn_method in ['naive', 'group3', 'group5']
        self.cnn_method = cnn_method
        self.in_channels = in_channels
        if self.cnn_method == 'naive':
            self.conv = nn.Conv1d(in_channels=in_channels, out_channels=kernel_num, kernel_size=window_size,
                                  padding=(window_size - 1) // 2)
        elif self.cnn_method == 'group3':
            assert kernel_num % 3 == 0
            self.conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=kernel_num // 3, kernel_size=1, padding=0)
            self.conv2 = nn.Conv1d(in_channels=self.in_channels, out_channels=kernel_num // 3, kernel_size=3, padding=1)
            self.conv3 = nn.Conv1d(in_channels=self.in_channels, out_channels=kernel_num // 3, kernel_size=5, padding=2)
        else:
            assert kernel_num % 5 == 0
            self.conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=kernel_num // 5, kernel_size=1, padding=0)
            self.conv2 = nn.Conv1d(in_channels=self.in_channels, out_channels=kernel_num // 5, kernel_size=2, padding=0)
            self.conv3 = nn.Conv1d(in_channels=self.in_channels, out_channels=kernel_num // 5, kernel_size=3, padding=1)
            self.conv4 = nn.Conv1d(in_channels=self.in_channels, out_channels=kernel_num // 5, kernel_size=4, padding=1)
            self.conv5 = nn.Conv1d(in_channels=self.in_channels, out_channels=kernel_num // 5, kernel_size=5, padding=2)

    # Input
    # feature : [batch_size, feature_dim, length]
    # Output
    # out     : [batch_size, kernel_num, length]
    def forward(self, feature):
        if self.cnn_method == 'naive':
            return F.relu(self.conv(feature))  # [batch_size, kernel_num, length]
        elif self.cnn_method == 'group3':
            return F.relu(torch.cat([self.conv1(feature), self.conv2(feature), self.conv3(feature)], dim=1))
        else:
            padding_zeros = torch.zeros([feature.size(0), self.in_channels, 1], device=self.device)
            return F.relu(torch.cat([self.conv1(feature),
                                     self.conv2(torch.cat([feature, padding_zeros], dim=1)),
                                     self.conv3(feature),
                                     self.conv4(torch.cat([feature, padding_zeros], dim=1)),
                                     self.conv5(feature)], dim=1))
