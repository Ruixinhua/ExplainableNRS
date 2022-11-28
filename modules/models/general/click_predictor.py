from math import sqrt

import torch
import torch.nn as nn


class DotProduct(nn.Module):
    def __init__(self):
        super(DotProduct, self).__init__()

    def forward(self, candidate_news_vector, user_vector):
        """
        Args:
            candidate_news_vector: batch_size, candidate_size, X
            user_vector: batch_size, X
        Returns:
            (shape): batch_size
        """
        # batch_size, candidate_size
        if len(user_vector.shape) == 3:  # user_vector.shape = (batch_size, candidate_size, X)
            probability = torch.sum(user_vector * candidate_news_vector, dim=-1)
        else:
            probability = torch.bmm(candidate_news_vector, user_vector.unsqueeze(dim=-1)).squeeze(dim=-1)
        return probability


class DNNClickPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=None):
        super(DNNClickPredictor, self).__init__()
        if hidden_size is None:
            # TODO: is sqrt(input_size) a good default value?
            hidden_size = int(sqrt(input_size))
        self.dnn = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, candidate_news_vector, user_vector):
        """
        Args:
            candidate_news_vector: batch_size, X
            user_vector: batch_size, X
        Returns:
            (shape): batch_size
        """
        # batch_size
        return self.dnn(torch.cat((candidate_news_vector, user_vector), dim=-1)).squeeze()
