import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent))


# class CombLoss(nn.Module):
#     def __init__(self, feat_dim, logits_dim, num_classes, m=0.35, s=32):
#         super(CombLoss, self).__init__()
#         self.lmc_loss = LargeMarginCosineLoss(feat_dim, num_classes, m, s)
#         self.ce_loss = LargeMarginCosineLoss(logits_dim, num_classes, m, s)

#     def forward(
#         self,
#         embeddings: torch.Tensor,
#         logits: torch.Tensor,
#         labels: torch.Tensor,
#     ) -> Tuple[torch.Tensor, torch.Tensor]:
#         # Calculate large margin cosine loss
#         loss = self.lmc_loss(embeddings, labels) + self.ce_loss(logits, labels)

#         return loss


class LargeMarginCosineLoss(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, m=0.35, s=32):
        super(LargeMarginCosineLoss, self).__init__()
        self.num_classes = num_classes
        self.m = m
        self.s = s
        self.eps = 1e-8

        self.w = nn.Parameter(torch.zeros(num_classes, embed_dim))  # [C, D]
        nn.init.xavier_normal_(self.w)

    def forward(self, x, labels):
        x_norm = F.normalize(x, p=2, dim=1, eps=self.eps)  # [B, D]
        w_norm = F.normalize(self.w, p=2, dim=1, eps=self.eps)  # [C, D]

        xw_norm = torch.mm(x_norm, w_norm.t())  # [B, C]

        one_hot = torch.zeros_like(xw_norm)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        margin = self.m * one_hot
        logits = self.s * (xw_norm - margin)

        return F.cross_entropy(logits, labels)

    # def forward(self, x, labels):
    #     x_norm = F.normalize(x, p=2, dim=1)
    #     w_norm = F.normalize(self.w, p=2, dim=1)

    #     cosine = F.linear(x_norm, w_norm)
    #     phi = cosine - self.m
    #     # --------------------------- convert label to one-hot ---------------------------
    #     one_hot = torch.zeros(cosine.size(), device="cuda")
    #     # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
    #     one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
    #     # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
    #     output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
    #     # you can use torch.where if your torch.__version__ is 0.4
    #     output *= self.s
    #     # print(output)

    #     return F.cross_entropy(output, labels)


# class TripletSoftmaxLoss(nn.Module):
#     def __init__(self, margin=0.5):
#         super(TripletSoftmaxLoss, self).__init__()
#         self.softmax_loss = nn.CrossEntropyLoss()
#         self.triplet_loss = SemiHardTripletLoss(margin)

#     def forward(self, embeddings, logits, labels):
#         # Calculate loss
#         softmax_loss = self.softmax_loss(logits, labels)
#         triplet_loss = self.triplet_loss(embeddings, labels)

#         # Calculate percision
#         predictions = torch.argmax(logits, dim=1)
#         correct = (predictions == labels).sum().item()
#         percision = correct / labels.size(0)

#         loss = triplet_loss + softmax_loss
#         return loss, percision


# class SemiHardTripletLoss(nn.Module):

#     def __init__(self, margin=0.5):
#         super(SemiHardTripletLoss, self).__init__()
#         self.margin = margin

#     def forward(self, embeddings, labels):
#         embeddings = F.normalize(embeddings, p=2, dim=1)
#         distance_matrix = torch.cdist(embeddings, embeddings, p=2)

#         loss = 0.0
#         triplet_count = 0

#         for label in labels:
#             label_mask = labels == label
#             label_indices = torch.where(label_mask)[0]
#             if label_indices.size(0) < 2:
#                 continue

#             negative_indices = torch.where(~label_mask)[0]

#             anchor_positives = list(combinations(label_indices.tolist(), 2))
#             for anchor_positive in anchor_positives:
#                 ap_distance = distance_matrix[anchor_positive[0], anchor_positive[1]]

#                 an_distances = distance_matrix[anchor_positive[0], negative_indices]
#                 an_distances = an_distances[an_distances > ap_distance]

#                 loss_values = F.relu(ap_distance - an_distances + self.margin)

#                 if len(loss_values):
#                     loss += loss_values.max()
#                     triplet_count += 1

#         if triplet_count > 1:
#             loss /= triplet_count
#         else:
#             loss = torch.tensor(1e-6, requires_grad=True, device=embeddings.device)

#         return loss
