import sys
from pathlib import Path
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
import numpy as np


sys.path.append(str(Path(__file__).parent))


class Estimator:
    def __init__(
        self,
        save_dir: Union[str, Path],
        device: torch.device,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

    def estimate(self, model: nn.Module, dataloader: DataLoader):
        model.eval()

        features, labels = self._extract_features(model=model, dataloader=dataloader)
        similarity_matrix = self._cosine_similarity(features)  # (N, N)

        acc = self._accuracy(similarity_matrix, labels)
        rank1 = self._rank1_accuracy(similarity_matrix, labels)
        tar = self._tar_at_far(similarity_matrix, labels, target_far=1e-6)

        return {"acc": acc, "rank_1": rank1, "tar_far": tar}

    def _extract_features(self, model: nn.Module, dataloader: DataLoader):
        features, labels = [], []

        with torch.no_grad():
            for images, targets in dataloader:
                if not isinstance(images, torch.Tensor):
                    raise TypeError(
                        f"Expected images to be torch.Tensor, got {type(images)}"
                    )
                if not isinstance(targets, torch.Tensor):
                    raise TypeError(
                        f"Expected targets to be torch.Tensor, got {type(targets)}"
                    )

                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                embeddings = model(images)
                embeddings = F.normalize(embeddings, p=2, dim=1)

                features.append(embeddings)
                labels.append(targets)

            features = torch.cat(features)
            labels = torch.cat(labels)

            unique_labels, counts = torch.unique(labels, return_counts=True)
            valid_labels = unique_labels[counts >= 2]
            valid_mask = torch.isin(labels, valid_labels)

            filtered_features = features[valid_mask]
            filtered_labels = labels[valid_mask]

        return filtered_features, filtered_labels

    def _cosine_similarity(
        self,
        features: torch.Tensor,
        normalize=False,
    ):
        if normalize:
            features = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.mm(features, features.t())  # (N, N)

        similarity_matrix.fill_diagonal_(-1)

        return similarity_matrix

    def _accuracy(
        self, similarity_matrix: torch.Tensor, labels: torch.Tensor, threshold=0.95
    ):

        N = len(labels)
        triu_mask = torch.triu(
            torch.ones(N, N, dtype=torch.bool, device=labels.device),
            diagonal=1,
        )
        identity_mask = labels.unsqueeze(1) == labels.unsqueeze(0)

        predictions = similarity_matrix > threshold

        correct_predictions = (predictions == identity_mask) & triu_mask
        total_comparisons = triu_mask.sum()

        accuracy = (
            correct_predictions.sum().float().item() / total_comparisons.float().item()
        )

        return accuracy

    def _rank1_accuracy(
        self,
        similarity_matrix: torch.Tensor,
        labels: torch.Tensor,
    ):
        """
        Get rank-1 accuracy.
        """
        total = labels.size(0)

        top1_indices = similarity_matrix.argmax(dim=1)
        predicted_labels = labels[top1_indices]

        correct = (predicted_labels == labels).sum().item()

        return correct / total

    def _tar_at_far(
        self, similarity_matrix: torch.Tensor, labels: torch.Tensor, target_far=1e-6
    ) -> float:
        identity_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        valid_mask = ~torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)

        tar_values = []

        for i in range(similarity_matrix.shape[0]):
            current_similarity = similarity_matrix[i, valid_mask[i]]
            current_labels = identity_mask[i, valid_mask[i]]

            fprs, tprs, _ = roc_curve(
                current_labels.cpu().numpy(), current_similarity.cpu().numpy()
            )

            idx = np.where(fprs <= target_far)[0]
            tar_values.append(tprs[idx[-1]] if len(idx) > 0 else 0.0)

        return np.mean(tar_values)

    # def _tar_at_far(
    #     self, similarity_matrix: torch.Tensor, labels: torch.Tensor, target_far=1e-6
    # ) -> float:
    #     N = len(labels)
    #     triu_mask = torch.triu(
    #         torch.ones(N, N, dtype=torch.bool, device=labels.device), diagonal=1
    #     )
    #     identity_mask = labels.unsqueeze(1) == labels.unsqueeze(0)

    #     positive_sim = similarity_matrix[identity_mask & triu_mask].cpu().numpy()
    #     negative_sim = similarity_matrix[(~identity_mask) & triu_mask].cpu().numpy()

    #     thresholds = np.sort(np.concatenate([positive_sim, negative_sim]))

    #     best_tar = 0
    #     best_threshold = 0

    #     for threshold in thresholds:
    #         false_accepts = np.sum(negative_sim >= threshold)
    #         true_accepts = np.sum(positive_sim >= threshold)

    #         total_negatives = len(negative_sim)
    #         total_positives = len(positive_sim)

    #         if total_negatives == 0 or total_positives == 0:
    #             continue

    #         far = false_accepts / total_negatives
    #         tar = true_accepts / total_positives

    #         if far <= target_far and tar > best_tar:
    #             best_tar = tar
    #             best_threshold = threshold

    #     return best_tar, best_threshold

    # def _rank1_accuracy(
    #     self,
    #     gallery_features: torch.Tensor,
    #     gallery_labels: torch.Tensor,
    #     probe_features: torch.Tensor,
    #     probe_labels: torch.Tensor,
    # ):
    #     """
    #     Get rank-1 accuracy.
    #     """
    #     correct = 0
    #     total = probe_labels.size(0)

    #     similarity_matrix = torch.mm(
    #         probe_features, gallery_features.t()
    #     )  # (N_probe, N_gallery)
    #     top1_indices = similarity_matrix.argmax(dim=1)
    #     predicted_labels = gallery_labels[top1_indices]

    #     correct = (predicted_labels == probe_labels).sum().item()
    #     total = probe_labels.size(0)

    #     return correct / total

    # def _tar_at_far(
    #     self, features: torch.Tensor, labels: torch.Tensor, target_far=1e-6
    # ) -> float:
    #     similarity_matrix = torch.mm(features, features.t())  # (N_feture, N_feture)
    #     # Generate mask, remove self-match
    #     N = len(labels)
    #     triu_mask = torch.triu(
    #         torch.ones(N, N, dtype=torch.bool, device=features.device), diagonal=1
    #     )
    #     identity_mask = labels.unsqueeze(1) == labels.unsqueeze(0)

    #     # Get positive pairs and negative pairs
    #     positive_pairs = identity_mask & triu_mask
    #     negative_pairs = (~identity_mask) & triu_mask

    #     # get score
    #     positive_scores = similarity_matrix[positive_pairs]  # same ID
    #     negative_scores = similarity_matrix[negative_pairs]  # different ID

    #     far_threshold = torch.quantile(negative_scores, 1 - target_far)

    #     tar = (positive_scores > far_threshold).float().mean().item()

    #     return tar

    # def _cal_binary_cls_curve(y_hat, y_score, thresholds):
    #     """
    #     Compute the binary classification curve
    #     :param y_hat: ground-truth label, [n_sample]
    #     :param y_score: predict label, [n_sample]
    #     :param thresholds: thresholds, [n_threshold]
    #     :return: FPS: [n_threshold], TPS: [n_threshold]
    #     """
    #     assert len(y_hat) == len(y_score)

    #     y_hat = y_hat == 1

    #     n_thresh = len(thresholds)
    #     fps = np.zeros((n_thresh,))
    #     tps = np.zeros((n_thresh,))

    #     for i, thresh in enumerate(thresholds):
    #         y_preds = np.greater(y_score, thresh)
    #         y_true_preds = np.logical_and(y_preds, True)
    #         tps[i] = np.sum(np.logical_and(y_hat, y_true_preds))
    #         fps[i] = np.sum(np.logical_and(np.logical_not(y_hat), y_true_preds))

    #     return fps, tps

    # def plot_pr_curve(self, y_true: np.ndarray, y_score: np.ndarray, names):
    #     num_classes = len(names)
    #     # One-hot encoding
    #     y_true_bin = label_binarize(y_true, classes=range(num_classes))

    #     # Compute Precision-Recall
    #     mAP = 0
    #     precisions = []
    #     recalls = []

    #     for i in range(num_classes):
    #         precision, recall, _ = precision_recall_curve(
    #             y_true_bin[:, i], y_score[:, i]
    #         )
    #         precisions.append(precision)
    #         recalls.append(recall)
    #         mAP += average_precision_score(y_true_bin[:, i], y_score[:, i])

    #     max_len = max(len(p) for p in precisions)
    #     mean_precision = np.zeros(max_len)
    #     mean_recall = np.zeros(max_len)

    #     for i in range(num_classes):
    #         precision = np.interp(
    #             np.linspace(0, 1, max_len), recalls[i][::-1], precisions[i][::-1]
    #         )
    #         recall = np.interp(
    #             np.linspace(0, 1, max_len), recalls[i][::-1], recalls[i][::-1]
    #         )
    #         mean_precision += precision
    #         mean_recall += recall

    #     mean_precision /= num_classes
    #     mean_recall /= num_classes
    #     mAP /= num_classes

    #     # Plot Macro-average PR Curve
    #     plt.plot(
    #         mean_recall,
    #         mean_precision,
    #         color="blue",
    #         label=f"all classes {mAP:.3f} mAP",
    #     )
    #     plt.xlim([-0.1, 1.1])
    #     plt.ylim([-0.1, 1.1])
    #     plt.xlabel("Recall")
    #     plt.ylabel("Precision")
    #     plt.title("PR Curve", fontsize=16)
    #     plt.legend()
    #     plt.savefig(Path(self.save_dir / "pr_curve.png"))
    #     plt.close()

    # def plot_roc_curve(self, y_true: np.ndarray, y_score: np.ndarray, names):
    #     num_classes = len(names)
    #     y_true_bin = label_binarize(y_true, classes=range(num_classes))

    #     fpr = {}
    #     tpr = {}
    #     roc_auc = {}

    #     for i in range(num_classes):
    #         fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])

    #     all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    #     mean_tpr = np.zeros_like(all_fpr)

    #     for i in range(num_classes):
    #         mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    #     mean_tpr /= num_classes
    #     mean_tpr[0] = 0.0
    #     mean_tpr[-1] = 1.0
    #     mean_auc = auc(all_fpr, mean_tpr)

    #     plt.figure(figsize=(8, 6))

    #     plt.plot(
    #         all_fpr,
    #         mean_tpr,
    #         color="blue",
    #         label=f"Macro-average ROC (AUC = {mean_auc:.3f})",
    #     )

    #     plt.plot([0, 1], [0, 1], color="gray", linestyle="--")

    #     plt.xlim([-0.1, 1.1])
    #     plt.ylim([-0.1, 1.1])
    #     plt.xlabel("False Positive Rate")
    #     plt.ylabel("True Positive Rate")
    #     plt.title("ROC Curve", fontsize=16)
    #     plt.legend(loc="lower right")

    #     plt.savefig(Path(self.save_dir / "roc_curve.png"))


# class GradCAM:
#     def __init__(self, model: SHOUNet, layers: List[nn.Module]):
#         self.model = model
#         self.layers = layers
#         self.activations = []
#         self.gradients = []

#         self.model.eval()

#         for layer in layers:
#             layer.register_forward_hook(self._save_activation)
#             layer.register_full_backward_hook(self._save_gradient)

#     def _save_activation(self, module, input, output):
#         self.activations.append(output.detach())

#     def _save_gradient(self, module, grad_input, grad_output):
#         self.gradients.append(grad_output[0].detach())

#     def _normalize_cam(self, cam):
#         cam = np.maximum(cam, 0)
#         cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
#         return cam

#     def generate(self, input_tensor):
#         self.activations = []
#         self.gradients = []
#         emb = self.model(input_tensor)[0]

#         target = torch.norm(emb, p=2, dim=1)

#         self.model.zero_grad()
#         target.backward(retain_graph=True)

#         activations = self.activations[0].cpu().numpy()[0]
#         gradients = self.gradients[0].cpu().numpy()[0]

#         weights = np.mean(gradients, axis=(1, 2), keepdims=True)
#         cam = np.sum(weights * activations, axis=0)
#         cam = self._normalize_cam(cam)

#         return cam

#     def show_gradcams(
#         self, image: Union[Path, str, Image.Image, np.ndarray], cam, alpha=0.5
#     ):
#         if isinstance(image, (Path, str)):
#             image = Image.open(image)
#         if isinstance(image, Image.Image):
#             image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

#         height, width = image.shape[:2]

#         cam = cv2.resize(cam, (width, height))
#         heatmap = cv2.applyColorMap(255 - np.uint8(255 * cam), cv2.COLORMAP_JET)
#         heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

#         overlay = (image * (1 - alpha) + heatmap * alpha).astype(np.uint8)
#         cv2.imwrite(
#             "E:\\LatexFiles\\Indentification of individual sea bass\\images\\overlay_4.jpg",
#             overlay,
#         )
