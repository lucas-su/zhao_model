# import torch
#
# class confusion:
#     _device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
#     def __init__(self, n_classes: int = 10):
#         self._matrix = torch.zeros(n_classes * n_classes).to(self._device)
#         self._n = n_classes
#
#     def cpu(self):
#         self._matrix.cpu()
#
#     def cuda(self):
#         self._matrix.cuda()
#
#     def to(self, device: str):
#         self._matrix.to(device)
#
#     def __add__(self, other):
#         if isinstance(other, ConfusionMatrix):
#             self._matrix.add_(other._matrix)
#         elif isinstance(other, tuple):
#             self.update(*other)
#         else:
#             raise NotImplemented
#         return self
#
#     def update(self, prediction: torch.tensor, label: torch.tensor):
#         conf_data = prediction * self._n + label
#         conf = conf_data.bincount(minlength=self._n * self._n)
#         self._matrix.add_(conf)
#
#     @property
#     def value(self):
#         return self._matrix.view(self._n, self._n).T

import numpy as np
import scipy
import scipy.ndimage

def conf_scores_weighted(candidate, gt, beta=1.0):
    """
    Compute the Weighted F-beta measure (as proposed in "How to Evaluate Foreground Maps?" [Margolin et al. - CVPR'14])
    Original MATLAB source code from:
        [https://cgm.technion.ac.il/Computer-Graphics-Multimedia/Software/FGEval/resources/WFb.m]
    :param candidate: FG - Binary/Non binary candidate map with values in the range [0-1]
    :param gt: Binary ground truth map
    :param beta: attach 'beta' times as much importance to Recall as to Precision (default=1)
    :result: the Weighted F-beta score
    """

    candidate = np.array(candidate.cpu())
    gt = np.array(gt.cpu())

    if np.min(candidate) < 0.0 or np.max(candidate) > 1.0:
        raise ValueError("'candidate' values must be inside range [0 - 1]")

    if gt.dtype in [np.bool, np.bool_, np.bool8]:
        gt_mask = gt
        not_gt_mask = np.logical_not(gt_mask)
        gt = np.array(gt, dtype=candidate.dtype)
    else:
        if not np.all(np.isclose(gt, 0) | np.isclose(gt, 1)):
            raise ValueError("'gt' must be a 0/1 or boolean array")
        gt_mask = np.isclose(gt, 1)
        not_gt_mask = np.logical_not(gt_mask)
        gt = np.asarray(gt, dtype=candidate.dtype)

    E = np.abs(candidate - gt)
    dist, idx = scipy.ndimage.morphology.distance_transform_edt(not_gt_mask, return_indices=True)

    # Pixel dependency
    Et = np.array(E)
    # To deal correctly with the edges of the foreground region:
    Et[not_gt_mask] = E[idx[0, not_gt_mask], idx[1, not_gt_mask]]
    sigma = 5.0
    EA = scipy.ndimage.gaussian_filter(Et, sigma=sigma, truncate=3 / sigma,
                                       mode='constant', cval=0.0)
    min_E_EA = np.minimum(E, EA, where=gt_mask, out=np.array(E))

    # Pixel importance
    B = np.ones(gt.shape)
    B[not_gt_mask] = 2 - np.exp(np.log(1 - 0.5) / 5 * dist[not_gt_mask])
    Ew = min_E_EA * B

    # Final metric computation
    eps = np.spacing(1)
    TPw = np.sum(gt) - np.sum(Ew[gt_mask])
    FPw = np.sum(Ew[not_gt_mask])
    TNw = (1-np.sum(Ew[gt_mask]))*(1-np.sum(gt))
    FNw = np.sum(Ew[gt_mask]) * np.sum(gt)
    # if gt_mask.any():
    #     R = 1 - np.mean(Ew[gt_mask])  # Weighed Recall
    # else:
    #     R = 0
    # P = TPw / (eps + TPw + FPw)  # Weighted Precision

    # Q = 2 * (R * P) / (eps + R + P)  # Beta=1
    # Q = (1 + beta**2) * (R * P) / (eps + R + (beta * P)) # return TPw and FPw and calculate Q over entire epoch

    return TPw, FPw, TNw, FNw