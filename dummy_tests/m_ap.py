from typing import *
from typing import Any, List, Optional
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from collections import defaultdict
import numpy as np
from copy import deepcopy
import math

class BaseMetricResults(dict):
    """Base metric class, that allows fields for pre-defined metrics."""

    def __getattr__(self, key: str) -> torch.Tensor:
        """Get a specific metric attribute."""
        # Using this you get the correct error message, an AttributeError instead of a KeyError
        if key in self:
            return self[key]
        raise AttributeError(f"No such attribute: {key}")

    def __setattr__(self, key: str, value: torch.Tensor) -> None:
        """Set a specific metric attribute."""
        self[key] = value

    def __delattr__(self, key: str) -> None:
        """Delete a specific metric attribute."""
        if key in self:
            del self[key]
        raise AttributeError(f"No such attribute: {key}")

class COCOMetricResults(BaseMetricResults):
    """Class to wrap the final COCO metric results including various mAP/mAR values."""

    __slots__ = (
        "map",
        "map_50",
        "map_75",
        "map_small",
        "map_medium",
        "map_large",
        "mar_1",
        "mar_10",
        "mar_100",
        "mar_small",
        "mar_medium",
        "mar_large",
        "map_per_class",
        "mar_100_per_class",
    )

class extended_map(MeanAveragePrecision):
    def __init__(self, box_format: str = "xyxy", iou_type: str = "bbox", iou_thresholds = None, rec_thresholds = None, max_detection_thresholds = None, class_metrics: bool = False, **kwargs: Any) -> None:
        super().__init__(box_format, iou_type, iou_thresholds, rec_thresholds, max_detection_thresholds, class_metrics, **kwargs)


    def _get_classes(self) -> List:
        """Return a list of unique classes found in ground truth and detection data."""
        if (len(self.detection_labels) > 0 or len(self.groundtruth_labels) > 0):
            return torch.cat(self.detection_labels + self.groundtruth_labels).unique().tolist()
        return []

    def compute(self):
        """Compute metric."""
        classes = self._get_classes()
        precisions, recalls = self._calculate(classes)
        map_val, mar_val = self._summarize_results(precisions, recalls)

        # if class mode is enabled, evaluate metrics per class
        map_per_class_values = torch.tensor([-1.0])
        mar_max_dets_per_class_values = torch.tensor([-1.0])
        if self.class_metrics:
            map_per_class_list = []
            mar_max_dets_per_class_list = []

            for class_idx, _ in enumerate(classes):
                cls_precisions = precisions[:, :, class_idx].unsqueeze(dim=2)
                cls_recalls = recalls[:, class_idx].unsqueeze(dim=1)
                cls_map, cls_mar = self._summarize_results(cls_precisions, cls_recalls)
                map_per_class_list.append(cls_map.map)
                mar_max_dets_per_class_list.append(cls_mar[f"mar_{self.max_detection_thresholds[-1]}"])

            map_per_class_values = torch.tensor(map_per_class_list, dtype=torch.float)
            mar_max_dets_per_class_values = torch.tensor(mar_max_dets_per_class_list, dtype=torch.float)

        metrics = COCOMetricResults()
        metrics.update(map_val)
        metrics.update(mar_val)
        metrics.map_per_class = map_per_class_values
        metrics[f"mar_{self.max_detection_thresholds[-1]}_per_class"] = mar_max_dets_per_class_values
        metrics.classes = torch.tensor(classes, dtype=torch.int)
        return metrics
    