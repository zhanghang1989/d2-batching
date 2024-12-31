from typing import Any, Dict, List, Tuple, Union
import torch
from detectron2.structures import Instances

class BatchInstances(Instances):
    """Representing Batched version of Instances for Tracing mode export
    BatchInstances can be converted from/to a List of Instances
    """
    def __init__(self, image_sizes: torch.Tensor, batch_indices: torch.Tensor, **kwargs):
        self._image_sizes = image_sizes
        self._batch_indices = batch_indices
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_sizes(self) -> torch.Tensor:
        return self._image_sizes

    @property
    def batch_indices(self) -> torch.Tensor:
        return self._batch_indices

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instances":
        """Get access to the batch items using indices or slice
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
        pass
    
    def from_list(self, List[Instances]) -> "BatchInstances":
        pass

    def to_list(self) -> List[Instances]:
        pass
