import random
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class EncoderBatch:
    seqs: Tensor
    padding_mask: Tensor
    input_ids_mask: Tensor

    def numel(self) -> int:
        return self.padding_mask.sum().item()

    def size(self):
        return self.seqs.size(0)

    def split(self, index: int) -> tuple["EncoderBatch"] | tuple["EncoderBatch", "EncoderBatch"]:
        if isinstance(index, float):
            raise ValueError('split does not support floating point microbatch_size.')

        if index < 1:
            raise ValueError('split does not support microbatch_size less than 1.')

        if index > self.size():
            raise ValueError('split does not support microbatch_size greater than or equal to the batch size.')

        if index == self.size():
            return self,

        return EncoderBatch(
            self.seqs[:index],
            self.padding_mask[:index],
            self.input_ids_mask[:index]
        ), EncoderBatch(
            self.seqs[index:],
            self.padding_mask[index:],
            self.input_ids_mask[index:]
        )


@dataclass
class FlowMatchingBatch(EncoderBatch):
    x_start: Tensor
    x_t: Tensor
    noise: Tensor
    t: Tensor

    def split(self, index: int) -> tuple["FlowMatchingBatch", "FlowMatchingBatch"]:
        return FlowMatchingBatch(
            self.seqs[:index],
            self.padding_mask[:index],
            self.input_ids_mask[:index],
            self.x_start[:index],
            self.x_t[:index],
            self.noise[:index],
            self.t[:index]
        ), FlowMatchingBatch(
            self.seqs[index:],
            self.padding_mask[index:],
            self.input_ids_mask[index:],
            self.x_start[index:],
            self.x_t[index:],
            self.noise[index:],
            self.t[index:]
        )

    @classmethod
    def from_shortcut_fm_batch(cls, shortcut_fm_batch: "ShortcutFMBatch") -> "FlowMatchingBatch":
        return cls(
            shortcut_fm_batch.seqs,
            shortcut_fm_batch.padding_mask,
            shortcut_fm_batch.input_ids_mask,
            shortcut_fm_batch.x_start,
            shortcut_fm_batch.x_t,
            shortcut_fm_batch.noise,
            shortcut_fm_batch.t
        )


@dataclass
class ShortcutFMBatch(FlowMatchingBatch):
    shortcut_size: Tensor

    @classmethod
    def from_flow_matching_batch(cls, fm_batch: FlowMatchingBatch, shortcut_size: Tensor) -> "ShortcutFMBatch":
        return cls(
            fm_batch.seqs,
            fm_batch.padding_mask,
            fm_batch.input_ids_mask,
            fm_batch.x_start,
            fm_batch.x_t,
            fm_batch.noise,
            fm_batch.t,
            shortcut_size
        )

    def split(self, index: int) -> tuple["ShortcutFMBatch", "ShortcutFMBatch"]:
        fm1, fm2 = super().split(index)
        return (
            self.from_flow_matching_batch(fm1, self.shortcut_size[:index]),
            self.from_flow_matching_batch(fm2, self.shortcut_size[index:])
        )


def collate(
        batch: list[dict[str, Tensor]],
        mark_first_padding: bool = False,
        mark_second_padding: bool = False
) -> EncoderBatch:
    """Collates a batch of dictionaries into an EncoderBatch.

    Args:
        batch: A list of dictionaries, where each dictionary represents a single
            item in the batch and contains the keys "seqs", "padding_mask", and
            "input_ids_mask" with corresponding tensors.
        mark_first_padding: If True, marks the first padding token (0) as 1 in padding_mask.
        mark_second_padding: If True, marks the second padding token (0) as 1 in padding_mask.

    Returns:
        An EncoderBatch object containing the collated tensors.
    """

    random.shuffle(batch)

    # Transpose the list of dictionaries into a dictionary of lists
    transposed_batch = {k: [item[k] for item in batch] for k in batch[0]}

    # Stack the tensors along the first dimension (batch dimension)
    collated_batch = {k: torch.stack(v) for k, v in transposed_batch.items()}

    # Modify padding_mask if requested
    if mark_first_padding or mark_second_padding:
        padding_mask = collated_batch["padding_mask"]
        batch_size, seq_len = padding_mask.shape

        for i in range(batch_size):
            # Find indices where padding starts (0s)
            padding_indices = (padding_mask[i] == 0).nonzero(as_tuple=True)[0]

            if len(padding_indices) > 0 and mark_first_padding:
                # Mark first padding token as 1
                padding_mask[i, padding_indices[0]] = 1

            if len(padding_indices) > 1 and mark_second_padding:
                # Mark second padding token as 1
                padding_mask[i, padding_indices[1]] = 1

        collated_batch["padding_mask"] = padding_mask

    return EncoderBatch(**collated_batch)