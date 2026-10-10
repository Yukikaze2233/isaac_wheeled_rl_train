"""Grouped physical-delay reads on Isaac Lab's native cursor-based buffer."""
import torch

from isaaclab_experimental.utils.buffers import CircularBuffer


class GroupedDelayBuffer(CircularBuffer):
    """Keep native reset/history semantics without device-scalar hot-path checks.

    The owner always appends before reading. Reset IDs are already known, so
    priming does not need a device boolean check. One gather selects each
    feature's independent age; the full history is never rolled during stepping.
    """
    def __init__(self, max_len, count, device, groups):
        self._capacity = max_len
        super().__init__(max_len, count, device)
        self._pending_resets = []
        self._groups = torch.tensor(groups, dtype=torch.long, device=device)
        self._columns = torch.arange(len(groups), device=device)
        self._environments = self._ALL_INDICES[:, None]

    @property
    def max_length(self):
        return self._capacity

    def reset(self, batch_ids=None):
        super().reset(batch_ids)
        if batch_ids is None or len(batch_ids):
            self._pending_resets.append(slice(None) if batch_ids is None else batch_ids.clone())

    def append(self, data):
        if data.shape != (self.batch_size, len(self._groups)):
            raise ValueError("Physical delay buffer input shape differs")
        data = data.to(self._device)
        if self._buffer is None:
            self._buffer = data.unsqueeze(0).expand(self._capacity, *data.shape).clone()
        else:
            for ids in self._pending_resets:
                self._buffer[:, ids] = data[ids]
        self._pending_resets.clear()
        self._pointer = (self._pointer + 1) % self._capacity
        self._buffer[self._pointer].copy_(data)
        self._num_pushes += 1

    def read_groups(self, ages):
        if self._buffer is None:
            raise RuntimeError("Physical delay history must be appended before reading")
        valid = torch.minimum(ages, self._num_pushes[:, None] - 1).clamp_min(0)
        slots = (self._pointer - valid[:, self._groups]) % self._capacity
        return self._buffer[slots, self._environments, self._columns]

    def prime(self, data, batch_ids=None):
        """Flush selected histories to one sample without advancing the cursor.

        Other environments retain their histories and sample counts. The next
        append can then interpolate between this reset sample and new feedback.
        """
        if data.shape != (self.batch_size, len(self._groups)):
            raise ValueError("Physical delay buffer input shape differs")
        ids = slice(None) if batch_ids is None else batch_ids
        data = data.to(self._device)
        if self._buffer is None:
            self._buffer = data.unsqueeze(0).expand(self._capacity, *data.shape).clone()
            self._pointer = 0
        else:
            self._buffer[:, ids] = data[ids]
        self._num_pushes[ids] = 1
        # Priming resolves queued resets for these rows, while other queued
        # resets must still take their first sample from the next append.
        if self._pending_resets:
            pending = torch.zeros(self.batch_size, dtype=torch.bool, device=self._device)
            for reset_ids in self._pending_resets:
                pending[reset_ids] = True
            pending[ids] = False
            self._pending_resets = [pending]

    def read_groups_fractional(self, ages, angular_columns=()):
        """Interpolate independent group ages expressed in physics steps.

        Only the two bracketing samples are gathered from the native cursor.
        Angular columns follow the shortest arc around the newer coordinate.
        The owner may normalize direction vectors after interpolation.
        """
        if self._buffer is None:
            raise RuntimeError("Physical delay history must be appended before reading")
        valid = torch.minimum(ages, self._num_pushes[:, None] - 1).clamp(0, self._capacity - 1)
        lower = valid.floor().long()
        newer = self.read_groups(lower)
        older = self.read_groups(valid.ceil().long())
        delta = older - newer
        if len(angular_columns):
            angular_delta = delta[:, angular_columns]
            delta[:, angular_columns] = torch.atan2(angular_delta.sin(), angular_delta.cos())
        weight = (valid - lower)[:, self._groups].to(newer.dtype)
        return newer + weight * delta
