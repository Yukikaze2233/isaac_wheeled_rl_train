"""Training measurements and writer batching; no task or policy changes."""

def timed_call(timer, phase, function, *args, **kwargs):
    if timer is None:
        return function(*args, **kwargs)
    with timer.measure(phase):
        return function(*args, **kwargs)


def log_scalar_batch(writer, values, step):
    """Batch drained Python scalars without changing tags or writer backends."""
    if writer is None or not values:
        return
    from itertools import islice
    from tensorboard.compat.proto.summary_pb2 import Summary
    from torch.utils.tensorboard import SummaryWriter

    if type(writer) is not SummaryWriter:
        for tag, value in values.items():
            writer.add_scalar(tag, value, step)
        return
    items = iter(values.items())
    while batch := list(islice(items, 512)):
        summary = Summary(value=[Summary.Value(tag=tag, simple_value=value)
                                 for tag, value in batch])
        writer._get_file_writer().add_summary(summary, step)


def incoming_action_clipping_metrics(config, actions, loss_population_mask):
    """Read raw policy-order storage once/update, using the direct-loss mask.

    Four leg axes precede two wheels. Incoming actions have already passed the
    entry's finite check, but may differ from delayed/held applied commands.
    Target excess measures the SAME raw clamp in rad/rad_s, before leg angle
    wrapping, not a second target clamp. Empty populations are explicitly marked.
    No inputs are mutated and no counters survive an update.
    """
    import torch
    control = config["v5_control"]
    leg_bound = control["action_clip"]
    wheel_bound = control.get("wheel_action_clip", leg_bound)
    bounds = actions.new_tensor([leg_bound] * 4 + [wheel_bound] * 2)
    mask = loss_population_mask.reshape(-1)
    excess = torch.where(mask[:, None],
        (actions.detach().reshape(-1, 6).abs() - bounds).clamp_min(0), 0.)
    # Exactly 13 aggregate scalars cross to the host; never save a rollout copy.
    values = torch.cat((mask.sum().reshape(1).double(), (excess > 0).sum(0).double(),
                        excess.sum(0, dtype=torch.float64))).cpu().tolist()
    samples, counts, excess = int(values[0]), values[1:7], values[7:]
    prefix = "ActionClipping/incoming_direct_loss/"
    result = {prefix + "loss_transitions": samples,
              prefix + "has_loss_samples": int(samples > 0)}
    for group, start, stop, scale, unit in (
            ("legs", 0, 4, abs(control["leg_position_scale"]), "rad"),
            ("wheels", 4, 6, abs(control["wheel_velocity_scale"]), "rad_s")):
        denominator = samples * (stop - start)
        count = int(sum(counts[start:stop]))
        tag = prefix + group + "/"
        result.update({tag + "axis_samples": denominator, tag + "raw_clip_count": count,
            tag + "raw_clip_fraction": count / max(denominator, 1),
            tag + "target_clip_excess_mean_" + unit: sum(excess[start:stop]) * scale / max(denominator, 1)})
    return result


def fixed_flat_rehearsal_counts(loss_mask, lane_indices, bucket_count):
    """GPU reduction only; commit after PPO succeeds, never from a failed rollout."""
    import torch
    if (loss_mask.dtype != torch.bool or loss_mask.ndim != 2
            or lane_indices.dtype != torch.int64 or lane_indices.shape != (loss_mask.shape[1],)
            or lane_indices.device != loss_mask.device):
        raise ValueError("Rehearsal exposure requires the actual time-by-native-lane loss mask")
    result = torch.zeros(bucket_count, dtype=torch.int64, device=loss_mask.device)
    return result.scatter_add_(0, lane_indices, loss_mask.sum(0, dtype=torch.int64))


def log_recovery_outcomes(writer, recovery, step):
    """Write producer episode counts once, outside the physics/rollout loop."""
    curriculum = getattr(recovery, "pose_curriculum", None)
    snapshot = getattr(curriculum, "rollout_outcome_metrics", None)
    if writer is None or snapshot is None:
        return
    for tag, value in snapshot().items():
        writer.add_scalar(tag, value, step)
