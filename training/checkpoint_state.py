"""Explicit epoch boundaries for new and legacy training checkpoints."""


def next_training_epoch(checkpoint):
    epoch = int(checkpoint['epoch'])
    phase = checkpoint.get('checkpoint_phase')
    if phase == 'after_completed_epoch':
        completed = int(checkpoint['completed_epochs'])
        if completed != epoch + 1 or int(checkpoint['next_epoch']) != completed:
            raise ValueError('Inconsistent completed-epoch checkpoint metadata')
        return completed
    if phase is None:
        # Legacy model_latest/model_epoch files were saved BEFORE this epoch.
        # Their weights have completed epochs [0, epoch), not [0, epoch].
        return epoch
    raise ValueError('Unsupported checkpoint phase: ' + str(phase))


def completed_epoch_fields(epoch):
    if epoch < 0:
        raise ValueError('epoch must be nonnegative')
    return dict(epoch=epoch, checkpoint_phase='after_completed_epoch',
                completed_epochs=epoch + 1, next_epoch=epoch + 1)
