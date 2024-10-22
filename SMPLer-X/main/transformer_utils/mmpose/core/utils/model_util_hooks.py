# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import HOOKS
from mmengine.hooks import Hook

@HOOKS.register_module()
class ModelSetEpochHook(Hook):
    """The hook that tells model the current epoch in training."""

    def __init__(self):
        pass

    def before_epoch(self, runner):
        runner.model.module.set_train_epoch(runner.epoch + 1)
