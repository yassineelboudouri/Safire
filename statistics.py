import numpy as np


class TrainingStatistics:
    """
    Computes statistics per epoch.
    """

    def __init__(self):
        self.g_loss = float("inf")
        self.d_loss = float("inf")
        self.g_losses = []
        self.d_losses = []

    def on_training_step(self, losses):
        if "g_loss" in losses.keys():
            self.g_losses.append(losses["g_loss"])
            self.g_loss = np.mean(self.g_losses)
        else:
            self.d_losses.append(losses["d_loss"])
            self.d_loss = np.mean(self.d_losses)

    def get_progbar_postfix(self):
        return {
            "g_loss": "%1.4f" % self.g_loss,
            "d_loss": "%1.4f" % self.d_loss
        }
