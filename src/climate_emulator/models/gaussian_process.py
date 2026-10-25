"""Optional full-field GP with the official additive feature-group kernel."""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from climate_emulator.data.preprocessing import PreparedData
from climate_emulator.models.base import Emulator, restore_field


class GaussianProcessEmulator(Emulator):
    """Exact full-field shared-kernel GP; never silently compresses target fields."""

    def __init__(self, maxiter: int = 1000):
        self.maxiter = maxiter

    def fit(self, data: PreparedData, validation: PreparedData | None = None) -> "GaussianProcessEmulator":
        """Optimize the shared covariance with optional GPflow dependency."""
        import gpflow
        if data.features.shape[1] != 12:
            raise ValueError("Official GP groups require exactly five EOFs per aerosol")
        self.mask, self.scale = data.mask.copy(), data.scale
        self.scaler = StandardScaler().fit(data.features)
        x = self.scaler.transform(data.features).astype(np.float64)
        y = (data.target[:, self.mask] / self.scale).astype(np.float64)
        kernel = (gpflow.kernels.Matern32(active_dims=[0]) + gpflow.kernels.Matern32(active_dims=[1]) + gpflow.kernels.Matern32(lengthscales=np.ones(5), active_dims=list(range(2, 7))) + gpflow.kernels.Matern32(lengthscales=np.ones(5), active_dims=list(range(7, 12))))
        model = gpflow.models.GPR((x, y), kernel, mean_function=gpflow.mean_functions.Constant())
        gpflow.optimizers.Scipy().minimize(model.training_loss, model.trainable_variables, options={"maxiter": self.maxiter})
        # Freeze an explicit numeric posterior, avoiding serialization of TF graphs.
        covariance = model.kernel(x).numpy() + np.eye(len(x)) * float(model.likelihood.variance.numpy())
        self.cholesky = np.linalg.cholesky(covariance)
        self.mean = np.asarray(model.mean_function(x).numpy()[0])
        self.alpha = np.linalg.solve(self.cholesky.T, np.linalg.solve(self.cholesky, y - self.mean))
        self.x = x
        self.kernel_parameters = [{"variance": float(k.variance.numpy()), "lengthscales": np.asarray(k.lengthscales.numpy()), "active_dims": list(k.active_dims)} for k in model.kernel.kernels]
        self.training_keys = list(data.keys)
        return self

    def predict(self, data: PreparedData) -> np.ndarray:
        """Evaluate a frozen Matérn posterior using NumPy, without GPflow imports."""
        x = self.scaler.transform(data.features)
        covariance = np.zeros((len(x), len(self.x)))
        for parameter in self.kernel_parameters:
            dims = parameter["active_dims"]
            delta = (x[:, None, dims] - self.x[None, :, dims]) / parameter["lengthscales"]
            radius = np.sqrt(3 * np.sum(delta ** 2, axis=-1))
            covariance += parameter["variance"] * (1 + radius) * np.exp(-radius)
        prediction = (self.mean + covariance @ self.alpha) * self.scale
        return restore_field(prediction, self.mask)
