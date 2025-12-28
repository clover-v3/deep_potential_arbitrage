import numpy as np
import pandas as pd

class PhaseSpaceController:
    """
    Control Logic: Converts Physical Quantities to Trading Signals.

    Equations:
    Signal = - Force * sqrt(Stiffness) * Filter
    """
    def __init__(self, stiffness_threshold: float = 0.5, velocity_filter: bool = True):
        self.stiffness_thresh = stiffness_threshold
        self.velocity_filter = velocity_filter

    def compute_signal(
        self,
        force: np.ndarray,
        stiffness: np.ndarray,
        velocity: np.ndarray = None
    ) -> np.ndarray:
        """
        Args:
            force: (N, D) - Restoring force.
            stiffness: (N, 1) - Confidence/tightness.
            velocity: (N, D) - Current price trend (f_t - f_{t-1}).

        Returns:
            Signal: (N, D) - Target weights (unscaled).
        """
        # 1. Base Signal: - Force
        # Force > 0 means f is "too high" -> Pull down -> Short.
        # So Signal ~ -Force.

        # 2. Confidence Scaling
        # signal = -force * sqrt(k)
        # If k is small (loose cluster), signal is weak.
        confidence = np.sqrt(stiffness)
        # Apply threshold? Or just soft scaling.
        # Let's use soft scaling.

        raw_signal = -force * confidence

        # 3. Velocity Filter (Kinetic Energy Check)
        if self.velocity_filter and velocity is not None:
            # Check Divergence (Catching Falling Knife)
            # If Force > 0 (Pull Down) but Velocity > 0 (Still Shooting Up) -> Diverging
            # Product Force * Velocity > 0 means Diverging?
            # Yes. Force opposes motion in harmonic oscillator ONLY when returning to equilibrium.
            # No, wait.
            # Harmonic Oscillator: F = -kx.
            # If x > 0 (Displaced right), F < 0 (Pull left).
            # If v > 0 (Moving right), F*v < 0. (Slowing down, but diverging from origin).
            # If v < 0 (Moving left), F*v > 0. (Speeding up towards origin).

            # We want to enter when it is "Turning" or "Converging".
            # Diverging (x increasing away from mean): Dangerous.
            # Converging (x decreasing towards mean): Safe.

            # In our Force definition: Force = 2Lf.
            # If f > 0 (Overvalued), Force > 0 ?
            # L = D - A. Lf = Df - Af.
            # If f_i is high vs neighbors, Lf_i > 0.
            # So Force > 0 means "Restoring Force points Down"?
            # Wait. F = ma. If F points down, it accelerates down.
            # If our "Force" term is just the Gradient 2Lf...
            # The restoring force is -Grad E = -2Lf.
            # So Physical Force G = -2Lf.

            # Let's clarify inputs.
            # Model returns 'force' key as 2Lf.
            # So Physical Restoring Force = - input_force.

            physical_force = -force

            # Dot Product P = Physical_F * Velocity
            # If P > 0: Force and Velocity aligned -> Accelerating in direction of force -> Good (Regression started).
            # If P < 0: Force opposes Velocity -> Decelerating (Still moving away) -> Bad (Knife Catching).

            # Refined Logic:
            power = physical_force * velocity

            # Filter Mask: Only trade if Power > -epsilon (allow slight opposition near turning point)
            # Or strictly Power > 0.
            diverging_mask = power < 0

            # Apply mask
            raw_signal[diverging_mask] = 0.0

        return raw_signal
