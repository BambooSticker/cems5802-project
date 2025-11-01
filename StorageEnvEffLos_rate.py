import gymnasium as gym
from gymnasium import spaces
import numpy as np


class IdealStorageShutdownEnv(gym.Env):
    """
    Gym environment based on the IdealStorageShutdown MILP model.
    
    The environment uses true value ranges in its action and observation space definitions.
    
    The agent chooses:
      - A continuous power uptake decision in [P_min, P_max].
      - A "switch" action in [0, 1], interpreted as binary (off=0, on=1).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, S_max, Delta_max, theta_min, theta_max, zeta, cEl=None,
                penalty_ramping_lambda=15, penalty_demand_lambda=50, penalty_storage_lambda=50):
        super().__init__()

        # ---------------------
        # Environment settings
        # ---------------------
        self.S_max = S_max      # Max storage (surplus energy)

        self.P_nom = 1.0       # Nominal power (baseline consumption)
        self.theta_min = theta_min       # Minimum power (on)
        self.theta_max = theta_max        # Maximum power (on)
        self.Delta_max = Delta_max    # Max ramping step
        self.zeta = zeta

        # Penalty coefficient:
        self.penalty_ramping_lambda = penalty_ramping_lambda
        self.penalty_demand_lambda = penalty_demand_lambda
        self.penalty_storage_lambda = penalty_storage_lambda

        # If an array of electricity prices is given, store it
        self.cEl = cEl
        self.horizon = len(self.cEl)  # Number of timesteps in an episode

        self.price_range = (self.cEl.min(), self.cEl.max())

        # action space bounds (true value range)
        self.act_low = np.array([
            self.theta_min * self.P_nom,
        ], dtype=np.float32)

        self.act_high = np.array([
            (1 + self.theta_max) * self.P_nom,
        ], dtype=np.float32)
        
        # Normalized action space: [-1,1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # observation space bounds (true value range)
        self.obs_low = np.array([
            -self.S_max,                    # S
            self.theta_min * self.P_nom,      # P_prev
            self.price_range[0],              # price
            -10,              # delta_price
        ], dtype=np.float32)

        self.obs_high = np.array([
            self.S_max,                     # S
            (1 + self.theta_max) * self.P_nom,  # P_prev
            self.price_range[1],              # price
            10,              # delta_price
        ], dtype=np.float32)

        # Normalized observation space: [-1,1]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Initialize internal state
        self.reset()

    def _normalize(self, value, low, high):
        """Normalize a scalar value to the range [-1, 1]."""
        return 2 * ((value - low) / (high - low + 1e-8)) - 1

    def _denormalize_action(self, norm_action):
        """Convert an action from normalized [-1,1] space to its true range."""
        return (norm_action + 1) * (self.act_high - self.act_low) / 2 + self.act_low

    def _get_obs(self):
        """
        Return a normalized 4D NumPy array with the state, mapped to [-1,1].
        """
        raw_obs = np.array([
            self.S,
            self.P_prev,
            self.price,
            self.delta_price,
        ], dtype=np.float32)

        # Normalize each element based on its corresponding min and max
        normalized_obs = np.empty_like(raw_obs)
        normalized_obs[0] = self._normalize(self.S, self.obs_low[0], self.obs_high[0])
        normalized_obs[1] = self._normalize(self.P_prev, self.obs_low[1], self.obs_high[1])
        normalized_obs[2] = self._normalize(self.price, self.obs_low[2], self.obs_high[2])
        normalized_obs[3] = self._normalize(self.delta_price, self.obs_low[3], self.obs_high[3])
        
        return normalized_obs

    def reset(self, seed=42, options=None, S0_rand=True):
        if seed is not None:
            np.random.seed(seed)

        self.unmatched_demand = 0.0
        
        # Time step
        self.t = 0

        self.delta_price = 0.0

        # Storage level
        if S0_rand:
            self.S = np.random.uniform(-self.S_max, self.S_max)
        else:
            self.S = 0.0

        # Previous power (start at nominal)
        self.P_prev = self.P_nom

        # Prepare the price array for this episode
        if self.cEl is not None and len(self.cEl) > self.horizon:
            self.episode_prices = self.cEl
        else:
            # fallback: use entire cEl as is
            self.episode_prices = self.cEl if self.cEl is not None else np.zeros(self.horizon)

        self.price = self.episode_prices[0] if len(self.episode_prices) > 0 else 0.0

        # Initialize penalties
        self.penalty_storage = 0.0
        self.penalty_ramping = 0.0
        self.penalty_demand = 0.0

        normalized_obs = self._get_obs()
        return normalized_obs, {}

    def step(self, normalized_action):
        # If the episode is over (just a safety check)
        if self.t >= self.horizon:
            return self._get_obs(), 0.0, True, False, {}

        self.t += 1

        # Reset penalties each step
        self.unmatched_demand = 0.0
        self.over_product = 0.0
        self.penalty_storage = 0.0
        self.penalty_ramping = 0.0
        self.penalty_demand = 0.0

        # Update price for current step
        if self.t < len(self.episode_prices):
            price_prev = self.price
            self.price = self.episode_prices[self.t]
            self.delta_price = self.price - price_prev
            
        # Denormalize action from [-1,1] to true action space
        action = self._denormalize_action(normalized_action)
        power_uptake = float(action[0])

        # ---------------------------
        # Actual power uptake
        # ---------------------------
        # Compute efficiency loss
        eff_rate = (1 - self.zeta * ((self.P_nom - power_uptake) / (self.P_nom - self.P_nom * self.theta_min))**2)
        actual_power = eff_rate * power_uptake
        
        # Upper and lower bounds of power uptake
        lower_bound = self.P_prev - self.Delta_max
        upper_bound = self.P_prev + self.Delta_max

        # Ramping limit penalty
        if power_uptake > upper_bound:
            self.penalty_ramping = power_uptake - upper_bound
        if power_uptake < lower_bound:
            self.penalty_ramping = lower_bound - power_uptake
        
        # ---------------------------
        # Update storage
        # ---------------------------
        S_new = self.S + (actual_power - self.P_nom)
        
        if S_new < -self.S_max:
            # If the storage level is too low to meet demand, demand penalty charged
            self.unmatched_demand = (abs(S_new) - self.S_max)
            self.penalty_demand = self.unmatched_demand
                
        if S_new > self.S_max:
            # If the production is beyond the storage capacity
            self.over_product = (abs(S_new) - self.S_max)
            self.penalty_storage = self.over_product
                
        S_new = np.clip(S_new, -self.S_max, self.S_max)

        # ---------------------------
        # Compute reward
        # ---------------------------
        cost = self.price * power_uptake

        self.penalty_ramping *= self.penalty_ramping_lambda
        self.penalty_demand *= self.penalty_demand_lambda
        self.penalty_storage *= self.penalty_storage_lambda
        
        penalty_sum = (
            self.penalty_ramping**2 +
            self.penalty_demand**2 + 
            self.penalty_storage**2
        )
            
        reward = -cost - 1 * penalty_sum

        # ---------------------------
        # Store updated state
        # ---------------------------
        self.S = S_new
        self.P_prev = actual_power

        done = (self.t >= self.horizon)

        normalized_obs = self._get_obs()
            
        return normalized_obs, reward, done, False, {}

    def render(self):
        print(
            f"Time={self.t}, S={self.S:.2f}, P_prev={self.P_prev:.2f}, "
            f"price={self.price:.3f}, "
            f"demand_pen={self.penalty_demand:.2f}, ramp_pen={self.penalty_ramping:.2f}, "
            f"unmatched demand={self.unmatched_demand:.2f}"
        )

    def close(self):
        pass
