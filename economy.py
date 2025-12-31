
import numpy as np
from firm import Firm
from household import Household


class Economy:
    """
    Economy (two-firm, many-household ABM)

    Core design:
    - Two firms supply differentiated baskets that differ in greenness share (s).
    - Each household allocates a unit mass of consumption across the two firms (q1 + q2 = 1).
    - Households choose quantities by grid-search utility maximisation (see Household.choose_quantities_grid).
    - Firms update prices via a Lerner-style rule using an estimated (arc) elasticity.
    - A single scalar "peer_signal" (social norm) is updated each cycle from the previous cycle's market outcome.
    - Tipping is detected when Firm 1's market share exceeds a threshold for a persistence window.

    Reproducibility:
    - All randomness comes from self.rng = np.random.default_rng(seed),
      which is passed into firms and used to draw household heterogeneity.
    """

    def __init__(
        self,
        n_households=500,
        seed=None,
        beta_a=2.0,
        beta_b=8.0,
        eta_a=2.0,
        eta_b=8.0,
        lam_mu=-0.2,
        lam_sigma=0.6,
        beta_scale=1.0,
        eta_scale=1.0,
        rho=0.2,
        gamma=0.5,
        delta=0.5
    ):
        """
        Initialise the economy, create agents, and set initial conditions.

        Household heterogeneity at t=0:
        - beta ~ Beta(beta_a, beta_b) * beta_scale
          Interpreted as intrinsic taste for greenness (positive utility weight).
        - eta  ~ Beta(eta_a, eta_b) * eta_scale
          Interpreted as disutility weight on embodied emissions (positive penalty weight).
        - lam  ~ Lognormal(lam_mu, lam_sigma)
          Interpreted as price penalty weight (positive), where lam_mu can be negative
          but the draw is still strictly positive (because lognormal > 0).

       Shared behavioural parameters (passed to every Household):
        - rho   : habit adjustment speed (controls how quickly habit updates toward last period’s choice)
        - delta : social reinforcement / salience weight in utility (weights the contemporaneous peer_signal term)

   
        Notes on initialisation:
        - Firms are created first, then initial prices are set at tau=0, sigma=0 (pre-policy baseline).
        - Households are created next and their habit dictionaries are initialised once firm IDs are known.
        """
        # Create a local random number generator (reproducible with seed)
        self.rng = np.random.default_rng(seed)

        # Storage containers for agents
        self.firms = []         # list of Firm objects
        self.households = {}    # dict: household_id -> Household object

        # Store distribution parameters for heterogeneous household tastes/penalties
        self.beta_a = beta_a            # Beta distribution shape for beta
        self.beta_b = beta_b
        self.eta_a = eta_a              # Beta distribution shape for eta
        self.eta_b = eta_b
        self.lam_mu = lam_mu            # lognormal mean parameter for lam
        self.lam_sigma = lam_sigma      # lognormal sigma parameter for lam
        self.beta_scale = beta_scale    # optional scaling of beta draws
        self.eta_scale = eta_scale      # optional scaling of eta draws

        # Habit + peer parameters (same for all households in a given run)
        self.rho = rho
        self.gamma = gamma
        self.delta = delta

        # Peer signal is a scalar social norm, updated each cycle (start neutral at t=0)
        self.peer_signal = 0.0

        # --- Create agents at t=0 (initial conditions) ---

        self.create_firms()  # instantiate Firm objects
        for firm in self.firms:
            # Set initial (baseline) prices before any policy is applied
            firm.set_initial_price(tau=0.0, sigma=0.0)

        self.create_households(n_households)  # instantiate Household objects
        self.init_household_habits()          # initialise each household's habit dictionary

        # --- Tipping detection and history storage ---

        self.tipping_threshold = 0.65     # market share threshold for "tipped" state (Firm 1)
        self.tipping_persistence = 10     # number of consecutive cycles above threshold required
        self._above_counter = 0           # internal counter of consecutive above-threshold cycles
        self.tipping_point = None         # cycle index where tipping is first recorded (if any)

        self.market_shares = []           # list of dicts: {firm_id: share} per cycle

    def create_firms(self):
        """
        Create two firms at t=0 with fixed green shares (core specification).

        Notes:
        - firm_id=1 is typically the "greener" firm (higher green_share).
        - firm_id=2 is typically the "less green" firm.
        - rng is passed in so firm experimentation (if any) is reproducible.
        """
        # Create exactly two firms (two-option environment)
        self.firms = [
            Firm(firm_id=1, green_share=0.7, rng=self.rng),  # greener firm
            Firm(firm_id=2, green_share=0.3, rng=self.rng),  # less-green firm
        ]

    def create_households(self, n_households):
        """
        Create households at t=0 with heterogeneous preferences drawn from distributions.

        Distributions (drawn once at initialisation):
        - beta ~ Beta(beta_a, beta_b) * beta_scale
        - eta  ~ Beta(eta_a, eta_b) * eta_scale
        - lam  ~ Lognormal(lam_mu, lam_sigma)

        Implementation notes:
        - Beta draws are in (0,1); scale factors allow calibration beyond that range.
        - Lognormal draws are strictly positive; consistent with interpreting lam as a penalty weight.
        - rho, gamma, delta are passed to every Household to define habit/peer strength.
        """
        # Draw heterogeneous intrinsic green tastes (beta) for all households
        betas = self.rng.beta(self.beta_a, self.beta_b, size=n_households) * self.beta_scale

        # Draw heterogeneous emissions penalty weights (eta) for all households
        etas = self.rng.beta(self.eta_a, self.eta_b, size=n_households) * self.eta_scale

        # Draw heterogeneous price penalty weights (lam) for all households (lognormal > 0)
        lams = self.rng.lognormal(mean=self.lam_mu, sigma=self.lam_sigma, size=n_households)

        # Create household objects and store them by household_id
        self.households = {}
        for i in range(n_households):
            self.households[i] = Household(
                household_id=i,       # unique ID
                beta=float(betas[i]), # intrinsic green taste (utility weight)
                eta=float(etas[i]),   # emissions penalty weight
                lam=float(lams[i]),   # price penalty weight
                rho=self.rho,         # habit update speed
                gamma=self.gamma,     # habit utility weight
                delta=self.delta,     # peer utility weight
            )

    def init_household_habits(self):
        """
        Initialise each household's habit dictionary once firms exist.

        For two firms, the default "neutral" initial habit is equal weight:
            habit[firm1] = habit[firm2] = 0.5

        This ensures habit utility is well-defined from the first decision onwards.
        """
        # Collect firm IDs (keys used in q dictionaries)
        firm_ids = [f.firm_id for f in self.firms]

        # Neutral initial habit weight per firm (e.g., 1/2 for two firms)
        initial_value = 1.0 / len(firm_ids)

        # Initialise each household's habit dictionary
        for i in self.households:
            self.households[i].init_habit(firm_ids, initial_value=initial_value)

    def compute_peer_signal(self):
        """
        Return the peer signal households observe in the current cycle.

        Interpretation:
        - peer_signal is a scalar summary of social norms / observed behaviour.
        - In this implementation, it is updated at the end of each cycle using the
          previous cycle's market share of Firm 1 (the greener firm).
        """
        return self.peer_signal  # households take this as given at the start of the cycle

    def run_cycle(self, tau, sigma):
        """
        Execute one simulation cycle.

        Cycle order (state timing is important):
        1) Read peer signal from last cycle (social norm observed at start of cycle)
        2) Households choose quantities (q) given prices, greenness, emissions, and peer signal
        3) Update household habits using chosen quantities (habit evolves endogenously)
        4) Aggregate market shares across households
        5) Update peer signal for the NEXT cycle (based on current market outcome)
        6) Record firm quantities (for elasticity learning in the firm's price rule)
        7) Firms update prices for the NEXT cycle (learning/adjustment step)
        8) Save market share history
        9) Check tipping: share threshold + persistence window

        Returns:
        - firm_shares: dict {firm_id: market_share} for this cycle
        """
        # 1) Peer signal observed this cycle (from last cycle outcome)
        peer_signal = self.compute_peer_signal()

        # 2) Households choose quantities (each household chooses a split across two firms)
        quantities = {}  # dict: household_id -> {firm_id: q_k}
        for i, household in self.households.items():
            # Build the current state dictionaries required by Household.choose_quantities_grid
            prices = {f.firm_id: f.price for f in self.firms}               # p_k
            greenness = {f.firm_id: f.greenness() for f in self.firms}      # g_k
            emissions = {f.firm_id: f.emissions_intensity() for f in self.firms}  # e_k
            alpha = {f.firm_id: 1.0 for f in self.firms}                    # baseline weights (here symmetric)

            # Household chooses q = {firm_id_1: q1, firm_id_2: q2} by grid search
            quantities[i] = household.choose_quantities_grid(
                p=prices,
                g=greenness,
                e=emissions,
                alpha=alpha,
                peer_signal=peer_signal
            )

            # 3) Update habit using the just-chosen quantities (q becomes "q_prev" next cycle)
            household.update_habit(quantities[i])

        # 4) Aggregate market shares across households (firm-level outcomes)
        firm_shares = self.calculate_market_shares(quantities)

        # 5) Update peer signal for NEXT cycle
        # Here: peer_signal_{t+1} = market_share of Firm 1 in cycle t
        self.peer_signal = firm_shares[1]

        # 6) Record firm-level quantities for elasticity learning in firms
        # Each household consumes a unit mass (q1+q2=1), so total demand per cycle is n_households.
        total = len(self.households)
        for firm in self.firms:
            # Convert share to total quantity purchased from firm in this cycle
            firm.quantity_history.append(firm_shares[firm.firm_id] * total)

        # 7) Firms update prices for NEXT cycle (cost depends on current policy tau, sigma)
        for firm in self.firms:
            firm.price_update(tau=tau, sigma=sigma)

        # 8) Save share history every cycle (used for plots, tipping checks, etc.)
        self.market_shares.append(firm_shares)

        # 9) Tipping detection: threshold + persistence
        # Tipping is recorded when Firm 1 share is >= threshold for 'tipping_persistence' consecutive cycles.
        if self.tipping_point is None:
            if firm_shares[1] >= self.tipping_threshold:
                self._above_counter += 1   # increment consecutive-above counter
            else:
                self._above_counter = 0    # reset if the condition is broken

            if self._above_counter >= self.tipping_persistence:
                # Store the first cycle of the persistence window as the tipping point
                self.tipping_point = len(self.market_shares) - self.tipping_persistence

        return firm_shares

    def calculate_market_shares(self, quantities):
        """
        Calculate market shares for each firm from household quantity allocations.

        Inputs:
        - quantities: dict {household_id: {firm_id: q_k}}

        Output:
        - dict {1: firm_1_share, 2: firm_2_share}

        Notes:
        - Each household chooses q values that sum to 1 (up to numerical eps).
        - total_quantity aggregates across all households and both firms.
        """
        # Total quantity purchased across all households and firms
        total_quantity = sum(sum(q.values()) for q in quantities.values())

        # Total quantity purchased from Firm 1 (explicitly sum its component)
        firm_1_quantity = sum(quantities[i][1] for i in quantities)

        # Total quantity purchased from Firm 2 is the residual (since only two firms)
        firm_2_quantity = total_quantity - firm_1_quantity

        # Convert firm quantities into market shares (avoid division by zero)
        firm_1_share = firm_1_quantity / total_quantity if total_quantity > 0 else 0.0
        firm_2_share = firm_2_quantity / total_quantity if total_quantity > 0 else 0.0

        return {1: firm_1_share, 2: firm_2_share}

    def summary(self):
        """
        Print an end-of-simulation summary.

        - If tipping was never detected, report "NOT detected."
        - If tipping was detected, print market shares at the tipping point
          and the cycle index where tipping occurred.

        Note:
        - tipping_point is stored as a cycle index in self.market_shares.
        """
        if self.tipping_point is None:
            print("Tipping point: NOT detected.")
            return

        print("Market Shares at Tipping Point:")
        print(f"Firm 1: {self.market_shares[self.tipping_point][1] * 100:.2f}%")
        print(f"Firm 2: {self.market_shares[self.tipping_point][2] * 100:.2f}%")
        print(f"Tipping point occurred at cycle {self.tipping_point}")
