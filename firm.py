import numpy as np

class Firm:
    def __init__(self, firm_id, green_share,
                 base_cost_brown=1.0, base_cost_green=1.0,
                 overhead_cost=0.0, markup=0.10, rng=None):

        self.firm_id = firm_id
        self.green_share = green_share  # s in [0,1] # Proportion of green in the product - as opposed to brown producs- 

        # Technology / costs
        self.base_cost_brown = base_cost_brown 
        self.base_cost_green = base_cost_green
        self.overhead_cost = overhead_cost  # lump sum each period

        # Pricing
        self.markup = markup 
        self.price = None
        self.last_elasticity = None 

        # History (needed for elasticity from last two periods)
        self.price_history = []
        self.quantity_history = []
        self.profit_history = []

        # RNG (passed from Economy)
        self.rng = rng  # random number generator for reproducibitlity 
        """
        We do this: 
        rng = np.random.default_rng(42)
        x = rng.uniform()

        Instead of this: 
        np.random.seed(42)
        x = np.random.uniform()
        """

    # ---------------- Costs, emissions, greenness ----------------

    def unit_cost(self, tau, sigma):
        """
        Blended unit cost:
          - brown part pays carbon tax - tau
          - green part gets subsidy - sigma
        """
        s = self.green_share

        cost_green = self.base_cost_green - sigma  # green part gets subsidy
        cost_brown = self.base_cost_brown + tau # brown part pays carbon tax


        #  safeguard to avoid negative unit costs)
        if cost_green < 0.0:
            cost_green = 0.0

        """
        # Blended unit cost calculation example for Firm 1 - Firm 2 will be similar - 
        #
        # +----------------------+-------------------------------+--------+
        # | Component            | Formula                       | Value  |
        # +----------------------+-------------------------------+--------+
        # | Green share          | s                             | 0.70   |
        # | Brown share          | 1 - s                         | 0.30   |
        # | Green unit cost      | base_cost_green - sigma       | 1.15   |
        # | Brown unit cost      | base_cost_brown + tau         | 1.10   |
        # | Green contribution   | s * cost_green                | 0.805  |
        # | Brown contribution   | (1 - s) * cost_brown          | 0.330  |
        # +----------------------+-------------------------------+--------+
        # | Blended unit cost    | sum of contributions          | 1.135  |
        # +----------------------+-------------------------------+--------+
        """

        return s * cost_green + (1.0 - s) * cost_brown 

    def greenness(self):
        """Return the firm's greenness share (s ∈ [0,1])."""
        return self.green_share

    def emissions_intensity(self, e_green=0.2, e_brown=1.0):
        """
        Blended emissions intensity per unit.

        Example calculation:

        +----------------------+---------------------------+--------+
        | Component            | Formula                   | Value  |
        +----------------------+---------------------------+--------+
        | Green share          | s                         | 0.70   |
        | Brown share          | 1 - s                     | 0.30   |
        | Green emissions      | e_green                   | 0.20   |
        | Brown emissions      | e_brown                   | 1.00   |
        | Green contribution   | s * e_green               | 0.140  |
        | Brown contribution   | (1 - s) * e_brown         | 0.300  |
        +----------------------+---------------------------+--------+
        | Emissions intensity  | sum of contributions      | 0.440  |
        +----------------------+---------------------------+--------+
 
        The firm's per-unit emissions are a convex combination of green and brown
        production intensities. Increasing the green share monotonically lowers
        emissions, bounded between e_green and e_brown.
        """
        s = self.green_share
        return s * e_green + (1.0 - s) * e_brown

    # ---------------- Pricing ----------------

    def set_initial_price(self, tau, sigma):
        """
        Set price at t=0 using p = c*(1+markup)

         Example:
         green_share s        = 0.70
         base_cost_green      = 1.20
         base_cost_brown      = 1.00
         tau (carbon tax)     = 0.10
         sigma (subsidy)      = 0.05
         markup               = 0.25

         Step 1 (unit cost):
         cost_green = 1.20 - 0.05 = 1.15
         cost_brown = 1.00 + 0.10 = 1.10
          c = 0.70*1.15 + 0.30*1.10 = 1.135

         Step 2 (price):
         p0 = c*(1+markup) = 1.135*(1.25) = 1.41875

        """
        c = self.unit_cost(tau, sigma)
        self.price = c * (1.0 + self.markup)

        self.price_history = [self.price]
        self.quantity_history = []
        self.profit_history = []

        return self.price

    def estimate_elasticity_last_two(self):
        """
        Estimate price elasticity of demand using ARC elasticity
        based on the last two observations of price and quantity.

        Arc elasticity measures responsiveness using percentage changes
        computed around the midpoint (average) of price and quantity,
        rather than a single base point. This makes it symmetric and
        well-suited to discrete, period-by-period adjustments.

        Formula:
           elasticity =
                ( (q2 - q1) / ((q1 + q2)/2) )
                --------------------------------
                ( (p2 - p1) / ((p1 + p2)/2) )

        Example:
         Period t-1:
            p1 = 1.40
            q1 = 0.60

        Period t:
            p2 = 1.50
            q2 = 0.54

        Step 1: differences
            dq = q2 - q1 = -0.06
            dp = p2 - p1 =  0.10

        Step 2: averages
            q_avg = (0.60 + 0.54) / 2 = 0.57
            p_avg = (1.40 + 1.50) / 2 = 1.45

        Step 3: percentage changes
            dq / q_avg = -0.06 / 0.57 ≈ -0.105
            dp / p_avg =  0.10 / 1.45 ≈  0.069

        Step 4: elasticity
            elasticity ≈ -0.105 / 0.069 ≈ -1.52


            A 1% increase in price is associated with an
            approximate 1.5% decrease in quantity demanded.
        """

        if len(self.price_history) < 2 or len(self.quantity_history) < 2:
            return None

        p1, p2 = self.price_history[-2], self.price_history[-1] # Gets last two -recorded- prices
        q1, q2 = self.quantity_history[-2], self.quantity_history[-1] # Gets last two -recorded- quantities

        if p1 <= 0 or p2 <= 0:
            return None   # avoid division by zero or negative prices. Elasticity strictly positive. 

        dp = p2 - p1   # need to know the price difference to calculate the change in price 
        dq = q2 - q1  # need to know the quantity difference to calculate the change in quantity

        p_avg = 0.5 * (p1 + p2) 
        q_avg = 0.5 * (q1 + q2)

        """
        Instead of asking: “% change relative to where?” (p1? p2?)
        we use Arc elasticity: we use the midpoint to avoid asymmetry.
        """
        if abs(dp) < 1e-12 or abs(q_avg) < 1e-12:
            return None  # avoid division by -almost- zero


        elasticity = (dq / q_avg) / (dp / p_avg) # actual arc elasticity formula 
        return elasticity



    def price_update(self, tau, sigma, step=0.05, elasticity_floor=0.2):
        """
        Updates markup (and thus price) using a Lerner-style target.

        In a simple monopoly benchmark with constant elasticity demand, optimal pricing
        satisfies the Lerner condition:

        L = (p - c) / p = 1 / |epsilon|

        where:
           - p is price
           - c is marginal/unit cost
           - epsilon is the price elasticity of demand (negative for downward-sloping demand)
           - |epsilon| is used because L is positive

        Interpretation:
            - If demand is more elastic (|epsilon| large), consumers are sensitive to price,
              so the firm sets a smaller markup (L is small).
            - If demand is less elastic (|epsilon| small), consumers are insensitive to price,
              so the firm can sustain a larger markup (L is large).

         IMPLEMENTATION IN THIS MODEL
         ----------------------------
         1) Compute current unit cost c under policy (tau, sigma).
         2) Estimate arc elasticity epsilon using the last two observed (p, q) pairs.
           If unavailable, keep markup unchanged.

         3) Compute the Lerner target:  L_target = 1 / |epsilon|.
            - Apply an elasticity floor to avoid extreme outcomes when |epsilon| is near zero.
            - Cap L_target below 1 to avoid infinite markups.
         4) Convert Lerner index L_target into the markup over cost:
              markup = (p - c)/c = L / (1 - L)
         5) Adjust markup gradually toward the target using a smoothing step in [0,1]:
              markup_new = (1-step)*markup_old + step*markup_target
     
         This prevents violent oscillations in price from noisy two-point elasticity estimates.

         NUMERICAL EXAMPLE (matches the code)
         ------------------------------------
         Suppose at time t we have:

           unit cost:            c = 1.135
           current markup:       markup_old = 0.25     (25% cost-plus)
           step:                 step = 0.05          (5% adjustment per period)

         Elasticity estimate from last two periods (arc elasticity):
         epsilon = -1.52   ->  |epsilon| = 1.52

         1) Lerner target:
            L_target = 1/|epsilon| = 1/1.52 = 0.6579

           (apply floor/cap if needed; here it is unchanged)

         2) Convert Lerner target to markup target:
            markup_target = L_target / (1 - L_target)
                    = 0.6579 / (1 - 0.6579)
                    = 0.6579 / 0.3421
                    = 1.923

          Interpretation: if demand is this inelastic, the Lerner benchmark implies
          a very high markup. (This is why the model includes floors/caps and gradual adjustment.)

         3) Gradual adjustment:
         markup_new = (1-step)*markup_old + step*markup_target
                 = 0.95*0.25 + 0.05*1.923
                 = 0.2375 + 0.09615
                 = 0.33365

         4) New price:
         p_new = c*(1 + markup_new)
            = 1.135*(1.33365)
            ≈ 1.513

         So the price increases from cost-plus 25% toward the Lerner-implied markup, but only
         partially because step is small.

         NOTES / SAFEGUARDS
         ------------------
         - elasticity_floor prevents unrealistically huge markups when |epsilon| is near 0.
         - the Lerner cap (0.95) prevents division by (1 - L) near zero, which would generate
          near-infinite markups.
         - using only the last two periods makes epsilon noisy; smoothing (step < 1) is therefore
          essential for stability in an ABM setting.

        """
        c = self.unit_cost(tau, sigma)
        """
         In line with our simulation, tau is the carbon tax, and sigma 
          -the subsidy for green products- are included on the unit cost.
       
         The price formation loop is as follow: 
       
           POLICY (tau, sigma)
             ↓
          Firm unit cost  c(t)
             ↓
          Firm price      p(t)
             ↓
          Household demand q(t)
             ↓
          Elasticity ε(t)
             ↓
          Firm markup update
             ↺
        """

        eps = self.estimate_elasticity_last_two()
        if eps is None:
            # not enough info yet; keep markup unchanged
            self.price = c * (1.0 + self.markup)
            self.price_history.append(self.price)
            return self.price

        eps_abs = abs(eps)
        if eps_abs < elasticity_floor:
            eps_abs = elasticity_floor

        """
         Note on elasticity_floor = 0.2

         Setting elasticity_floor = 0.2 enforces a lower bound on the absolute value of the
         estimated price elasticity used in the Lerner benchmark:

         |epsilon| >= 0.2

         Since the Lerner target is computed as:

         L_target = 1 / |epsilon|

         this implies an upper bound:

         L_target <= 1 / 0.2 = 5

         In practice, we also cap the Lerner target at 0.95 (to avoid infinite markups when
         converting the Lerner index into a cost-plus markup), so this cap dominates the
         extreme case.

         The main purpose of elasticity_floor is robustness: arc elasticity is estimated
         from only the last two (p, q) observations and can be noisy. Without a floor,
         |epsilon| could become arbitrarily close to zero, leading the model to treat demand
         as “almost perfectly inelastic” and generating implausibly large markups.

         To avoid extreme and unstable markups arising from noisy two-point elasticity estimates, 
         we impose a lower bound on the absolute elasticity used in the Lerner benchmark.

         """
        self.last_elasticity = eps_abs

        lerner_target = 1.0 / eps_abs

        # avoid infinite markups
        if lerner_target >= 0.95:
            lerner_target = 0.95

        """
         Convert Lerner target to markup:
         L = (p - c)/p  -> markup = (p - c)/c = L/(1 - L)

         Suppose:
             unit cost           c = 1.00
             Lerner index        L = 0.40

         Step 1: compute markup from Lerner index
             markup = L / (1 - L)
                    = 0.40 / (1 - 0.40)
                    = 0.40 / 0.60
                    = 0.6667

         Step 2: compute price from markup
              p = c * (1 + markup)
                = 1.00 * (1 + 0.6667)
                = 1.6667

         Check (recover the Lerner index):
             (p - c) / p = (1.6667 - 1.00) / 1.6667 = 0.40

             A Lerner index of 0.40 corresponds to a 66.7% cost-plus markup.
        """
       
        target_markup = lerner_target / (1.0 - lerner_target)

        # Gradual adjustment
        self.markup = (1.0 - step) * self.markup + step * target_markup

        self.price = c * (1.0 + self.markup)
        self.price_history.append(self.price)

        return self.price

    # ---------------- Profits ----------------

    def compute_profit(self, quantity, tau, sigma):
        """
        Profit = revenue - variable costs - overhead (lump sum).

    
        Numerical example for compute_profit(quantity, tau, sigma)

        Assume the firm has:
             green_share s        = 0.70
             base_cost_green      = 1.20
             base_cost_brown      = 1.00
             tau (carbon tax)     = 0.10
             sigma (subsidy)      = 0.05
             price (already set)  = 1.50
             overhead_cost        = 0.20   (lump-sum per period)
             quantity sold        = 0.60

            Step 1: compute policy-adjusted unit cost c
              cost_green = base_cost_green - sigma = 1.20 - 0.05 = 1.15
              cost_brown = base_cost_brown + tau   = 1.00 + 0.10 = 1.10

             c = s*cost_green + (1-s)*cost_brown
              = 0.70*1.15 + 0.30*1.10
              = 0.805 + 0.330
              = 1.135

            Step 2: compute revenue
             revenue = price * quantity
                     = 1.50 * 0.60
                     = 0.90

            Step 3: compute variable costs
              variable_cost = c * quantity
                    = 1.135 * 0.60
                    = 0.681

            Step 4: compute profit (subtract overhead)
             profit = revenue - variable_cost - overhead_cost
                    = 0.90 - 0.681 - 0.20
                    = 0.019

        
              The firm makes a small positive profit this period because the markup over
              unit cost is just enough (given quantity) to cover the fixed overhead.
        """
        c = self.unit_cost(tau, sigma)

        revenue = self.price * quantity
        variable_cost = c * quantity
        profit = revenue - variable_cost - self.overhead_cost

        self.profit_history.append(profit)
        return profit

    # ---------------- Adaptation beyond price ----------------

    def adapt_strategy(self, profit_threshold=0.0, delta_s=0.02):
        """
        Very simple non-price adaptation:
        - If last profit < threshold, experiment slightly by adjusting green_share.
        - Otherwise keep unchanged.

        Reproducible because we use self.rng (passed from Economy).


        Very simple non-price adaptation:
          - If last profit < threshold, experiment slightly by adjusting green_share.
          - Otherwise keep unchanged.

  
         This is a bounded, local-search (trial-and-error) rule on the firm's non-price
         attribute (green_share). When profits are weak, the firm explores a small
         change in product composition (greenness) rather than changing prices directly.
         This captures incremental experimentation under uncertainty rather than global
         optimisation.

         Numerical examples:

        CASE A: Profit meets threshold (no experimentation)
              profit_threshold = 0.00
              last_profit      = +0.05
              current s        = 0.70

            Since last_profit >= profit_threshold, keep strategy:
                 s_new = 0.70

         CASE B: Profit below threshold (experiment by ±delta_s)
             profit_threshold = 0.00
             last_profit      = -0.03
              delta_s          = 0.02
               current s        = 0.70

           Since last_profit < profit_threshold, the firm experiments:
            direction ∈ {-1, +1}

         Example B1 (direction = +1):
            s_new = 0.70 + 1*0.02 = 0.72

         Example B2 (direction = -1):
            s_new = 0.70 - 1*0.02 = 0.68

          Bounds:
            If current s = 0.99 and direction = +1:
                s_new = 0.99 + 0.02 = 1.01 -> clipped to 1.00

            If current s = 0.01 and direction = -1:
                s_new = 0.01 - 0.02 = -0.01 -> clipped to 0.00

       
            Reproducibility note:
            direction is sampled using self.rng.choice([-1, +1]).
            Provided self.rng is seeded once (e.g., in Economy), the sequence of
            experiments is reproducible across runs.
        """ 
        if len(self.profit_history) == 0:
            return self.green_share

        last_profit = self.profit_history[-1]

        if last_profit < profit_threshold:
            if self.rng is None:
                direction = 1.0
            else:
                direction = self.rng.choice([-1.0, 1.0])

            self.green_share += direction * delta_s

            # keep within [0,1]
            if self.green_share < 0.0:
                self.green_share = 0.0
            if self.green_share > 1.0:
                self.green_share = 1.0

        return self.green_share
