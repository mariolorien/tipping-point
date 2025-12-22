import numpy as np


class Household:
    """
    Notation:
      i = household (this object)
      k = firm/product index (each firm supplies one basket)
      q[k] = quantity bought from firm k at time t, with sum_k q[k] = 1
    """

    def __init__(self, household_id, beta, eta, lam, rho=0.2, gamma=0.5, delta=0.5):
        self.household_id = household_id
        self.beta = beta  # Baseline preference for Firm 1 (Greener firm) - Pre Policy Tastes
        self.eta = eta # Baseline preference for Firm 2 (Less green firm) - Pre Policy Tastes
        self.lam = lam # Price sensitivity: how strongly households respond to price changes  

        self.utility_function = None

        # Habit formation
        self.rho = rho # How strong habit formation is; 0 = no habit, 1 = full weight on last choice - path dependence 
        self.gamma = gamma # Weight of habit in utility function

        # Peer influence
        self.delta = delta # Weight of peer influence in utility function
        self.peer_signal_history = []

        # Habit state 
        self.habit = None

        """
        Note for Reviewers: 
        “Households are heterogeneous utility-maximisers with intrinsic tastes, price sensitivity, habit formation, 
        and social influence. Habit formation and peer effects enter utility rather than mechanically constraining choice,
        ensuring that all behaviour remains economically interpretable.
        No single mechanism is sufficient to generate tipping; instead, non-linear transitions arise from the interaction of weak 
        tastes, modest social influence, and temporal persistence.”
        """

    def as_dict(self):

        """
        Note for Reviewers: 
        Converts a series of internal states of the Household into a dictionary for easy logging or analysis.
        Can be extended to add more attributes if needed for inspection or otherwise.  
        """
        return {"beta": self.beta, "eta": self.eta, "lam": self.lam}


    # ---------------- Utility components ----------------

    def baseline_utility(self, q, alpha):
        eps = 1e-12
        total = 0.0
        for k in q:
            total += alpha[k] * np.log(q[k] + eps)
        return total

        """
        Note for Reviewers: 
        First part of the Cobb-Douglas style Utility Function. 
        eps is a small constant to avoid log(0) when consumption is 0 - very unrealistic - but needed for numerical stability.
        q is a dictionary of quantities purchased from each firm k.
        alpha is a dictionary of baseline preference weights for each firm k.

        """

    def green_taste_utility(self, q, g):
        total = 0.0
        for k in q:
            total += g[k] * q[k]
        return self.beta * total

        """
        Note for Reviewers:
        Positive element of the Utility Function that captures intrinsic taste for green products.
        q is a dictionary of quantities purchased from each firm k.
        g is a dictionary of greenness levels for each firm k.

        Calculation Example: 

        q = {"A": 0.70, "B": 0.30} Household always buys 0.70 from Firm A, and 0.30 from Firm B
        g = {"A": 0.80, "B": 0.20} Firm A's product is 80% green, Firm B's product is 20% green
        beta = 2.0  Household intrinsic Green Taste 

        0.80 * 0.70 + 0.20 * 0.30 = 0.62
        0.62 * 2.0 = 1.24 Utility contribution from green taste

        1.24 is added to the total utility from this term.

        """

    def price_disutility(self, q, p):
        total = 0.0
        for k in q:
            total += p[k] * q[k]
        return self.lam * total

        """
        Note for Reviewers:
        Negative element of the Utility Function that captures disutility from spending on consumption.
        q is a dictionary of quantities purchased from each firm k.
        p is a dictionary of prices charged by each firm k.
        lam (self.lam) is the households price-sensitivity parameter (typically negative), so higher spending lowers utility.

        Calculation Example:

        q   = {"A": 0.70, "B": 0.30}  Household buys 0.70 from Firm A and 0.30 from Firm B
        p   = {"A": 1.50, "B": 1.00}  Firm A price is 1.50 per unit; Firm B price is 1.00 per unit
        lam = -0.20                   Household price sensitivity (negative ⇒ spending reduces utility)

        Step 1: compute total expenditure (price * quantity, summed across firms):
        1.50 * 0.70 + 1.00 * 0.30 = 1.05 + 0.30 = 1.35   (total spending)

        Step 2: convert spending into utility (disutility) using lam:
        1.35 * (-0.20) = -0.27   (utility contribution from price/disutility term)

        -0.27 is added to the total utility from this term (i.e., it lowers total utility).
        """

    def emissions_disutility(self, q, e):
        total = 0.0
        for k in q:
            total += e[k] * q[k]
        return self.eta * total

        """
        Note for Reviewers: 
        Negative element of the Utility Function that captures disutility from emissions embodied in consumption.
        q is a dictionary of quantities purchased from each firm k.
        e is a dictionary of emissions intensity levels for each firm k (e.g., kgCO2 per unit, or an emissions index per unit).
        eta (self.eta) is the household s emissions sensitivity parameter (typically negative), so higher embodied emissions lowers utility.

        Calculation Example:

        q   = {"A": 0.70, "B": 0.30}   Household buys 0.70 from Firm A and 0.30 from Firm B
        e   = {"A": 0.40, "B": 1.20}   Firm A emits 0.40 per unit; Firm B emits 1.20 per unit
        eta = -0.50                    Household emissions sensitivity (negative ⇒ emissions reduce utility)

        Step 1: compute total embodied emissions (emissions * quantity, summed across firms):
        0.40 * 0.70 + 1.20 * 0.30 = 0.28 + 0.36 = 0.64   (total embodied emissions)

        Step 2: convert emissions into utility (disutility) using eta:
        0.64 * (-0.50) = -0.32   (utility contribution from emissions/disutility term)

        -0.32 is added to the total utility from this term (i.e., it lowers total utility).

        """

    def habit_utility(self, q):
        if self.habit is None:
            return 0.0

        total = 0.0
        for k in q:
            total += self.habit.get(k, 0.0) * q[k]
        return self.gamma * total

        """
        Note for Reviewers:
        Positive element of the Utility Function that captures habit formation (preference for consuming what was consumed before).
        self.habit is a dictionary storing the households habit “strength” for each firm k (e.g., from past choices).
        q is a dictionary of current quantities purchased from each firm k.
        gamma (self.gamma) scales the importance of habit in utility.

        If self.habit is None, the household has no habit state yet (e.g., at initialisation), so this term contributes 0.

        Calculation Example:

        habit = {"A": 0.60, "B": 0.20}  Household has stronger habit toward Firm A than Firm B
        q     = {"A": 0.70, "B": 0.30}  Household currently buys 0.70 from A and 0.30 from B
        gamma = 0.50                    Habit salience in utility

        Step 1: compute habit-weighted consumption (habit * quantity, summed across firms):
        0.60 * 0.70 + 0.20 * 0.30 = 0.42 + 0.06 = 0.48

        Step 2: scale by gamma to obtain the utility contribution:
        0.48 * 0.50 = 0.24   (utility contribution from habit term)

        0.24 is added to the total utility from this term.

        """

    def peer_utility(self, q, peer_signal, g=None):
        if peer_signal is None:
            peer_signal = 0.0

        if g is None:
            return self.delta * float(peer_signal)

        total = 0.0
        for k in q:
            total += g[k] * q[k]
        return self.delta * float(peer_signal) * total

        """
        Note for Reviewers:    
        Positive element of the Utility Function that captures peer (social) influence.

        Inputs:
        - q: dictionary of quantities purchased from each firm k.
        - peer_signal: a scalar capturing the strength of the social signal (e.g., “how pro-green my peers are”,
          or the observed share of green consumption in the population / neighbourhood / reference group).
        - g (optional): dictionary of greenness levels for each firm k.

        Parameter:
        - delta (self.delta): strength of peer influence (how much the household cares about the social signal).

        Behaviour (two cases):

        CASE 1 (g is None):
        - Peer influence enters utility as a simple additive term that depends only on peer_signal.
        - Utility contribution = delta * peer_signal
        - This is a “pure norm/signal” effect that does not depend on what the household buys in this step.

        Calculation Example (Case 1):
        peer_signal = 0.60   (e.g., peers are moderately pro-green / green norm is moderate)
        delta       = 0.50   (household moderately sensitive to peers)

        Utility = 0.50 * 0.60 = 0.30
        0.30 is added to total utility from this peer term.

        CASE 2 (g is provided):
        - Peer influence is amplified by the households own greenness-weighted consumption:
          total = sum_k (g[k] * q[k])
          Utility contribution = delta * peer_signal * total
        - Interpretation: the social signal matters more when the households bundle is greener;
          i.e., “social approval” (or social alignment) is stronger when one actually consumes green content.

        Calculation Example (Case 2):
        q           = {"A": 0.70, "B": 0.30}   Household buys 0.70 from A and 0.30 from B
        g           = {"A": 0.80, "B": 0.20}   Firm A is 80% green; Firm B is 20% green
        peer_signal = 0.60                    Social signal strength
        delta       = 0.50                    Peer influence strength

        Step 1: compute greenness-weighted consumption:
        total = 0.80 * 0.70 + 0.20 * 0.30 = 0.56 + 0.06 = 0.62

        Step 2: multiply by peer_signal and delta:
        Utility = 0.50 * 0.60 * 0.62
                = 0.30 * 0.62
                = 0.186

        0.186 is added to total utility from this peer term.

        Notes:
        - If peer_signal is None, it is treated as 0.0, so the peer term contributes nothing.
        - delta governs the overall magnitude of peer influence; higher delta makes social effects stronger.

        """


    # ---------------- Habit update ----------------

    def init_habit(self, firm_ids, initial_value=0.5):
        self.habit = {}
        for k in firm_ids:
            self.habit[k] = float(initial_value)

        """
        Note for Reviewers: 
        Dictionary. Household start with 50% preference for each firm.
        self.habit == {
        "A": 0.5,
        "B": 0.5
        }
        """

    def update_habit(self, q_prev):
        if self.habit is None:
            self.init_habit(list(q_prev.keys()), initial_value=1.0 / max(1, len(q_prev)))

        for k in q_prev:
            self.habit[k] = (1.0 - self.rho) * self.habit.get(k, 0.0) + self.rho * q_prev[k]
        """
        Note for Reviewers: 
        Example:

        q_prev = {"A": 0.7, "B": 0.3}
        rho = 0.2
        Initial habit = {"A": 0.5, "B": 0.5}    
        Updated habit:
        Firm A : (1 - 0.2) * 0.5 + 0.2 * 0.7 = 0.4 + 0.14 = 0.54
        Firm B : (1 - 0.2) * 0.5 + 0.2 * 0.3 = 0.4 + 0.06 = 0.46
        New habit = {"A": 0.54, "B": 0.46}
        """
    # ---------------- Total utility ----------------

    def update_utility_function(self, q, p, g, e, alpha, peer_signal=None):
        base = self.baseline_utility(q, alpha)
        green = self.green_taste_utility(q, g)
        price_penalty = self.price_disutility(q, p)
        emissions_penalty = self.emissions_disutility(q, e)

        habit_term = self.habit_utility(q)
        peer_term = self.peer_utility(q, peer_signal, g=g)

        U = base + green + habit_term + peer_term - price_penalty - emissions_penalty

        self.utility_function = U
        return U
        """
        Note for Reviewers: 
        We count greenes twice: 
        (1) green and (2) peer_term. This is intended: 
        in (1) households like greenness intrinsically (beta) and they get extra utility from greenness
        when, in (2) peers signal its socially valued (delta * peer_signal)

        (i) Baseline utility:
            A standard log-linear (Cobb Douglas type) utility over consumption shares,
            capturing diminishing marginal utility and ensuring interior solutions.

        (ii) Intrinsic green taste:
            The term `green = beta * sum_k(g_k q_k)` captures households intrinsic
            pro-environmental preferences. This reflects private moral satisfaction
            from consuming greener products, independent of prices or social context.

        (iii) Peer (social) reinforcement:
            The term `peer_term = delta * peer_signal * sum_k(g_k q_k)` captures social
            reinforcement of green behaviour. Greenness therefore enters utility a
            second time, conditional on the contemporaneous peer signal.
            This “double counting” is intentional: intrinsic moral preferences and
            social approval are treated as distinct channels.

        (iv) Habit formation:
            The habit term captures within-household persistence and path dependence,
            reflecting a preference for repeating past consumption patterns.

        (v) Price and emissions penalties:
            Price and emissions enter utility as penalties proportional to total
            expenditure and embodied emissions, respectively. The penalty weights
            (lambda, eta) are positive and these terms are subtracted from total utility,
            so higher prices or emissions reduce utility.

        Overall, households remain fully rational utility maximisers; behavioural and
        social mechanisms operate exclusively through preference shifters rather than
        through ad hoc choice rules.
        """

    # ---------------- Grid choice + peer logging ----------------

    def record_peer_signal(self, peer_signal):
        self.peer_signal_history.append(float(peer_signal))

    def choose_quantities_grid(self, p, g, e, alpha, peer_signal=0.0, grid_steps=101):
        """
        Grid-search over splits between two firms.
        Returns q = {firm_id_1: q1, firm_id_2: q2}

        Inputs are dicts keyed by firm_id: p, g, e, alpha.
        """
        # record peer signal ONCE per cycle 
        self.record_peer_signal(peer_signal)

        firm_ids = sorted(list(p.keys()))    # SORTING  firms' IDs
        if len(firm_ids) != 2:
            raise ValueError("Grid search currently expects exactly 2 firms.") 
        """
        The decision process is simplified to two firms for tractability.
        Rather than modelling a full general demand system, households allocate a unit mass of consumption between
        two competing firms. Households choose consumption shares by discretely approximating a 
        continuous two-option allocation problem.
        """

        k1, k2 = firm_ids[0], firm_ids[1]  # Store keys on k1 and K2 ( k1 == "A" k2 == "B") 

        eps = 1e-6 
        """
        We add eps to avoid log(0) in the utility function when consumption quantity is zero.
        Explanation: 
        For a positive number that is very close to zero, the result of ln(eps_ is simply a finite 
        but extremely large negative floating-point number, not truly infinite; hence the need to add eps
        to the calculation. 

        """
        best_q = None # Store the best quantity allocation found
        best_U = -1e30 # Initialize best utility to a very low number, easy to beat

        """
        For the rest of the method: 

        grid_steps = 101 → 101 candidate splits (≈ 1% resolution)
        s / (grid_steps - 1) → evenly spaced points from 0 to 1
        (1 - 2*eps) → shrink interval to avoid boundaries
        + eps → shift away from zero
        q2 = 1 - q1 → enforce unit-sum bundle 
        """

        """
        For each s in our grid of 101 steps, we calculate the quantities, as follow: 
        1	0.01	0.000001 + 0.999998 * 0.01	0.010001	0.989999
        2	0.02	0.000001 + 0.999998 * 0.02	0.020001	0.979999
        3	0.03	0.000001 + 0.999998 * 0.03	0.030001	0.969999
        4	0.04	0.000001 + 0.999998 * 0.04	0.040001	0.959999    
        ....
        ....
        ....
        101	1.00	0.000001 + 0.999998 * 1.00	1.000000	0.000000

        At each "pass" in our loop, the q1 and q2 are used to calculate the 
        household utility function and stored in U.
        Simultaneously, at each pass we store the quantity with the firm in q 
        q = {k1: q1, k2: q2}
        If the utility function is better than the best found so far, we update best_U and best_q. 
        """
        for s in range(grid_steps):
            q1 = eps + (1.0 - 2.0 * eps) * (s / (grid_steps - 1))
            q2 = 1.0 - q1

            q = {k1: q1, k2: q2} # candidate quantity allocation

            U = self.update_utility_function(q, p, g, e, alpha, peer_signal=peer_signal)

            if U > best_U:
                best_U = U
                best_q = q

        return best_q
