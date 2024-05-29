from MWA import MWA
import numpy as np
import math

class MWBA(MWA):
    def __init__(self, custom_epsilon=None, *args, **kwargs):
        """Constructor for MWBA class. Inherits from MWA. Addition of epsilon heuristic, an estimate for
            the exploration parameter epsilon provided by chatgpt.
        """
        super().__init__(*args, **kwargs)
        # rwthsa to chatgpt kai m eipe pws ena kalo guess gia to epsilon einai auto
        # petaksa min kai max mesa gia na ta kanw bound sto [0,1]
        # epsilon closer to 0 q pio konta sto p I guess
        # megalytero epsilon perissotero random(?) mallon oxi
        # megalytero epsilon -> ligotero dependence sto p. Ola ta alla einai koina (k, epsilon etc) 
        # opote ontws pio (pseudo)random
        # ara eksou kai giati to leme exploration sta slides. I'm just retarded I guess
        self.custom_epsilon = custom_epsilon
        self.heuristic_epsilon = max(0, min(1, math.sqrt(self.K * math.log(self.K)/self.T)))

    def algo_execution(self):
        #epsilon = 0.2
        epsilon =  self.custom_epsilon if self.custom_epsilon else self.heuristic_epsilon
        ita = self.custom_ita if self.custom_ita else self.optimal_ita
        for t in range(self.T):
            normalized_weights = self.weights / self.weights.sum()
            q_array = [((1-epsilon) * p) + (epsilon/self.K) for p in normalized_weights]
            q_array = np.array(q_array)
            normalized_q_array = q_array / q_array.sum() # might be norm already, not taking any chances
            
            chosen_stock = np.random.choice(self.K, p=normalized_q_array)
            transaction_fee = self.transaction_fees[chosen_stock]
            
            daily_returns = self.data[t]
            actual_returns = [daily_change - transaction_fee for daily_change in daily_returns]
            
            self.cumulative_profits[t] = self.cumulative_profits[t-1] + actual_returns[chosen_stock]
            best_hindsight_profit = max(actual_returns)
            self.cumulative_regret[t] = self.cumulative_regret[t-1] + (best_hindsight_profit - actual_returns[chosen_stock])
            
            losses = [0 for _ in range(self.K)]
            losses[chosen_stock] = self.calculate_loss_for_expert(self.data[t][chosen_stock], transaction_fee)/q_array[chosen_stock]
            
            for k in range(self.K):
                self.weights[k] *= pow((1 - ita), losses[k])
                
        return self.cumulative_profits, self.cumulative_regret
                
                