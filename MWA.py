import math
import numpy as np

class MWA():
    def __init__(self, data, fees = False, custom_ita = None) -> None:
        """Initializes the MWA algorithm with the necessary

        Args:
            data (_type_): data extracted from the stocks.csv file
            fees (bool): True if transaction fees are enabled, False otherwise
            custom_ita (_type_, optional): a custom value for ita. If none is provided the "optimal" value will be used.
                Exclusively used for testing and isn't required. Defaults to None.
        """
        self.data = data
        self.T = self.data.shape[0]
        self.K = self.data.shape[1]
        self.optimal_ita =  math.sqrt(math.log(self.K)/self.T) # learning rate vash tou poso einai to K kai to T opws eipe h thewria
        self.custom_ita = custom_ita
        self.transaction_fees = [(0.5)/100 * i for i in range(1, self.K+1)] if fees else [0 for _ in range(self.K)]
        self.weights = np.ones(self.K)
        self.MAX_GAIN = 0.2005097706032286
        self.MAX_LOSS = -0.8086677997123807 - 5/100
        
        # kathara gia to peirama, oriaka ntroph m pou ta vazw edw alla w/e
        # pws alliws tha ta ekana track omws? no idea
        self.cumulative_profits = np.zeros(self.T)
        self.cumulative_regret = np.zeros(self.T)
        
    def calculate_loss_for_expert(self, daily_change, transaction_fee=0) -> float:
        """_summary_

        Args:
            daily_change (float): The daily change experienced by a stock in decimal form (e.g. 0.01 for 1%)
            transaction_fee (int, optional): The transaction fee, if any. Defaults to 0.

        Returns:
            float: The loss (I_i^t) experienced by expert (stock) i at time t.
        """
        true_change = (1 * (daily_change)) - transaction_fee
        normalized_change = (true_change - self.MAX_LOSS) / (self.MAX_GAIN - self.MAX_LOSS)
        normalized_loss = 1 - normalized_change
        return normalized_loss
        
    def algo_execution(self) -> tuple[np.ndarray, np.ndarray]:
        """Executes the MWA algorithm for the given data and returns the cumulative profits and cumulative regret

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the cumulative profits and cumulative regret
        """
        
        ita = self.custom_ita if self.custom_ita else self.optimal_ita
        for t in range(self.T):
            normalized_weights = self.weights / self.weights.sum()
            
            chosen_stock = np.random.choice(self.K, p=normalized_weights)
            # auto mporei na mpei kai se comment mias kai to trans fee to chosen stock den mas noiazei (?)
            #transaction_fee = self.transaction_fees[chosen_stock]
            
            daily_returns = self.data[t]
            actual_returns = [daily_change - self.transaction_fees[i] for i, daily_change in enumerate(daily_returns)]
            
            self.cumulative_profits[t] = self.cumulative_profits[t-1] + actual_returns[chosen_stock]
            best_hindsight_profit = max(actual_returns)
            self.cumulative_regret[t] = self.cumulative_regret[t-1] + (best_hindsight_profit - actual_returns[chosen_stock])
            
            
            for k in range(self.K):
                transaction_fee = self.transaction_fees[k]
                self.weights[k] *= pow((1 - ita), self.calculate_loss_for_expert(self.data[t][k], transaction_fee))
                
        return self.cumulative_profits, self.cumulative_regret
    
    