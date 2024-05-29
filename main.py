import numpy as np
import matplotlib.pyplot as plt
from MWA import MWA
from MWBA import MWBA

data = np.loadtxt('stocks.csv', delimiter=',')


def main():
    mwa_no_fees = MWA(data = data, fees = False)
    mwa_fees = MWA(data =data, fees = True)
    mwba = MWBA(data = data, fees = True)
    #print(f"Max loss: {mwa_fees.calculate_loss_for_expert(mwa_fees.MAX_LOSS)}\nMax gain: {mwa_fees.calculate_loss_for_expert(mwa_fees.MAX_GAIN)}")
    
    cum_prof_1, cum_reg_1 = mwa_no_fees.algo_execution()
    cum_prof_2, cum_reg_2 = mwa_fees.algo_execution()
    cum_prof_3, cum_reg_3 = mwba.algo_execution()
    
    # task 1 plots
    plt.figure()
    plt.plot(cum_reg_1)
    plt.title('Cumulative Regret')
    plt.xlabel('Days')
    plt.ylabel('Regret')
    plt.figure()
    plt.plot(cum_prof_1)
    plt.title('Cumulative Profits')
    plt.xlabel('Days')
    plt.ylabel('Profits')

    plt.show()

    # task 2 plots
    plt.figure()
    plt.plot(cum_prof_1, label='No transaction fee')
    plt.plot(cum_prof_2, label='With transaction fee')
    plt.title('Cumulative Profits')
    plt.xlabel('Days')
    plt.ylabel('Profits')
    plt.legend()

    plt.figure()
    plt.plot(cum_reg_1, label='No transaction fee')
    plt.plot(cum_reg_2, label='With transaction fee')
    plt.title('Cumulative Regret')
    plt.xlabel('Days')
    plt.ylabel('Regret')
    plt.legend()

    plt.show()

    # task 3 plots
    plt.figure()
    plt.plot(cum_prof_2, label='Experts')
    plt.plot(cum_prof_3, label='Bandits')
    plt.title('Cumulative Profits')
    plt.xlabel('Days')
    plt.ylabel('Profits')
    plt.legend()

    plt.figure()
    plt.plot(cum_reg_2, label='Experts')
    plt.plot(cum_reg_3, label='Bandits')
    plt.title('Cumulative Regret')
    plt.xlabel('Days')
    plt.ylabel('Regret')
    plt.legend()

    plt.show()
    
def experimental(fees = False):
    mwa_1 = MWA(data, fees)
    mwa_2 = MWA(data, fees, 0.2)
    mwa_3 = MWA(data, fees, 0.3)
    mwa_4 = MWA(data, fees, 0.4)
    # highest ita possible
    mwa_5 = MWA(data, fees, 0.5) 
    
    cum_prof_1, cum_reg_1 = mwa_1.algo_execution()
    cum_prof_2, cum_reg_2 = mwa_2.algo_execution()
    cum_prof_3, cum_reg_3 = mwa_3.algo_execution()
    cum_prof_4, cum_reg_4 = mwa_4.algo_execution()
    cum_prof_5, cum_reg_5 = mwa_5.algo_execution()
    print(f"Weights = {mwa_2.weights}")
    print(f"Probabilites = {mwa_2.weights/mwa_2.weights.sum()}")
    
    plt.figure()
    plt.plot(cum_prof_1, label='ita = optimal')
    plt.plot(cum_prof_2, label='ita = 0.2')
    plt.plot(cum_prof_3, label='ita = 0.3')
    plt.plot(cum_prof_4, label='ita = 0.4')
    plt.plot(cum_prof_5, label='ita = 0.5')
    plt.title('Cumulative Profits')
    plt.xlabel('Days')
    plt.ylabel('Profits')
    plt.legend()
    
    plt.figure()
    plt.plot(cum_reg_1, label='ita = optimal')
    plt.plot(cum_reg_2, label='ita = 0.2')
    plt.plot(cum_reg_3, label='ita = 0.3')
    plt.plot(cum_reg_4, label='ita = 0.4')
    plt.plot(cum_reg_5, label='ita = 0.5')
    plt.title('Cumulative Regret')
    plt.xlabel('Days')
    plt.ylabel('Regret')
    plt.legend()
    
    plt.show()
    
def ita_var_averages(n_runs = 10, fees = False):
    itas = ["Optimal", 0.2, 0.3, 0.4, 0.5, 0.6]
    
    avg_profits_all = []
    avg_regrets_all = []
    
    for ita in itas:
        # making it pretty
        ita_val = ita if ita != "Optimal" else None
        cum_profits = []
        cum_regrets = []
        for _ in range(n_runs):
            mwa = MWA(data, fees, ita_val)
            cum_prof, cum_reg = mwa.algo_execution()
            cum_profits.append(cum_prof)
            cum_regrets.append(cum_reg)

        cum_profits = np.array(cum_profits)
        cum_regrets = np.array(cum_regrets)
    
        avg_profits = cum_profits.mean(axis=0)
        avg_regrets = cum_regrets.mean(axis=0)
    
        avg_profits_all.append(avg_profits)
        avg_regrets_all.append(avg_regrets)
    
    plt.figure()
    for i, ita in enumerate(itas):
        plt.plot(avg_profits_all[i], label=f'ita={ita}')
    plt.title(f'Average Cumulative Profits with fees={fees}')
    plt.xlabel('Days')
    plt.ylabel('Profits')
    plt.legend()
    
    plt.figure()
    for i, ita in enumerate(itas):
        plt.plot(avg_regrets_all[i], label=f'ita={ita}')
    plt.title(f"Average Cumulative Regret with fees={fees}")
    plt.xlabel('Days')
    plt.ylabel('Regret')
    plt.legend()
    
    
    plt.show()
    
def epsilon_var_averages(n_runs = 10, fees = False):
    epsilons = ["Heuristic", 0.1, 0.2, 0.3, 0.4, 0.5, 0.999]
    
    avg_profits_all = []
    avg_regrets_all = []
    
    for epsilon in epsilons:
        epsilon_val = epsilon if epsilon != "Heuristic" else None
        cum_profits = []
        cum_regrets = []
        for _ in range(n_runs):
            mwba = MWBA(data=data, fees=fees, custom_epsilon=epsilon_val)
            cum_prof, cum_reg = mwba.algo_execution()
            cum_profits.append(cum_prof)
            cum_regrets.append(cum_reg)

        cum_profits = np.array(cum_profits)
        cum_regrets = np.array(cum_regrets)
    
        avg_profits = cum_profits.mean(axis=0)
        avg_regrets = cum_regrets.mean(axis=0)
    
        avg_profits_all.append(avg_profits)
        avg_regrets_all.append(avg_regrets)
    
    plt.figure()
    for i, epsilon in enumerate(epsilons):
        plt.plot(avg_profits_all[i], label=f'epsilon={epsilon}')
    plt.title(f'Average Cumulative Profits with fees={fees}')
    plt.xlabel('Days')
    plt.ylabel('Profits')
    plt.legend()
    
    plt.figure()
    for i, epsilon in enumerate(epsilons):
        plt.plot(avg_regrets_all[i], label=f'epsilon={epsilon}')
    plt.title(f"Average Cumulative Regret with fees={fees}")
    plt.xlabel('Days')
    plt.ylabel('Regret')
    plt.legend()
    
    
    plt.show()
    
 
if __name__ == '__main__':
    main()
    #experimental(False)
    #experimental(True)
    #ita_var_averages(fees=False)
    #ita_var_averages(fees=True)
    #epsilon_var_averages(fees=True)