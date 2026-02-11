import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd


### Market order data from LOBSTER ###
# def get_market_order_sizes(file_path):
#     """
#     Extracts aggregated market order sizes from a LOBSTER message file.
#     """
#     # Define column names based on LOBSTER documentation 
#     # 1. Time, 2. Type, 3. Order ID, 4. Size, 5. Price, 6. Direction
#     names = ['Time', 'Type', 'OrderID', 'Size', 'Price', 'Direction']
    
#     # Load the message file
#     df = pd.read_csv(file_path, header=None, names=names)
    
#     # Filter for executions:
#     # Type 4: Execution of a visible limit order 
#     # Type 5: Execution of a hidden limit order 
#     executions = df[df['Type'].isin([4, 5])].copy()
    
#     # Aggregate "walking the book" events:
#     # Group by Time and Direction and sum the Size to get the total 
#     # volume of the original aggressive market order.
#     market_orders = executions.groupby(['Time', 'Direction'])['Size'].sum()
    
#     # Return as a numpy array for statistical fitting
#     return market_orders.values

# # File path for the provided data
# amazon = 'AMZN_2012-06-21_34200000_57600000_message_1.csv'
# apple = 'AAPL_2012-06-21_34200000_57600000_message_1.csv'
# google = 'GOOG_2012-06-21_34200000_57600000_message_1.csv'
# intel = 'INTC_2012-06-21_34200000_57600000_message_1.csv'
# microsoft = 'MSFT_2012-06-21_34200000_57600000_message_1.csv'

# # Extract the sizes
# companies = [amazon, apple, google, intel, microsoft]

# frames = [get_market_order_sizes(company) for company in companies]

# mo_sizes = []
# for frame in frames:
#     mo_sizes = np.append(mo_sizes, frame)

mo_sizes = np.random.pareto(1.274, 10000) + 1


plt.hist(mo_sizes, bins = 100, edgecolor = "darkblue")
plt.xlabel("Market Order Size")
plt.ylabel("Frequency")
plt.title("Histogram of Market Order Sizes")
plt.savefig("Histogram")
plt.clf()

### 

### FIND CUTOFF ###

#Lav en log log plot af CCDF (complementary CDF) som skal være en lige linje for at den følger power law

sorted_sizes = np.sort(mo_sizes)[::-1]  # Sort descending
n = len(sorted_sizes)
ccdf = np.arange(1, n + 1) / n  # P(X > x) for each x

plt.figure(figsize=(10, 6))
plt.loglog(sorted_sizes, ccdf, 'b.', markersize=2, alpha=0.5)
plt.xlabel("Market Order Size (log scale)")
plt.ylabel("P(X > x) (log scale)")
plt.title("Empirical CCDF - Full Data")
plt.savefig("Empirical_CCDF.png")
plt.clf()

# Try different cutoffs to find where power law behavior starts
cutoffs_to_try = [1, 50, 100, 200, 500]

plt.figure(figsize=(12, 8))
for cutoff in cutoffs_to_try:
    tail_data = mo_sizes[mo_sizes >= cutoff]
    sorted_tail = np.sort(tail_data)[::-1]
    n_tail = len(sorted_tail)
    ccdf_tail = np.arange(1, n_tail + 1) / n_tail
    
    plt.loglog(sorted_tail, ccdf_tail, '.', markersize=3, alpha=0.5, label=f'cutoff={cutoff}')

plt.xlabel("Market Order Size (log scale)")
plt.ylabel("P(X > x) (log scale)")
plt.title("Empirical CCDF with Different Cutoffs")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.savefig("Empirical_CCDF_cutoffs.png")
plt.clf()

###

### Fit power law til halen

cutoff = 1  # Choose based on visual inspection

tail_data = mo_sizes[mo_sizes >= cutoff]
n_tail = len(tail_data)

# Bruger Hill's estimator

zeta_mle = (n_tail - 1) / np.sum(np.log(tail_data / cutoff))
print(f"Cutoff: {cutoff}")
print(f"Number of observations in tail: {n_tail}")
print(f"MLE estimated zeta (tail exponent): {zeta_mle:.4f}")

# Plot fit vs data
sorted_tail = np.sort(tail_data)[::-1]
ccdf_empirical = np.arange(1, n_tail + 1) / n_tail

# Theoretical CCDF: P(X > x) = (x_min / x)^alpha
x_theory = np.linspace(cutoff, sorted_tail[0], 1000)
ccdf_theory = (cutoff / x_theory) ** zeta_mle

plt.figure(figsize=(10, 6))
plt.loglog(sorted_tail, ccdf_empirical, 'b.', markersize=3, alpha=0.5, label='Empirical CCDF')
plt.loglog(x_theory, ccdf_theory, 'r-', linewidth=2)
plt.xlabel("Market Order Size (log scale)")
plt.ylabel("P(X > x) (log scale)")
plt.title(f"Power Law Fit (cutoff={cutoff})")
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.savefig("Power_Law_Fit.png")
plt.clf()