# -*- coding: utf-8 -*-
###################################################################
# The code is referenced from: 
# https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_tipping_problem.html#example-plot-tipping-problem-py
####################################################################

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

###################################################################
# INPUT VARIABLES
#   "Monthly Income of Loan Recipient" 
#   "Credit history of Loan Recipient"
# OUTPUT VARIABLES
#   "Credit Score of Loan Recipient" 
###################################################################

#Range of Monthly Income will be between 0 and 20,000 USD
x_income = np.arange(0, 20001, 1)

#Range of Credit History will be between 0 and 10
x_credit_history = np.arange(0, 11, 1)

#Range of Credit Score will be between 0 and 10
x_credit_score  = np.arange(0, 11, 1)

###################################################################
# FUZZY MEMBERSHIP FUNCTIONS
#   Monthly Income will be classified as Low, Medium, High
#   Credit History will be classified as Poor, Average, Good
#   Credit Score will be classified as Low, Medium, High
###################################################################

#Income Membership Function
income_low = fuzz.trimf(x_income, [0, 0, 7500])
income_mid = fuzz.trimf(x_income, [2500, 10000, 17500])
income_high = fuzz.trimf(x_income, [12500, 20000, 20000])

#Credit History Membership Function
history_poor = fuzz.trimf(x_credit_history, [0, 0, 5])
history_avg = fuzz.trimf(x_credit_history, [0, 5, 10])
history_good = fuzz.trimf(x_credit_history, [5, 10, 10])

#Credit Score Membership Function
score_low = fuzz.trimf(x_credit_score, [0, 0, 5])
score_mid = fuzz.trimf(x_credit_score, [0, 5, 10])
score_high = fuzz.trimf(x_credit_score, [5, 10, 10])

###################################################################
# VISUALIZATION OF MEMBERSHIP FUNCTIONS
###################################################################

# Initiating Figure Structure
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(10, 10))

# Plotting the Income Membership Functions
ax0.plot(x_income, income_low, 'r', linewidth=1.5, label='Low')
ax0.plot(x_income, income_mid, 'b', linewidth=1.5, label='Medium')
ax0.plot(x_income, income_high, 'g', linewidth=1.5, label='High')
ax0.set_title('Monthly Income of Loan Recipient')
ax0.legend()

# Plotting the Credit History Membership Functions 
ax1.plot(x_credit_history, history_poor, 'r', linewidth=1.5, label='Low')
ax1.plot(x_credit_history, history_avg, 'b', linewidth=1.5, label='Medium')
ax1.plot(x_credit_history, history_good, 'g', linewidth=1.5, label='High')
ax1.set_title('Credit History of Loan Recipient')
ax1.legend()

# Plotting the Credit Score Membership Functions
ax2.plot(x_credit_score, score_low, 'r', linewidth=1.5, label='Low')
ax2.plot(x_credit_score, score_mid, 'b', linewidth=1.5, label='Medium')
ax2.plot(x_credit_score, score_high, 'g', linewidth=1.5, label='High')
ax2.set_title("Credit Score of Loan Recipient")
ax2.legend()

# Plotting all the graphs
plt.tight_layout()

###################################################################
# FUZZIFICATION 
#   Income used in this code = USD 13,500
#   Credit History used in this code = 8.5
###################################################################

#Degree of Membership for Income Membership Function
income_deg_low = fuzz.interp_membership(x_income, income_low, 13500)
income_deg_mid = fuzz.interp_membership(x_income, income_mid, 13500)
income_deg_high = fuzz.interp_membership(x_income, income_high, 13500)

#Degree of Membership for Credit History Membership Function
history_deg_poor = fuzz.interp_membership(x_credit_history, history_poor, 8.5)
history_deg_avg = fuzz.interp_membership(x_credit_history, history_avg, 8.5)
history_deg_good = fuzz.interp_membership(x_credit_history, history_good, 8.5)

##################################################################
# FUZZY RULES
#   RULE 1: IF Income is high AND Credit History is good
#           THEN Credit Score is high
#   RULE 2: IF Income is low OR Credit History is poor
#           THEN Credit Score is low
#   RULE 3: IF Income is medium AND Credit History is good
#           THEN Credit Score is medium
##################################################################

# RULE 1 applied and remaining area clipped 
rule1 = np.fmin(income_deg_high, history_deg_good)
score_act_high = np.fmin(rule1, score_high)

# RULE 2 applied and remaining area clipped
rule2 = np.fmax(income_deg_low, history_deg_poor)
score_act_low = np.fmin(rule2, score_low)

# RULE 3 applied and remaining area clipped
rule3 = np.fmin(income_deg_mid, history_deg_good)
score_act_mid = np.fmin(rule3, score_mid)

##################################################################
# VISUALIZATION
##################################################################

fig, ax0 = plt.subplots(figsize=(10, 5))
ax0.plot(x_credit_score, score_low, 'r', linewidth=0.5, linestyle='--', )
ax0.fill_between(x_credit_score, score_act_low, facecolor='r', alpha=0.7)
ax0.plot(x_credit_score, score_mid, 'b', linewidth=0.5, linestyle='--', )
ax0.fill_between(x_credit_score, score_act_mid, facecolor='b', alpha=0.7)
ax0.plot(x_credit_score, score_high, 'g', linewidth=0.5, linestyle='--', )
ax0.fill_between(x_credit_score, score_act_high, facecolor='g', alpha=0.7)

plt.tight_layout()

##################################################################
# RULE AGGREGATION
##################################################################

agg = np.fmax(score_act_high, np.fmax(score_act_low, score_act_mid))

###################################################################
# DEFUZZIFICATION
# Centroid technique used as the defuzzification method
###################################################################

defuzz_result = fuzz.defuzz(x_credit_score, agg, 'centroid')
print(defuzz_result)

##################################################################
# VISUALIZATION
##################################################################

score_act = fuzz.interp_membership(x_credit_score, agg, defuzz_result)

fig, ax0 = plt.subplots(figsize=(10, 5))
ax0.plot(x_credit_score, score_low, 'r', linewidth=0.5, linestyle='--', )
ax0.plot(x_credit_score, score_mid, 'b', linewidth=0.5, linestyle='--')
ax0.plot(x_credit_score, score_high, 'g', linewidth=0.5, linestyle='--')
ax0.plot([defuzz_result, defuzz_result], [0, score_act], 'k', linewidth=1.5, alpha=0.9)
ax0.fill_between(x_credit_score, agg, alpha=0.7)

plt.tight_layout()






















