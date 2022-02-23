import math
import pandas as pd
import numpy as np
import numpy_financial as npf


# Entering the loan amount
isCorrectAmount = True
while isCorrectAmount:
    enteredLoanAmount = float(input("Enter Loan amount (Min - 1,000,000): "))
    if enteredLoanAmount < 1_000_000:
        print("Minimum Loan amount is 1,000,000")
        enteredLoanAmount = float(input("Enter Loan amount (Min - 1,000,000): "))
    else:
        isCorrectAmount = False
        
# Entering the age
isCorrectAge = True
while isCorrectAge:
    enteredAge = int(input("Enter Age (18 - 50 years): "))
    if enteredAge < 18 or enteredAge > 50:
        print("Sorry, age does not meet criteria!")
        enteredAge = int(input("Enter Age (18 - 50 years):"))
    else:
        isCorrectAge = False

# Entering the loan term
isCorrectTerm = True
while isCorrectTerm:
    enteredTerm = int(input("Enter Loan term (3 - 35 years): "))
    if enteredTerm < 3 or enteredTerm > 35:
        print("Sorry, loan term does not meet criteria!")
        enteredTerm = int(input("Enter Loan term (3 - 35 years):"))
    else:
        isCorrectTerm = False
        
# Entering the gender
enteredGender = input("Enter Gender (MALE or FEMALE): ")


# Assumptions
general_assumptions = pd.read_csv('data/general_assumptions.csv')
surrender_rates = pd.read_csv('data/surrender_rates.csv')
mortality_rates = pd.read_csv('data/mortality_rates.csv')
disability_rates = pd.read_csv('data/disability_rates.csv')
critical_illness = pd.read_csv('data/critical_illness_rates.csv')
amortization_assumptions = pd.read_csv('data/amortization_assumptions.csv')

sumAssured = enteredLoanAmount
coverExpiry = enteredAge + enteredTerm
if coverExpiry > 55:
    print("Cover exceeds 55 years !")


def vlookup(value,df,coli,colo):
    """ 
    vlookup function to reference other tables
    """
    return next(iter(df.loc[df[coli]==value][colo]), None)


# Mortality Calculations
radix = float((general_assumptions[(general_assumptions['assumption'] == 'radix')]).amount)
def mortality_calc():
    """ Function to do calculations based on mortality rates """
    mortality = []
    lx = []
    dx = []
    Dx = []
    Nx = []
    Cx = []
    Mx = []
    Rx = []
    Sx = []
    if enteredGender == 'male' or enteredGender == 'MALE':
        qx = 'Graduated Rates(qx) Males'
    else:
        qx = 'Graduated Rates(qx) Females'
        
    for i in range(len(mortality_rates)):
        agex = 18 + i
        if i == 0:
            lx_value = radix
            lx.append(lx_value)
        else:
            lx_value = (lx[-1]) - (vlookup(agex,mortality_rates,'age x',qx)*lx[-1])
            lx.append(lx_value)
        Dx_value = (lx_value * ((1+0.04)**(-agex)))
        Dx.append(Dx_value)
            
    for i in range(len(mortality_rates)):
        agex = 18 + i
        try:
            dx_value = lx[i] - lx[i+1]
        except IndexError:
            dx_value = lx[-1]
        dx.append(dx_value)
        Nx_value = sum(Dx[i:])
        Nx.append(Nx_value)
        Cx_value = ((1/((1+0.04)**(agex+1))) * dx_value)
        Cx.append(Cx_value)
    
    for i in range(len(mortality_rates)):
        Mx_value = sum(Cx[i:])
        Mx.append(Mx_value)
    
    for i in range(len(mortality_rates)):
        Rx_value = sum(Mx[i:])
        Rx.append(Rx_value)
    
    for i in range(len(mortality_rates)):
        Sx_value = sum(Nx[i:])
        Sx.append(Sx_value)
    
    for i in range(len(mortality_rates)):        
        agex = 18 + i
        qx_val = (vlookup(agex,mortality_rates,'age x',qx))
        mortality.append({'age x': agex, qx : qx_val, 'lx': lx[i], 'dx': dx[i], 'Dx': Dx[i],
                         'Nx': Nx[i], 'Cx': Cx[i], 'Mx': Mx[i], 'Rx': Rx[i], 'Sx': Sx[i]})
        
    return pd.DataFrame(mortality)
         

mortality_calculations = mortality_calc()
numpy_mortality = np.array(mortality_calculations)


# Assurances and Annuities
def assurances_calc():
    """ Function to calculate assurances and annuities """
    aNa = []
    ax_val_list = []
    qx_list = []
    lx_list = []
    Mx_list = []
    Dx = []
    axn_list = []
    ax_n_list = []
    dxn_list = []
    Ax_list = []
    Axn_list = []
    Ax1n_list = []
    t1v_list = []
    if enteredGender == 'male' or enteredGender == 'MALE':
        qx = 'Graduated Rates(qx) Males'
    else:
        qx = 'Graduated Rates(qx) Females'
    
    for i in range(46):
        X = enteredAge + i
        qx_val = vlookup(X, mortality_calculations, 'age x', qx)
        qx_list.append(qx_val)
        lx = vlookup(X, mortality_calculations, 'age x', 'lx')
        lx_list.append(lx)
        Dx_val = vlookup(X, mortality_calculations, 'age x', 'Dx')
        Dx.append(Dx_val)
        Mx = vlookup(X, mortality_calculations, 'age x', 'Mx')
        Mx_list.append(Mx)
        
    for i in range(46):
        t = i + 1
        X = enteredAge + i
        ax_val = (sum(Dx[i:]))/Dx[i]
        ax_val_list.append(ax_val)
        if i == 0:
            n = enteredTerm      
        if n < 1:
            n = 0
        aNa.append({'t': t, 'X': X, 'qx': qx_list[i], 'lx': lx_list[i], 'Dx': Dx[i],
                    'Mx': Mx_list[i], 'ax': ax_val_list[i], 'n': n})
        n = n-1
        
    ana_dataframe = pd.DataFrame(aNa)
    
    for i in range(46):
        X = enteredAge + i
        if i == 0:
            n = enteredTerm      
        if n < 1:
            n = 0
        axn = vlookup((X+n), ana_dataframe, 'X', 'ax')
        axn_list.append(axn)
        dxn = vlookup((X+n), ana_dataframe, 'X', 'Dx')
        dxn_list.append(dxn)
        n = n - 1
        ax_n = ax_val_list[i] - ((dxn/Dx[i])*axn)
        ax_n_list.append(ax_n)
        Ax = Mx_list[i]/Dx[i]
        Ax_list.append(Ax)
        
    ana_dataframe['ax+n'] = axn_list
    ana_dataframe['Dx+n'] = dxn_list
    ana_dataframe['ax:n'] = ax_n_list
    ana_dataframe['Ax'] = Ax_list
    
    for i in range(46):
        X = enteredAge + i
        if i == 0:
            n = enteredTerm      
        if n < 1:
            n = 0
        Axn = vlookup((X+n), ana_dataframe, 'X', 'Ax')
        Axn_list.append(Axn)
        n = n - 1
        
    ana_dataframe['Ax+n'] = Axn_list
    
    for i in range(46):
        Ax1n = Ax_list[i] - ((dxn_list[i]/Dx[i])*Axn_list[i])
        Ax1n_list.append(Ax1n)
        
    ana_dataframe['Ax1:n'] = Ax1n_list
        
    #Net Premium
    net_premium = (enteredLoanAmount*Ax1n_list[0])/ax_n_list[0]
    
    for i in range(46):
        t = i+1        
        if i == 0:
            t1v = 0
        elif t < enteredTerm:
            t1v = (enteredLoanAmount*Ax1n_list[i]) - (net_premium*ax_n_list[i])
        else:
            t1v = 0
        t1v_list.append(t1v)
        
    ana_dataframe['t-1 v reserve'] = t1v_list
    
    return ana_dataframe


assurances = assurances_calc()
numpy_assurances = np.array(assurances)


def create_decrements_table():
    """ function to create the decrements table """
    decrements_table = []
    years = 36
    if enteredGender == 'male' or enteredGender == 'MALE':
        dx = 'Graduated Rates(qx) Males'
        bx = 'males'
    else:
        dx = 'Graduated Rates(qx) Females'
        bx = 'females'
    
    for year in range(years):
        if year == 0:
            agex = enteredAge
            t_1apx = 1
        else:
            agex = enteredAge + year
            t_1apx = decrements_table[year-1]['t-1apx'] * decrements_table[year-1]['apx']
            
        if (year+1) <= enteredTerm:
            qdx = vlookup(agex,mortality_calculations,'age x',dx)
            qbx = vlookup(agex,disability_rates,'age x',bx)
            qcx = vlookup(agex,critical_illness,'age x',bx)
            aqdx = qdx*(1-(0.5*(qbx+qcx))+((1/3)*qbx*qcx))
            aqbx = qbx*(1-(0.5*(qdx+qcx))+((1/3)*qdx*qcx))
            aqcx = qcx*(1-(0.5*(qbx+qdx))+((1/3)*qbx*qdx))
            apx = 1-aqdx-aqbx
        else:
            qdx = 0
            qbx = qdx
            qcx = qdx
            aqdx = qdx
            aqbx = qdx
            aqcx = qdx
            apx = qdx
            agex = qdx
        
        decrements_table.append({'year t':year+1, 'age':agex, 'qdx':qdx, 'qbx':qbx, 'qcx':qcx, 'aqdx':aqdx, 'aqbx':aqbx, 
                                 'aqcx':aqcx, 'apx':apx, 't-1apx':t_1apx})
    
    return pd.DataFrame(decrements_table)


# Create the decrements table
decrements_table = create_decrements_table()
numpy_decrements = np.array(decrements_table)


def create_amortization_table():
    """ Function to create the armotization table """
    amortization_table = []    
    period = enteredTerm + 1
    principal = enteredLoanAmount
    installment = npf.pmt(0.042/1,enteredTerm,-principal)
    
    for i in range(period):
        if i == 0:
            payment = 0
            interest = 0
            new_principal = 0
            new_balance = principal
        else:
            payment = installment
            interest = npf.ipmt(0.042/1,i,enteredTerm,-principal)
            new_principal = npf.ppmt(0.042/1,i,enteredTerm,-principal)
            new_balance = amortization_table[i-1]['Balance'] - new_principal
        
        amortization_table.append({'Period':i, 'Payment':payment, 'Interest':interest, 'Principal':new_principal,
                                   'Balance':new_balance})
        
    return pd.DataFrame(amortization_table)


# Create the amortization table
amortization_table = create_amortization_table()
numpy_amortization = np.array(amortization_table)


# Determining Premiums and Profit
initial_expenses = 0.35
second_year_expenses = 0.15
subsequent_years_expenses = 0.10
initial_commission = 0.20
second_year_commission = 0.07
subsequent_year_commission = 0.05
critical_illness_rate = 0.5
risk_discount_rate = 0.1
annual_premium = (enteredLoanAmount*numpy_assurances[0][13])/(-0.55+0.37+numpy_assurances[0][10]-0.37*numpy_assurances[0][10])
 
def calculate_premiums_and_profit():
    """Function to calculate premiums and profit """
    premiums = []
    period = enteredTerm + 1   
    interest_rate = 0.1
    
    for i in range(period):
        t = i + 1
        if i == 0:
            expenses = initial_expenses * annual_premium
            commission = initial_commission * annual_premium
        elif i == 1:
            expenses = second_year_expenses * annual_premium
            commission = second_year_commission * annual_premium
        else:
            expenses = subsequent_years_expenses * annual_premium
            commission = subsequent_year_commission * annual_premium
        net_cashflow = annual_premium - expenses - commission
        interest = net_cashflow * interest_rate
        try:
            loan_death_cost = numpy_decrements[i][5] * numpy_amortization[i+1][-1]
            remaining_balance = (enteredLoanAmount-numpy_amortization[i+1][-1])*numpy_decrements[i][5]
            disability_cost = numpy_decrements[i][6]*numpy_amortization[i+1][-1]
            critical_illness_cost = critical_illness_rate*numpy_amortization[i+1][-1]*numpy_decrements[i][7]
            cost_increase_provisions = (numpy_decrements[i][8]*numpy_assurances[i+1][-1]) - numpy_assurances[i][-1]
        except IndexError:
            pass
        profit = net_cashflow+interest-loan_death_cost-remaining_balance-disability_cost-cost_increase_provisions
        profit_signature = profit*numpy_decrements[i][-1]
        discount_vt = (1/(1+risk_discount_rate))**t
        discounted_profit = profit_signature*discount_vt
        discount_vt1 =  (1/(1+risk_discount_rate))**(t-1)
        discounted_premium = discount_vt1*annual_premium*numpy_decrements[i][-1]
        premiums.append({'time t': t, 'Premium': annual_premium, 'Expenses': expenses, 'Commission': commission,
                        'Net cashflow': net_cashflow, 'Interest': interest, 
                         'Outstanding loan death cost': loan_death_cost,
                         'Remaining balance to beneficiaries': remaining_balance, 'Disability cost': disability_cost,
                        'Critical illness cost': critical_illness_cost, 'Cost of increase in provisions': cost_increase_provisions,
                        'Profit':profit, 'Profit Signature': profit_signature, 'Discount vt': discount_vt,
                        'Discounted profit': discounted_profit, 'Discount vt-1': discount_vt1, 'Discounted Premium': discounted_premium})
        
    return pd.DataFrame(premiums)


premiums_and_profit = calculate_premiums_and_profit()


# NPV of profits
NPV_profits = sum(premiums_and_profit['Profit'])

# NPV of premiums
NPV_premiums = sum(premiums_and_profit['Discounted Premium'])

# Profit Margin
profit_margin = round((NPV_profits/NPV_premiums)*100,2)

# Round off annual premium to 0 decimal places
annual_premium = round(annual_premium,0)

print("\n\n############## PROGRAM OUTPUT ##############")
print(f"\nAnnual Premium to pay: UGX {annual_premium}")
print(f"\nCover up to: {coverExpiry} years of age")
print(f"\nProfit Margin: {profit_margin}%")
