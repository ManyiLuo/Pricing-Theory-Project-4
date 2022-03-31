import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as st
import scipy.integrate as sciint
import scipy.optimize as opt
import csv
import ast


np.random.seed(0)

S0 = 1
V0 = 0.2**2
KAPPA = 3
THETA = 0.4**2
ETA = 1.5
RHO = -0.5


N_SIMS = 5000

K = [0.8,0.85,0.9,0.95,1.0,1.05,1.10,1.15,1.2]

DT = 1/1000

T_MAT = [1/4,1/2,1]

t_quart = np.arange(0,T_MAT[0]+DT,DT)
t_semi = np.arange(0,T_MAT[1]+DT,DT)
t_ann = np.arange(0,T_MAT[2]+DT,DT)

t_all = [t_quart, t_semi, t_ann]


def Brownian_Motions(rho, dt, t, n):
	wv = np.random.normal(loc=0,scale=np.sqrt(dt),size=(t.size,n))
	z = np.random.normal(loc=0,scale=np.sqrt(dt),size=(t.size,n))
	ws = rho * wv + np.sqrt(1- rho**2) * z

	return wv, ws


def milstein_v(v0, kappa, theta, eta, t, dt, n, wv):

	v = np.zeros((t.size,n))

	v[0,:] = v0

	for i in range(0,n):
		for j in range(1,t.size):
			v[j,i] = max(v[j-1,i] + kappa*(theta-v[j-1,i])*(dt) + eta*np.sqrt(v[j-1,i])*(wv[j-1,i]) + (1/4)*(eta**2)*((wv[j-1,i])**2 - dt),0)	

	return v

def euler_x(s0, v, t, dt, n, ws):

	x = np.zeros((t.size,n))

	x[0,:] = np.log(s0)

	for i in range(0,n):
		for j in range(1,t.size):
			x[j,i] = x[j-1,i] + (-1/2)*v[j-1,i]*dt + np.sqrt(v[j-1,i])*ws[j-1,i]

	return x

def option_value_a(s0, k, x, n):
	
	if k >= s0:
		C_array = np.maximum(s0*np.exp(x[-1,:]) - k, 0)
		C = np.sum(C_array) / n
		price = C
		price_lower, price_upper = st.t.interval(0.95, C_array.size-1, loc=np.mean(C_array), scale=st.sem(C_array))

	if k < s0:
		P_array = np.maximum(k - s0*np.exp(x[-1,:]), 0)
		P = np.sum(P_array) / n
		price = P
		price_lower, price_upper = st.t.interval(0.95, P_array.size-1, loc=np.mean(P_array), scale=st.sem(P_array))
		
	return price, price_lower, price_upper


def option_value_b(s0, k, v, t, rho, wv, t_mat, n):

	a = np.array([(-1/2) * sciint.simps(v[:,i],t) + rho* np.sum(np.sqrt(v[:,i]) * wv[:,i]) for i in range(n)])
	b_squared = np.array([(1 - rho ** 2) * sciint.simps(v[:,i],t) for i in range(n)])
	s0_tilde = s0*np.exp(a + (1/2)*b_squared)
	d_plus = (np.log(s0_tilde/k)+(1/2)*b_squared)/(np.sqrt(b_squared))
	d_minus = (np.log(s0_tilde/k)-(1/2)*b_squared)/(np.sqrt(b_squared))

	if k >= s0:
		C_array = s0_tilde*st.norm.cdf(d_plus) - k*st.norm.cdf(d_minus)
		C = np.sum(C_array) / n
		price = C
		price_lower, price_upper = st.t.interval(0.95, C_array.size-1, loc=np.mean(C_array), scale=st.sem(C_array))

	if k < s0:
		P_array = k*st.norm.cdf(-d_minus) - s0_tilde*st.norm.cdf(-d_plus)
		P = np.sum(P_array) / n
		price = P
		price_lower, price_upper = st.t.interval(0.95, P_array.size-1, loc=np.mean(P_array), scale=st.sem(P_array))

	return price, price_lower, price_upper


def option_value_BS(s0, k, sigma, t_mat):
	d_plus = (np.log(s0/k)+(1/2)*sigma**2 * t_mat)/(sigma * np.sqrt(t_mat))
	d_minus = (np.log(s0/k)-(1/2)*sigma**2 * t_mat)/(sigma * np.sqrt(t_mat))
	if k >= s0:
		C = s0*st.norm.cdf(d_plus) - k * st.norm.cdf(d_minus)
		price = C
	if k < s0:
		P = k*st.norm.cdf(-d_minus) - s0*st.norm.cdf(-d_plus)
		price = P

	return price



def find_imp_vol():
	imp_vol_a_all = []
	imp_vol_a_low_all = []
	imp_vol_a_up_all = []

	imp_vol_b_all = []
	imp_vol_b_low_all = []
	imp_vol_b_up_all = []

	parameters = []

	for i in range(len(t_all)):
		for strike in K:

			BM = Brownian_Motions(RHO, DT, t_all[i], N_SIMS)

			V = milstein_v(V0, KAPPA, THETA, ETA, t_all[i], DT, N_SIMS, BM[0])

			X = euler_x(S0, V, t_all[i], DT, N_SIMS, BM[1])

			
			def func_a(sigma):
				f = option_value_BS(S0, strike, sigma, T_MAT[i]) - option_value_a(S0, strike, X, N_SIMS)[0]
				return f

			def func_a_lower(sigma):
				f = option_value_BS(S0, strike, sigma, T_MAT[i]) - option_value_a(S0, strike, X, N_SIMS)[1]
				return f

			def func_a_upper(sigma):
				f = option_value_BS(S0, strike, sigma, T_MAT[i]) - option_value_a(S0, strike, X, N_SIMS)[2]
				return f
			
			def func_b(sigma):
				f = option_value_BS(S0, strike, sigma, T_MAT[i]) - option_value_b(S0, strike, V, t_all[i], RHO, BM[0], T_MAT[i] , N_SIMS)[0]
				return f

			def func_b_lower(sigma):
				f = option_value_BS(S0, strike, sigma, T_MAT[i]) - option_value_b(S0, strike, V, t_all[i], RHO, BM[0], T_MAT[i] , N_SIMS)[1]
				return f

			def func_b_upper(sigma):
				f = option_value_BS(S0, strike, sigma, T_MAT[i]) - option_value_b(S0, strike, V, t_all[i], RHO, BM[0], T_MAT[i] , N_SIMS)[2]
				return f

			imp_vol_a = opt.fsolve(func_a, 0.5, maxfev = 10000)
			imp_vol_a_low = opt.fsolve(func_a_lower, 0.5, maxfev = 10000)
			imp_vol_a_up = opt.fsolve(func_a_upper, 0.5, maxfev = 10000)
			
			imp_vol_b = opt.fsolve(func_b, 0.5, maxfev = 10000)
			imp_vol_b_low = opt.fsolve(func_b_lower, 0.5, maxfev = 10000)
			imp_vol_b_up = opt.fsolve(func_b_upper, 0.5, maxfev = 10000)

			imp_vol_a_all.append(imp_vol_a[0])
			imp_vol_a_low_all.append(imp_vol_a_low[0])
			imp_vol_a_up_all.append(imp_vol_a_up[0])

			imp_vol_b_all.append(imp_vol_b[0])
			imp_vol_b_low_all.append(imp_vol_b_low[0])
			imp_vol_b_up_all.append(imp_vol_b_up[0])

			parameters.append([strike, T_MAT[i]])

			print(strike,T_MAT[i])

			

	print(imp_vol_a_all)
	print(imp_vol_b_all)
	print(parameters)

	with open('volatilities.csv', 'w') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(parameters)
		writer.writerow(imp_vol_a_all)
		writer.writerow(imp_vol_a_low_all)
		writer.writerow(imp_vol_a_up_all)
		writer.writerow(imp_vol_b_all)
		writer.writerow(imp_vol_b_low_all)
		writer.writerow(imp_vol_b_up_all)

	return 0



def plotting():

	values =[]

	with open('volatilities.csv', 'r') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			values.append([lst.strip() for lst in row])

	k_t = np.array([ast.literal_eval(i) for i in values[0]])

	

	vol_a = [float(i) for i in values[2]]
	vol_a_low = [float(i) for i in values[4]]
	vol_a_up = [float(i) for i in values[6]]
	vol_b = [float(i) for i in values[8]]
	vol_b_low = [float(i) for i in values[10]]
	vol_b_up = [float(i) for i in values[12]]


	

	a = np.zeros((len(K),len(t_all)))
	a_low = np.zeros((len(K),len(t_all)))
	a_up = np.zeros((len(K),len(t_all)))
	b = np.zeros((len(K),len(t_all)))
	b_low = np.zeros((len(K),len(t_all)))
	b_up = np.zeros((len(K),len(t_all)))

	l = 0
	for j in range(0,len(t_all)):
		for i in range(0,len(K)):

			a[i,j] = vol_a[l]

			a_low[i,j] = vol_a_low[l]

			a_up[i,j] = vol_a_up[l]

			b[i,j] = vol_b[l]

			b_low[i,j] = vol_b_low[l]

			b_up[i,j] = vol_b_up[l]

			l += 1

			
	for i in range(0,len(t_all)):
		
		plt.plot(k_t[:len(K),0],a[:,i],label = 'Implied Volatility')
		plt.plot(k_t[len(K):2*len(K),0],a_low[:,i],label = 'Lower Quantile')
		plt.plot(k_t[2*len(K):,0],a_up[:,i],label = 'Upper Quantile')
		plt.legend(fontsize = 20)
		plt.grid()
		plt.xticks(fontsize = 20)
		plt.yticks(fontsize = 20)
		plt.xlabel('Strike (K)', fontsize = 20)
		plt.ylabel('Implied Volatility', fontsize = 20)
		plt.title('Discretization Method, T = {} '.format(T_MAT[i]), fontsize = 24)
		plt.show()

		plt.plot(k_t[:len(K),0],b[:,i],label = 'Implied Volatility')
		plt.plot(k_t[len(K):2*len(K),0],b_low[:,i],label = 'Lower Quantile')
		plt.plot(k_t[2*len(K):,0],b_up[:,i],label = 'Upper Quantile')
		plt.legend(fontsize = 20)
		plt.grid()
		plt.xticks(fontsize = 20)
		plt.yticks(fontsize = 20)
		plt.xlabel('Strike (K)', fontsize = 20)
		plt.ylabel('Implied Volatility', fontsize = 20)
		plt.title('Mixing Method, T = {} '.format(T_MAT[i]), fontsize = 24)
		plt.show()
	

	x,y = np.meshgrid(T_MAT,K)

	fig = plt.figure(figsize=(4,4))
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(x, y, a, cmap="plasma")
	ax.tick_params(axis='both', which='major', labelsize=13)
	ax.tick_params(axis='both', which='minor', labelsize=13)
	ax.set_xlabel('Maturity (T)', fontsize = 14)
	ax.set_ylabel('Strike (K)', fontsize = 14)
	ax.set_zlabel('Implied Volatility', fontsize = 14)
	ax.set_title('Discretization Method', fontsize = 24)
	plt.show()

	fig = plt.figure(figsize=(4,4))
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(x, y, b, cmap="plasma")
	ax.tick_params(axis='both', which='major', labelsize=13)
	ax.tick_params(axis='both', which='minor', labelsize=13)
	ax.set_xlabel('Maturity (T)', fontsize = 14)
	ax.set_ylabel('Strike (K)', fontsize = 14)
	ax.set_zlabel('Implied Volatility', fontsize = 14)
	ax.set_title('Mixing Method', fontsize = 24)
	plt.show()

	return 0


#find_imp_vol()

plotting()