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

t_list = []

for i in range(1,11):
	t_list.append(np.arange(0,0.1*i+DT,DT))

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

def claim_v(v, t, n):
	
	ind_values = np.array([sciint.simps(v[:,i],t) for i in range(n)])

	value = np.mean(ind_values)

	return value, ind_values

def claim_v2(v, t, n):

	ind_values = np.array([sciint.simps(v[:,i]**2,t) for i in range(n)])

	value = np.mean(ind_values)

	return value, ind_values

def claim_v_analytic(T):

	value = ((V0-THETA)/KAPPA)*(1 - np.exp(- KAPPA * T)) + THETA * T

	return value

def claim_v2_analytic(T):

	value = (V0**2 - (2*KAPPA * THETA + ETA**2)*(V0 - THETA)/KAPPA - (2 * KAPPA * THETA + ETA**2)*THETA/(2*KAPPA))*(1 - np.exp(-2*KAPPA*T))/(2 * KAPPA) + (2 * KAPPA * THETA + ETA**2)*(V0-THETA)*(1-np.exp(-KAPPA*T))/(KAPPA**2) + (2*KAPPA*THETA+ETA**2)*THETA*T/(2*KAPPA)

	return value

def Q2b():

	claims_v = []
	claims_v_analytic = []
	claims_v2 = []
	claims_v2_analytic = []
	times = []

	for i in range(len(t_list)):

		BM = Brownian_Motions(RHO, DT, t_list[i], N_SIMS)

		V = milstein_v(V0, KAPPA, THETA, ETA, t_list[i], DT, N_SIMS, BM[0])

		
		claims_v.append(claim_v(V,t_list[i],N_SIMS)[0])
		claims_v_analytic.append(claim_v_analytic(t_list[i][-1]))

		claims_v2.append(claim_v2(V,t_list[i],N_SIMS)[0])
		claims_v2_analytic.append(claim_v2_analytic(t_list[i][-1]))
		times.append(t_list[i][-1])

		#print('T = %.1f :' % (0.1*(i+1)), 'v payoff,', claim_v(V,t_list[i],N_SIMS)[0],'. v^2 payoff,', claim_v2(V,t_list[i],N_SIMS)[0])

		#print('T = %.1f :' % (0.1*(i+1)), 'v analytical payoff,', claim_v_analytic(t_list[i][-1]),'. v^2 analytical payoff,', claim_v2_analytic(t_list[i][-1]))

	plt.plot(times,claims_v,label = 'Simulated')
	plt.plot(times,claims_v_analytic,label = 'Analytical')
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	plt.xlabel('Maturity (T)', fontsize=20)
	plt.ylabel('Value', fontsize=20)
	plt.title('Contingent Claim Paying $\int_0^T v_s ds$ at T',fontsize=24)
	plt.grid()
	plt.legend(fontsize=20)
	plt.show()

	plt.plot(times,claims_v2,label = 'Simulated')
	plt.plot(times,claims_v2_analytic,label = 'Analytical')
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	plt.xlabel('Maturity (T)', fontsize=20)
	plt.ylabel('Value', fontsize=20)
	plt.title('Contingent Claim Paying $\int_0^T v_s^2 ds$ at T',fontsize=24)
	plt.grid()
	plt.legend(fontsize=20)
	plt.show()



def stock_price(s0,x):

	s_array = s0*np.exp(x[-1,:])

	s = np.mean(s_array)

	s_array

	return s, s_array


def euler_x_det(s0, v, t, dt, n, ws):

	v_avg = np.mean(v,axis=0)
	
	x = np.zeros((t.size,n))

	x[0,:] = np.log(s0)

	for i in range(0,n):
		for j in range(1,t.size):
			x[j,i] = x[j-1,i] + (-1/2)*v_avg[i]*dt + np.sqrt(v_avg[i])*ws[j-1,i]

	return x



def option_value_a(s0, k, x, n):
	
	if k >= s0:
		C_array = np.maximum(s0*np.exp(x[-1,:]) - k, 0)
		C = np.sum(C_array) / n
		price = C
		price_lower, price_upper = st.t.interval(0.95, C_array.size-1, loc=np.mean(C_array), scale=st.sem(C_array))
		price_array = C_array

	if k < s0:
		P_array = np.maximum(k - s0*np.exp(x[-1,:]), 0)
		P = np.sum(P_array) / n
		price = P
		price_lower, price_upper = st.t.interval(0.95, P_array.size-1, loc=np.mean(P_array), scale=st.sem(P_array))
		price_array = P_array
		
	return price, price_lower, price_upper, price_array


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

def option_value_det_vol_analytic(s0,k,t_mat):
	
	sigma_v_squared = (1/KAPPA)*(THETA-V0)*(np.exp(KAPPA*t_mat) - 1)+THETA*t_mat
	d_plus = (np.log(s0/k)+sigma_v_squared)/(np.sqrt(sigma_v_squared))
	d_minus = (np.log(s0/k)-sigma_v_squared)/(np.sqrt(sigma_v_squared))
	
	if k >= s0:
		C = s0*st.norm.cdf(d_plus) - k * st.norm.cdf(d_minus)
		price = C
	if k < s0:
		P = k*st.norm.cdf(-d_minus) - s0*st.norm.cdf(-d_plus)
		price = P

	return price


def cont_var_det_vol(s0,k,x,x_det,T,n):

	det_vol, lower, upper, det_vol_array = option_value_a(s0, k, x_det, n)

	option, lower, upper, option_array = option_value_a(s0, k, x, n)

	cov_yy = (1/(n-1)) * np.sum((det_vol_array-det_vol)**2)

	cov_xy = (1/(n-1)) * np.sum((option_array - option)*(det_vol_array-det_vol))

	gamma = cov_xy / cov_yy

	value = option + gamma * (option_value_det_vol_analytic(s0,k,T) - det_vol)

	value_array = option_array + gamma * (option_value_det_vol_analytic(s0,k,T) - det_vol_array)

	value_lower, value_upper = st.t.interval(0.95, value_array.size-1, loc=np.mean(value_array), scale=st.sem(value_array))

	return value, value_lower, value_upper


def cont_var_v(s0,k,x,v,t,T,n):

	claim, claim_array = claim_v(v,t,n)

	option, lower, upper, option_array = option_value_a(s0, k, x, n)

	cov_yy = (1/(n-1)) * np.sum((claim_array-claim)**2)

	cov_xy = (1/(n-1)) * np.sum((option_array - option)*(claim_array-claim))

	gamma = cov_xy / cov_yy

	value = option + gamma * (claim_v_analytic(T) - claim)

	value_array = option_array + gamma * (claim_v_analytic(T) - claim_array)

	value_lower, value_upper = st.t.interval(0.95, value_array.size-1, loc=np.mean(value_array), scale=st.sem(value_array))

	return value, value_lower, value_upper


def cont_var_v2(s0,k,x,v,t,T,n):

	claim, claim_array = claim_v2(v,t,n)

	option, lower, upper, option_array = option_value_a(s0, k, x, n)

	cov_yy = (1/(n-1)) * np.sum((claim_array-claim)**2)

	cov_xy = (1/(n-1)) * np.sum((option_array - option)*(claim_array-claim))

	gamma = cov_xy / cov_yy

	value = option + gamma * (claim_v2_analytic(T) - claim)

	value_array = option_array + gamma * (claim_v2_analytic(T) - claim_array)

	value_lower, value_upper = st.t.interval(0.95, value_array.size-1, loc=np.mean(value_array), scale=st.sem(value_array))

	return value, value_lower, value_upper



def cont_var_s(s0,k,x,n):

	stock, stock_array = stock_price(s0,x)

	option, lower, upper, option_array = option_value_a(s0, k, x, n)

	cov_yy = (1/(n-1)) * np.sum((stock_array-stock)**2)

	cov_xy = (1/(n-1)) * np.sum((option_array - option)*(stock_array-stock))

	gamma = cov_xy / cov_yy

	value = option + gamma * (1 - stock)

	value_array = option_array + gamma * (1 - stock_array)

	value_lower, value_upper = st.t.interval(0.95, value_array.size-1, loc=np.mean(value_array), scale=st.sem(value_array))

	return value, value_lower, value_upper


def cont_var_all(s0,k,x,x_det,v,t,T,n):

	det_vol, lower, upper, det_vol_array = option_value_a(s0, k, x_det, n)

	claim, claim_array = claim_v(v,t,n)

	claim2, claim2_array = claim_v2(v,t,n)

	stock, stock_array = stock_price(s0,x)

	option, lower, upper, option_array = option_value_a(s0, k, x, n)

	#[det_vol,det_vol_array],

	cont_var_list = [[det_vol,det_vol_array],[claim,claim_array],[claim2,claim2_array],[stock,stock_array]]

	cont_var_num = 4

	cov_yy = np.zeros((cont_var_num,cont_var_num))

	for i in range(cont_var_num):
		for j in range(cont_var_num):
			cov_yy[i,j] = (1/(n-1)) * np.sum((cont_var_list[i][1] - cont_var_list[i][0])*(cont_var_list[j][1]-cont_var_list[j][0]))

	cov_xy = np.zeros(cont_var_num)

	for l in range(cont_var_num):
		cov_xy[l] = (1/(n-1)) * np.sum((option_array - option)*(cont_var_list[l][1]-cont_var_list[l][0]))

	gamma = np.matmul(np.linalg.inv(cov_yy), cov_xy) 

	#+ gamma[0] * (option_value_det_vol_analytic(s0,k,T) - det_vol)
	
	value = option + gamma[0] * (option_value_det_vol_analytic(s0,k,T) - det_vol) + gamma[1] * (claim_v_analytic(T) - claim) +  gamma[2] * (claim_v2_analytic(T) - claim2) + gamma[3] * (1 - stock)

	#+ gamma[0] * (option_value_det_vol_analytic(s0,k,T) - det_vol_array)

	value_array = option_array + gamma[0] * (option_value_det_vol_analytic(s0,k,T) - det_vol_array) +  gamma[1] * (claim_v_analytic(T) - claim_array) + gamma[2] * (claim_v2_analytic(T) - claim2_array) + gamma[3] * (1 - stock_array)

	value_lower, value_upper = st.t.interval(0.95, value_array.size-1, loc=np.mean(value_array), scale=st.sem(value_array))

	return value, value_lower, value_upper


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

			X_DET = euler_x_det(S0, V, t_all[i], DT, N_SIMS, BM[1])

			var_vol = cont_var_det_vol(S0,strike,X,X_DET,T_MAT[i],N_SIMS)

			var_vol_det = cont_var_det_vol(S0,strike,X_DET,X_DET,T_MAT[i],N_SIMS)

			#var_v = cont_var_v(S0, strike, X, V, t_all[i], T_MAT[i], N_SIMS)

			#var_v_det = cont_var_v(S0, strike, X_DET, V, t_all[i], T_MAT[i], N_SIMS)

			#var_v2 = cont_var_v2(S0, strike, X, V, t_all[i], T_MAT[i], N_SIMS)

			#var_v2_det = cont_var_v2(S0, strike, X_DET, V, t_all[i], T_MAT[i], N_SIMS)

			#var_s = cont_var_s(S0, strike, X, N_SIMS)

			#var_s_det = cont_var_s(S0, strike, X_DET, N_SIMS)

			#var_all = cont_var_all(S0,strike,X,X_DET,V,t_all[i],T_MAT[i],N_SIMS)

			#var_all_det = cont_var_all(S0,strike,X_DET ,X_DET,V,t_all[i],T_MAT[i],N_SIMS)

			#print(var_vol, var_vol_det)

			a = var_vol

			b = var_vol_det

						
			def func_a(sigma):
				f = option_value_BS(S0, strike, sigma, T_MAT[i]) - a[0]
				return f

			def func_a_lower(sigma):
				f = option_value_BS(S0, strike, sigma, T_MAT[i]) - a[1]
				return f

			def func_a_upper(sigma):
				f = option_value_BS(S0, strike, sigma, T_MAT[i]) - a[2]
				return f
			
			def func_b(sigma):
				f = option_value_BS(S0, strike, sigma, T_MAT[i]) - b[0]
				return f

			def func_b_lower(sigma):
				f = option_value_BS(S0, strike, sigma, T_MAT[i]) - b[1]
				return f

			def func_b_upper(sigma):
				f = option_value_BS(S0, strike, sigma, T_MAT[i]) - b[2]
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

	with open('volatilities_all.csv', 'w') as csvfile:
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

	suffix = ['_vol']#,'_all']#['vol','v','v2','s']

	for suf in suffix:

		values =[]

		with open('volatilities{}.csv'.format(suf), 'r') as csvfile:
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

			#print((np.mean(np.absolute(a[:,i]-a_low[:,i]))+np.mean(np.absolute(a[:,i]-a_up[:,i])))/2)

			print((np.mean(np.absolute(b[:,i]-b_low[:,i]))+np.mean(np.absolute(b[:,i]-b_up[:,i])))/2)

		
			plt.plot(k_t[:len(K),0],a[:,i],label = 'Implied Volatility {}'.format('Variate'))
			plt.plot(k_t[len(K):2*len(K),0],a_low[:,i],label = 'Lower Quantile {}'.format('Variate'))
			plt.plot(k_t[2*len(K):,0],a_up[:,i],label = 'Upper Quantile {}'.format('Variate'))
			plt.legend(fontsize = 20)
			plt.grid()
			plt.xticks(fontsize = 20)
			plt.yticks(fontsize = 20)
			plt.xlabel('Strike (K)', fontsize = 20)
			plt.ylabel('Implied Volatility', fontsize = 20)
			plt.title('Heston Model with All Control Variates and the Mixing Method, T = {} '.format(T_MAT[i]), fontsize = 24)
			plt.show()
		
		
			plt.plot(k_t[:len(K),0],b[:,i],label = 'Implied Volatility {}'.format('Mixing'))
			plt.plot(k_t[len(K):2*len(K),0],b_low[:,i],label = 'Lower Quantile {}'.format('Mixing'))
			plt.plot(k_t[2*len(K):,0],b_up[:,i],label = 'Upper Quantile {}'.format('Mixing'))
			plt.legend(fontsize = 20)
			plt.grid()
			plt.xticks(fontsize = 20)
			plt.yticks(fontsize = 20)
			plt.xlabel('Strike (K)', fontsize = 20)
			plt.ylabel('Implied Volatility', fontsize = 20)
			plt.title('Deterministic Volatility Model with Control Variate Stock Price, T = {} '.format(T_MAT[i]), fontsize = 24)
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
		ax.set_title('Heston Model with Control Variate Stock Price', fontsize = 24)
		plt.show()

		fig = plt.figure(figsize=(4,4))
		ax = fig.add_subplot(111, projection='3d')
		ax.plot_surface(x, y, b, cmap="plasma")
		ax.tick_params(axis='both', which='major', labelsize=13)
		ax.tick_params(axis='both', which='minor', labelsize=13)
		ax.set_xlabel('Maturity (T)', fontsize = 14)
		ax.set_ylabel('Strike (K)', fontsize = 14)
		ax.set_zlabel('Implied Volatility', fontsize = 14)
		ax.set_title('Deterministic Volatility Model with Control Variate Stock Price', fontsize = 24)
		plt.show()
	

	return 0


find_imp_vol()

#plotting()


