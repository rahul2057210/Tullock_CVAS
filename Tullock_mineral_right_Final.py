from scipy.optimize import fsolve
from scipy.optimize import root
from scipy.optimize import minimize
import math
#import numpy as np
import matplotlib.pyplot as plt 
import random
from scipy.stats import uniform
from scipy.special import comb
import autograd
from autograd import grad
import autograd.numpy as np
import sys
from scipy.integrate import solve_ivp


#np.random.seed(0)

N = int(sys.argv[1])
eps = float(sys.argv[2])

v_lower = float(sys.argv[3])

v_higher = float(sys.argv[4])

g = 180*(N-1)

L = 50

sample = np.random.uniform(v_lower-eps,v_higher+eps,g)

## We will use importance sampling to evaluate the expectation, keep enough number of samples to get good approximation


def proposal_q():

	t = 1.0 #(v_higher-v_lower)+(2*eps)
	return ( (1.0/t)**(N-1))


def f(x):

	if min(v_higher,np.min(x)+eps)- max(v_lower,np.max(x)-eps    ) > 0.0:
		val = min(v_higher,np.min(x)+eps)- max(v_lower,np.max(x)-eps    )
	else:
		val = 0.0 

	return val


def f_super_eff(x): # x is a L*N*g matrix, we require output L*g vector

	val = np.minimum(v_higher*np.ones(( L,g )),np.min(x,1)+eps)- np.maximum(v_lower*np.ones(( L,g )),np.max(x,1)-eps    )

	return np.maximum(val,0.0)



def exp_valuation(x):

	val  = 0.5*(   min( np.min(x)+eps,v_higher   )+max(np.max(x)-eps,v_lower  )     )

	return val 


def exp_valuation_super_eff(x): # x is a L*N*g matrix, we require output L*g vector

	val  = 0.5*(   np.minimum( np.min(x,1)+eps,v_higher*np.ones(( L,g ))  )+ np.maximum(np.max(x,1)-eps,v_lower*np.ones(( L,g ))  )     )

	return val



# L is the number of samples required for Monte-Carlo expectation
def gen_sample(L):

	Index_Tensor = np.zeros((L,N-1,len(sample)))

	# First for loop iterates over {\gamma_i}_{i=1}^{K}
	for i in range(len(sample)): 

		gamma_i = sample[i]
		index_matrix_i = np.zeros((L,N-1))


		# Second for loop runs L times where L: number of samples required for estimating integral
		for l in range(L):

			sample_gen = []

			sample_gen.append(gamma_i)

			# Third for loop runs (N-1) times since give \gamma_i we need to generate (N-1) dimesnional vector
			for j in range(N-1):

				# This vector consist of samples that satisfy the appropriate inequality
				xj_range = sample[np.where((sample >= max(v_lower-eps,(np.max(np.array(sample_gen))-(2.0*eps)) )  ) & \
				(sample <=min(v_higher+eps,(np.min(np.array(sample_gen))+(2.0*eps)) )   )   )]

				# This vector consists of indices of the original sample_vector that satisfy above inequality
				index_vec = np.where((sample >= max(v_lower-eps,(np.max(np.array(sample_gen))-(2.0*eps)) )  ) & \
				(sample <=min(v_higher+eps,(np.min(np.array(sample_gen))+(2.0*eps)) )   )   )

				index = random.randint(0,len(xj_range)-1)

				xj = xj_range[ index ]

				sample_gen.append(xj)

				index_matrix_i[l,j] = index_vec[0][index]



		Index_Tensor[:,:,i] = index_matrix_i


	return (Index_Tensor)


index = np.zeros((1,N,1))

index[0,N-1,0] = 1

def wasser_interdependent_value_eff_soln(x):

	x = np.abs( x.reshape((g)) )

	X1_matrix = sample[Index_Tensor.astype(int)]

	B1_matrix = x[Index_Tensor.astype(int)]

	X2_matrix = np.pad(X1_matrix, [(0,0),(0,1),(0,0) ], mode='constant', constant_values= 0)

	B2_matrix = np.pad(B1_matrix, [(0,0),(0,1),(0,0) ], mode='constant', constant_values= 0)

	A_val = np.ones((L,g ))*( sample.reshape((g)))

	X2_matrix = X2_matrix + (index*np.repeat(A_val.reshape((L,1,g)),N,axis=1      )   )

	B_val = np.ones((L,g ))*( x.reshape((g)))

	B2_matrix = B2_matrix + (index*np.repeat(B_val.reshape((L,1,g)),N,axis=1      )   )

	Final_val = ( np.sum(B1_matrix,1)/(  np.sum(B2_matrix,1)**2   ) )* exp_valuation_super_eff(X2_matrix)*\
	( f_super_eff(X2_matrix)/proposal_q()   )*\
	( (1.0/np.mean( f_super_eff(X2_matrix)/proposal_q(), 0 ))*np.ones((L,g)) )

	Final_val = np.mean( Final_val,0 )

	
	return (np.mean( (np.array(Final_val)-1)**2 ) )


def FPSB_sol(x):

	if x>v_lower+eps:

		#Y = np.exp( -(N/(2*eps))*(x - (v_lower+eps)  )   )*( (N/(N+1))*(1.0/(2*eps) )     )

		#Y = np.exp( -(N/(2*eps))*(x - (v_lower+eps)  )   )*( (2*eps)/(N+1)   )

		#val = x-( (2*eps)/N   )+ (Y/N)

		Y = np.exp( -(N/(2*eps))*(x - (v_lower+eps)  )   )*( (2*eps)*(1.0/(N+1) )     )

		val = x-((eps)/1.0  )+(Y/1.0)

	else:

		#val = (v_lower-eps) + (    (N*( x+eps-v_lower      )) /(N+1))

		val = v_lower  +   ( (x-(v_lower)+eps )/(N+1)    )

	return (val)


def F(t,s):

	theta_t = (t+eps-(v_higher))/(2*eps)

	k_t = (1/(2*eps))*   ( (1.0 - (theta_t**(N-1)) )/ (1.0 - (theta_t**(N)) )  )

	val = -(   (N*s*k_t)+( (N-1)-( N*(t+eps)*k_t    )     )     )

	return (val)

t_eval = np.arange(v_higher-eps, v_higher+eps, 0.1)

initial_value = FPSB_sol(v_higher-eps)

sol = solve_ivp(F, [v_higher-eps, v_higher+eps], [initial_value], t_eval=t_eval)


new_grad=grad(wasser_interdependent_value_eff_soln)
Index_Tensor = gen_sample(L)

x0 = np.random.uniform(0.2,0.9,g)  # Intial starting point for the optimization routine
#print(wasser_interdependent_value_soln(x0))
x_star = minimize(wasser_interdependent_value_eff_soln,x0,method='Newton-CG',jac=new_grad)  # We are using an inbuit optimizer to find the optimal bid minimizing squared error
beta_val = np.abs(x_star['x'])
print(x0)
print(x_star)


#np.savetxt('beta_val_N '+str(N)+'.txt',beta_val)
#np.savetxt('sample_val_N '+str(N)+'.txt',sample)

plt.scatter(sample,beta_val,label='Tullock-bids',linewidth=4)
plt.axvline(x=v_lower+eps,color='r', label=r'$ a+\epsilon $',linewidth=4)
plt.axvline(x=v_higher-eps,color='b', label=r'$ b-\epsilon $',linewidth=4)
plt.legend( fontsize = 10)
plt.show()

#### Revenue Calculation Tullock

L = 100
Index_Tensor = gen_sample(L)

Rev_Tullock = []

norm_const = 0.0 

M = 20000

for i in range(M):

	index_1 = random.randint(0,len(sample)-1)

	index_2 = random.randint(0,L-1)

	x_val = np.zeros((N))

	x_val[0] = sample[index_1]

	x_val[1:N] = sample[ (Index_Tensor[index_2,:,index_1]).astype(int) ]

	bid_val = np.zeros((N))

	bid_val[0] = beta_val[index_1]

	bid_val[1:N] = beta_val[ (Index_Tensor[index_2,:,index_1]).astype(int) ]

	density_ratio = f(np.array(x_val))/proposal_q()

	norm_const = norm_const+density_ratio

	Rev_Tullock.append( np.sum(bid_val)*density_ratio)


print('Revenue Tullock', np.mean(np.array(Rev_Tullock)*(M/norm_const)  ) )


#### Revenue calculation FPSB


bid_FPSB = []

for x in sample:
	if x<v_higher-eps:
		bid_FPSB.append(FPSB_sol(x))
	else:
		values = sol.y[0]
		index = 0 
		dist = 10000
		for i in range(len(t_eval)):
			if dist> ( (t_eval[i]-x  )**2):
				index = i 
				dist = (t_eval[i]-x  )**2

		bid_FPSB.append(values[index])

Rev_FPSB = []
bid_FPSB = np.array(bid_FPSB)
norm_const = 0.0 

M = 20000

for i in range(M):
	index_1 = random.randint(0,len(sample)-1)

	index_2 = random.randint(0,L-1)

	x_val = np.zeros((N))

	x_val[0] = sample[index_1]

	x_val[1:N] = sample[ (Index_Tensor[index_2,:,index_1]).astype(int) ]

	bid_val = np.zeros((N))

	bid_val[0] = bid_FPSB[index_1]

	bid_val[1:N] = bid_FPSB[ (Index_Tensor[index_2,:,index_1]).astype(int) ]


	density_ratio = f(np.array(x_val))/proposal_q()

	norm_const = norm_const+density_ratio

	Rev_FPSB.append( np.max(bid_val)*density_ratio)


print('Revenue FPSB', np.mean(np.array(Rev_FPSB)*(M/norm_const)  ) )

#### English Auction


Rev_EA = []
norm_const = 0.0 

M = 20000


def price_EA(x):

	y = x + 0.0 
	index_1 = np.argmax(x)   # Getting the index of max element

	y[index_1] = 0.0 		 # Removing the max element by setting it to zero

	index_2 = np.argmax(y)   

	y[index_1] = y[index_2]  # Storing second max element in place of max element  

	return (y)  # y is essentially the price in english auction



for i in range(M):
	index_1 = random.randint(0,len(sample)-1)

	index_2 = random.randint(0,L-1)

	x_val = np.zeros((N))

	x_val[0] = sample[index_1]

	x_val[1:N] = sample[ (Index_Tensor[index_2,:,index_1]).astype(int) ]

	price_val = exp_valuation( price_EA(x_val))


	density_ratio = f(np.array(x_val))/proposal_q()

	norm_const = norm_const+density_ratio

	Rev_EA.append( price_val*density_ratio)


print('Revenue EA', np.mean(np.array(Rev_EA)*(M/norm_const)  ) )