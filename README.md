# Tullock_CVAS
In this repository we have provided the python code for computing equilibrium bid for Tullock (lottery) in the case of common value affiliated setting (Milgrom and Weber [1982], paper link: https://www.jstor.org/stable/1911865). We have considered the mineral right setting as explained below

f(V) = Uniform[a,b],  f(x_i|V) = Uniform[V-eps,V+eps] i ={1,2,..,N} (see Kagel et al. 1987, paper link: https://www.jstor.org/stable/1913557)

V: Valuation 

x_i: Signal of player i


# Running command
python Tullock_mineral_right_Final.py N eps a b

N: number of players

eps: parameter required for f(x_i|V)

a: lower bound for f(V)

b: upper bound for f(V)

For Ex: python Tullock_mineral_right_Final.py 2 6 25 125



