########## For Counterfactual #################

## GAME PARAMETERS AND ACTIONS
# Payoff Parameter
param alpha;

# Probability Parameter
param m = 1;

# Action Parameters
param A1H;
param A2H;
param A1L;
param A2L;

# Probabilities
param prob1HH = m*exp(alpha*A1H)/(1+exp(alpha*A1H)+exp(alpha*A2H));
param prob1HL = m*exp(alpha*A1H)/(1+exp(alpha*A1H)+exp(alpha*A2L));
param prob1LH = m*exp(alpha*A1L)/(1+exp(alpha*A1L)+exp(alpha*A2H));
param prob1LL = m*exp(alpha*A1L)/(1+exp(alpha*A1L)+exp(alpha*A2L));
param prob2HH = m*exp(alpha*A2H)/(1+exp(alpha*A1H)+exp(alpha*A2H));
param prob2HL = m*exp(alpha*A2L)/(1+exp(alpha*A1H)+exp(alpha*A2L));
param prob2LH = m*exp(alpha*A2H)/(1+exp(alpha*A1L)+exp(alpha*A2H));
param prob2LL = m*exp(alpha*A2L)/(1+exp(alpha*A1L)+exp(alpha*A2L));

## PARAMETERS:
# 1 Actions
set A:={A1L,A1H};

# 2 Epsilon Shocks
param nR;
set R:=1..nR;
set Rcube:=1..nR^2;

# 3 "direction" of optimization
param lambda {i in 1..4};

# 6 Payoff Shocks grid
param Egrid {r in R};

# 7 Probability Mass of Payoff Shocks
param Mass {r in Rcube};

# 8 Epsilon

param epsilon;
param d {r in R, a in 1..4};

## VARIABLES:
# bce is characterized by actions, and INDEX of eps shock for every i=1,2,3
var bce{a1 in A, a2 in A, r1 in R, r2 in R} >=0, <=1;

# Prob of all outcomes
var PHH = sum{ r1 in R, r2 in R} bce[A1H,A2H,r1,r2];
var PHL = sum{ r1 in R, r2 in R} bce[A1H,A2L,r1,r2];
var PLH = sum{ r1 in R, r2 in R} bce[A1L,A2H,r1,r2];
var PLL = sum{ r1 in R, r2 in R} bce[A1L,A2L,r1,r2];

## THE PROBLEM:

# playing with the lambdas we can get all bounds
maximize Obj: lambda[1]*PHH + lambda[2]*PHL + lambda[3]*PLH + lambda[4]*PLL;

subject to

Sm1 : sum {a1 in A, a2 in A, r1 in R, r2 in R} bce[a1,a2,r1,r2] = 1 ;
Consist {r1 in R,r2 in R}: sum {a1 in A, a2 in A}  bce[a1,a2,r1,r2] =  Mass[(r1-1)*nR+r2];

# Pricing Game Payoff: Choice_Prob*(Price-Marginal Cost)
IC1_H { r1 in R } : (A1H-Egrid[r1])*(sum{r2 in R} (bce[A1H,A2L,r1,r2]*prob1HL + bce[A1H,A2H,r1,r2]*prob1HH)) - (A1L-Egrid[r1])*(sum{r2 in R} (bce[A1H,A2L,r1,r2]*prob1LL + bce[A1H,A2H,r1,r2]*prob1LH)) >= -epsilon * d[r1,2]; #IC constraint for player 1 when mediator says H
IC1_L { r1 in R } : (A1L-Egrid[r1])*(sum{r2 in R} (bce[A1L,A2L,r1,r2]*prob1LL + bce[A1L,A2H,r1,r2]*prob1LH)) - (A1H-Egrid[r1])*(sum{r2 in R} (bce[A1L,A2L,r1,r2]*prob1HL + bce[A1L,A2H,r1,r2]*prob1HH)) >= -epsilon * d[r1,1]; #IC constraint for player 1 when mediator says L
IC2_H { r2 in R } : (A2H-Egrid[r2])*(sum{r1 in R} (bce[A1L,A2H,r1,r2]*prob2LH + bce[A1H,A2H,r1,r2]*prob2HH)) - (A2L-Egrid[r2])*(sum{r1 in R} (bce[A1L,A2H,r1,r2]*prob2LL + bce[A1H,A2H,r1,r2]*prob2HL)) >= -epsilon * d[r2,4]; #IC constraint for player 2 when mediator says H
IC2_L { r2 in R } : (A2L-Egrid[r2])*(sum{r1 in R} (bce[A1L,A2L,r1,r2]*prob2LL + bce[A1H,A2L,r1,r2]*prob2HL)) - (A2H-Egrid[r2])*(sum{r1 in R} (bce[A1L,A2L,r1,r2]*prob2LH + bce[A1H,A2L,r1,r2]*prob2HH)) >= -epsilon * d[r2,3]; #IC constraint for player 2 when mediator says L

####   DEFINE THE PROBLEM   #####
# Name the problem  
problem Polytope_Pricing: 

# Choose the objective function
Obj,

# List the variables
bce, PHH, PHL, PLH, PLL,  # 

# List the constraints 
Sm1, Consist, IC1_H, IC1_L, IC2_H, IC2_L;
