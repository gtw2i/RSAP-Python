# Graham West
from copy import deepcopy
import sys 
import random
import numpy as np
import math
import pandas as pd
from subprocess import call
from scipy import optimize
from scipy import misc
from matplotlib import pyplot as plt
from matplotlib import image as img

##############
#    MAIN    #
##############

def main():
	
	##############
	# Section 01 #
	##############
	# These vars only affect the output of the program
	nBin     = 55	# number of bins in posterior plot
	nGenP    = nBin	# number of points in function plot
	progress = 0	# print algorithm state at each timestep
	##############
	
	
	##############
	# Section 02 #
	##############
	# These vars control the way RSAP functions
	toMod    = 1		# whether to use RSAP of plain Metropolis
	nStep    = 10000	# number of timesteps to take
	nDim     = 1		# number of parameters to estimate
	initSig  = 4.5		# likelihood sigma, represents scale of error/ease of acceptance
	
	# RSAP scaling/mixing parameters
	if( toMod ):	# RSAP chosen
		modAmp  = [ 0.1, 10.0 ]			# [ min scaling amplitude, max scaling amplitude ]
		modRate = [ 0.3, 0.3, 5000, 2500 ]	# [ min scaling rate, max scaling rate, length of RSAP interval, length of transition interval ]
	else:		# Metropolis chosen
		modAmp  = [ 1.0, 1.0 ]			# these values don't matter since toMod = 0, but they're set to non-adapting values anyway
		modRate = [ 0.0, 0.0, 0.0 ]		# these values don't matter since toMod = 0, but they're set to non-adapting values anyway
	# end
	
	# Proposal/prior distribution properties
	fitParam = range(nDim+1)	# list of parameters to estimate, by default includes all parameters plus initSig
	xLim   = []			# boundaries of the rectangular domain where the prior is non-zero
	pWidth = []			# fixed proposal widths
	for i in range(nDim):
		xLim.append( [-10.0, 10.0] )
		pWidth.append(  1.0 )
	# end
	xLim.append( [0, 100] )		# add initSig to list of parameters to estimate
	pWidth.append( 0.01 )		# add initSig to list of parameters to estimate
	xLim    = np.array(xLim)*1.0
	pWidth  = np.array(pWidth)*1.0
	##############
	
	
	##############
	# Section 03 #
	##############
	# Initialize the chain
	x_start = np.zeros(nDim+1)
	for j in range(nDim):
		x_start[j] = 8.0
	# end
	x_start[-1] = initSig
	##############
	
	
	##############
	# Section 04 #
	##############
	# Run RSAP/Metropolis
	chain, accRate, max_ps, max_lp = metropolis( x_start, nDim+1, nStep, fitParam, pWidth, xLim, progress, toMod,modRate,modAmp )
	##############
	
	
	##############
	# Section 05 #
	##############
	# Plotting
	ind2 = 0
	for ind2 in range(3):
#		plt.cla()
		
		fig, ax = plt.subplots( nrows=int(1), ncols=int(1), figsize=(9,9) )
		
		if( ind2 == 0 ):	# visualize the error function
			y = np.zeros((nDim,nGenP))
			xPlt = np.zeros(nGenP)
			for i in range(nDim):
				x = np.zeros(nDim+1)
				for j in range(nGenP):
					x[i] = xLim[i,0] + (xLim[i,1]-xLim[i,0])*j/(nGenP-1.0)
					y[i,j] = f(x)
					if( i == 0 ):
						xPlt[j] = x[i]
					# end
				# end
			# end
			
			for i in range(nDim):
				ax.plot(xPlt, y[i,:], 'b')
			# end
			ax.set_xlabel("x")
			ax.set_ylabel("error")
		# end
		if( ind2 == 1 ):	# plot the Markov chain trace over time
			ax.plot(chain[:,0], 'r')	# plot 0-th parameter
#			ax.plot(chain[:,-1], 'r')	# plot delta (probably don't want this on the same graph)
			ax.set_ylim(xLim[0])
			ax.set_xlabel("steps")
			ax.set_ylabel("x")
		# end
		if( ind2 == 2 ):	# plot the binned posterior distribution
			param = 0	# model parameter to plot posterior slice against
			burn = 0	# initial steps to ignore in posterior binning
			binC = BinPosterior( nBin, param, xLim, chain[burn:] )
			binC = binC/(1.0*np.sum(binC))
			x = []
			for i in range(nBin):
				x.append( xLim[0,0] + (xLim[0,1]-xLim[0,0])*i/(nBin-1.0) )
			# end
			
			ax.plot(x,binC, 'b')	# plot posterior estimate
			
			meanSig = np.mean(chain[:,-1])
			sig     = meanSig
			y       = np.zeros((nDim,nGenP))
			xPlt    = np.zeros(nGenP)
			for i in range(nDim):
				x = np.zeros(nDim+1)
				for j in range(nGenP):
					x[i] = xLim[i,0] + (xLim[i,1]-xLim[i,0])*j/(nGenP-1.0)
					y[i,j] = np.exp( -0.5*(f(x)/sig)**2)/(sig*np.sqrt(2*np.pi))	# calculate true posterior (comment out this step if your model is expensive)
					if( i == 0 ):
						xPlt[j] = x[i]
					# end
				# end
			# end
			
			y_norm = y[param,:]/np.sum( y[param,:] )
			ax.plot(xPlt, y_norm, 'r')	# plot true posterior (comment this out together with the above line)
			
			ax.set_xlabel("x")
			ax.set_ylabel("posterior distribution")
		# end
	# end
	
#	plt.tight_layout(w_pad=0.0, h_pad=0.0)
	plt.show()
	##############
	
# end

# This function contains the code for RSAP/Metropolis
def metropolis( start, nPar, nStep, fitParam, pWidth, xLim, progress, toMod, modRate, modAmp ):
	
	# counter fo acceptance rate
	n_acc = 0.0	
	
	# generate uniform samples for acceptance testing
	rTest = np.random.uniform(low=0, high=1, size=nStep)
	
	# generate non-adapted jumps sampled from a normal distribution (will be scaled later)
	cov  = np.diag(pWidth**2)
	zero = np.zeros(nPar)
	jumps = np.random.multivariate_normal( mean=zero, cov=cov, size=nStep )
	
	# set jumps for parameters which aren't being estimated to exactly zero
	for i in range(nStep):
		for j in fitParam:
			if( pWidth[j] == 0.0 ):
				jumps[i][j] = 0.0
	# end
	
	# initialize mixing probs
	modProb = np.array( [1.0/3.0, 1.0/3.0, 1.0/3.0] )
	
	# initialize vars which store the best current best solution
	max_ps = start
	max_lp = -np.inf
	
	# create chain
	chain = np.array([start,])
	cur = chain[-1]
	cur_lp, cur_err = log_posterior( start, xLim )
	
	# initialize counters
	isAcc = 1			# 1 if previous candidate was accepted
	rejects  = 0			# number of consecutive rejections
	degWide = np.zeros(nPar)	# widening degrees of adaptation
	degThin = np.zeros(nPar)	# thinning degrees of adaptation
	
	# RSAP/Metropolis
	for step in range(nStep):
		
		# print algorithm state	
		if( progress ):
			print "step: ", step
		# end
		
		# Get current position of chain
		cur = chain[-1]
		
		# get candidate state
		if( rejects == 0 or toMod == 0 ):	# Metropolis step
			cand = cur + jumps[step]
		else:					# RSAP step
			# calculate mixing probabilities
			modProb[2] = adapt2( step, modRate[2], modRate[3], 1.0/3.0, 1.0 )
			modProb[0] = 0.5*(1-modProb[2])
			modProb[1] = modProb[0]
				
			# perturb actual parameters
			for i in fitParam:
				# mixing test
				rAdapt = np.random.uniform(0,1)
				# scaling value
				mod = 1.0
				
				if( rAdapt <= modProb[0] ):			# THIN................
					degThin[i] += 1
					mod = adapt1( degThin[i], 1, modAmp[0], modRate[0] )
					jumps[step][i] *= mod
					cand[i] = cur[i] + jumps[step][i]
				elif( rAdapt <= modProb[0] + modProb[1] ):	# WIDE................
					degWide[i] += 1
					mod = adapt1( degWide[i], 1, modAmp[1], modRate[1] )
					jumps[step][i] *= mod
					cand[i] = cur[i] + jumps[step][i]
				else:						# FIXED...............
					cand[i] = cur[i] + jumps[step][i]
				# end
			# end
			# perturb initSig
			cand[-1] = cur[-1] + jumps[step][-1]
		# end
		
		# calculate log-post, acc. prob.
		cand_lp, cand_err = log_posterior(cand, xLim )
		acc_prob = np.exp(cand_lp - cur_lp)
		
		# determine acceptance/rejection
		if( rTest[step] <= acc_prob ):	# ACCEPTANCE..............
			isAcc   = 1
			rejects = 0
			n_acc  += 1
			cur     = cand
			cur_lp  = cand_lp
			cur_err = cand_err
			degWide = np.zeros(nPar)
			degThin = np.zeros(nPar)
			chain   = np.append(chain, [cur,], axis=0)
			
			# print algorithm state
			if( progress ):
				print "err:\t" + str(cur_err)
				print "accepted\n"
			# end
		else:				# REJECTION................
			isAcc    = 0
			rejects += 1
			chain    = np.append(chain, [cur,], axis=0)
			
			# print algorithm state
			if( progress ):
				print "err:\t" + str(cand_err)
				print "rejected\n"
			# end
		# end
		
		# store optimal solution
		if( cur_lp < max_lp ):
			max_lp = cur_lp
			max_ps = cur
		# end
	# end
	
	# calculate acc. rate
	acc_rate = (1.0*n_acc/nStep)
		
	return [ chain, acc_rate, max_ps, max_lp ]

# end

# objective function
def f( x ):
	
	nPar = len(x)-1
	
#	np.random.seed(int(100*np.abs(x[0])+1000000*np.abs(x[1])))
	
	# Ackley
	A = 20.0
	B = 4.0
	q1 = 0
	q2 = 0
	for i in range(nPar):
		q1 += x[i]**2
		q2 += np.cos(2*np.pi*x[i])
	# end
	y = A*( 1.0 - np.exp(-0.2*(q1/(1.0*nPar))**0.5) ) + B*( np.e - np.exp(q2/(1.0*nPar)) ) + 10**-15
	
	return y
	
# end

# calculate log-likelihood
def log_likelihood( x ):
	
	sig = x[-1]
	error = f(x)
	ll = -( np.log(2*np.pi*sig**2)/2 + error**2/(2*sig**2) )
	
	return ll, error
	
# end

# uniform prior
def log_prior( x, xLim ):
	
	inRange = 1
	for i in range(len(x)):
		if( not ( xLim[i,0] <= x[i] <= xLim[i,1] ) ):
			inRange = 0
		# end
	# end
	
	if( inRange ):
		return 0
	else:
		return -np.inf
	# end

# end

# calculate log-posterior
def log_posterior( x, xLim ):
	
	pri = log_prior(x, xLim)
	like, error = log_likelihood( x )
	
	if np.isfinite(pri):
		return pri + like, error
	else:
		return -np.inf, error
	# end
	
# end

# width-scoaing function
def adapt1( t, a, b, r ):
	return a - (a - b)*(1.0 - np.exp(-r*t))
# end

# mixing probabilities
def adapt2( t, t1, tw, a, b ):
	if( t < t1 ):
		y = a
	elif( t1 <= t and t < t1 + tw ):
#		y = a + a*( 1.0 - np.cos( np.pi*(t-t1)/tw ) )
		y = 2*a - a*np.cos( np.pi*(t-t1)/tw )
	else:
		y = b
	# end
	return y
# end

# bins one parameter
def BinPosterior(nBin, ind, xLim, chain):
	
	chain = chain[:,ind]
	
	binCnt = np.zeros(nBin)
	
	xmin = xLim[ind][0]
	xmax = xLim[ind][1]
	
	for i in range(len(chain)):
		x = float(chain[i])
		
		ii = int( (x - xmin) / (xmax - xmin) * nBin )
		
		if( ii > 0 and ii < nBin ):
		        binCnt[ii] = binCnt[ii] + 1
		# end
	# end
		
	return binCnt

# end


###########
# Execute #
###########
main()


