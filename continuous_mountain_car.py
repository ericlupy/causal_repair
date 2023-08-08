#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@author: Olivier Sigaud

A merge between two sources:

* Adaptation of the MountainCar Environment from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0), adapted by  'Tom Schaul, tom@idsia.ch'
and then modified by Arnaud de Broissia

* the OpenAI/gym MountainCar environment
itself from
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math

import numpy as np
from scipy.optimize import minimize 
from scipy.optimize import Bounds 

import gym
from gym import spaces
from gym.utils import seeding

class Continuous_MountainCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, init_pos=-0.5, init_vel=0):
        # dynamics parameters
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        self.goal_velocity = 0
        self.power = 0.0015 # original power is 0.0015, reduced power is 0.0012, 0.0010 is complete trash and succeeds very rarely; ended up not changing it
        #self.steepness = 0 # SELECTED IN RESET # 0.0035 # originally 0.0025, challenging = 0.0035
        self.steepness_vals = [0.0025] # [0.0025, 0.0035]
        self.steepness_probs = [1] # [0.5, 0.5]
        self.POSNOISESTD = 0.0 # 0.001
        self.VELNOISESTD = 0.0 # 0.0001

        self.low_state = np.array([self.min_position, -self.max_speed])
        self.high_state = np.array([self.max_position, self.max_speed])

        # bounds for drawing true values
        self.TRUECLEFT = 0 # -1  #0.5#-1# -2 -1  # 0.2
        self.TRUECRIGHT = 0 # 1  #self.TRUECLEFT #0#1
        self.TRUEDLEFT = 0 # -0.01  # -0.01 -0.02
        self.TRUEDRIGHT = 0 # 0.02  #self.TRUEDLEFT #0#0.01
        self.INITPOSLEFT = init_pos #-0.6
        self.INITPOSRIGHT = self.INITPOSLEFT #-0.4
        self.INITVELLEFT = init_vel
        self.INITVELRIGHT = self.INITVELLEFT

        # PF hyperparameters 
        self.PARTCT = 4000
        self.PFINITCLEFT = -1
        self.PFINITCRIGHT = 1
        self.PFINITDLEFT = -0.01
        self.PFINITDRIGHT = 0.02
        self.OVERALLWEIGHT = 30 # 50 #100 # overall weight for convergence; need smaller value here for wider initial sampling ranges, tried from 20 to 50, 100
        # used 50 before widening C 
        self.PFVELWEIGHT =  0.1#01#0.01 #0.1 #5 
        # used 0.1 before widening C 

            # adjustment for the magnitude of position-velocity. Used 0.05, 0.1, 0.5, 1, 2, 5, 10
            # semantically, 5 makes sense because the , but then the C estimates converge poorly/arbitrarily
            # in terms of performance, 0.1 works much better 
            # faster convergence when larger number
        #self.DISCOUNT = 0.99#0.01 #0.1 #5 

       # without fake "yellow" results
        self.VERIFASSN3 = lambda c,d,i: \
                -.01 <= d <= -.0075 and ( \
                -.56 <= i <= -.41 and (-1 <= c <= 1) \
                ) or \
                -.0075 <= d <= -.005 and ( \
                -.57 <= i <= -.42 and -1 <= c <= 1 \
                ) or \
                -.005 <= d <= -.0025 and ( \
                -.58 <= i <= -.42 and -1 <= c <= 1 or \
                -.41 <= i <= -.4 and -.2 <= c <= 1 \
                ) or  \
                -.0025 <= d <= 0 and ( \
                -.59 <= i <= -.58 and .2 <= c <= 1 or \
                -.58 <= i <= -.43 and -1 <= c <= 1 or \
                -.41 <= i <= -.4 and -1 <= c <= 1 \
                ) or \
                0 <= d <= .0025 and ( \
                -.58 <= i <= -.57 and -.4 <= c <= 1 or \
                -.57 <= i <= -.43 and -1 <= c <= 1 or \
                -.41 <= i <= -.4 and -1 <= c <= 1 \
                ) or \
                .0025 <= d <= .005 and ( \
                -.57 <= i <= -.56 and .1 <= c <= 1 or \
                -.56 <= i <= -.43 and -1 <= c <= 1 or \
                -.41 <= i <= -.4 and -1 <= c <= 1 \
                ) or \
                .005 <= d <= .0075 and ( \
                -.55 <= i <= -.54 and -.9 <= c <= 1 or \
                -.54 <= i <= -.43 and -1 <= c <= 1 or \
                -.42 <= i <= -.4 and -1 <= c <= 1 \
                ) or \
                .0075 <= d <= .01 and ( \
                -.54 <= i <= -.53 and (.5 <= c <= 1) or \
                -.53 <= i <= -.43 and (-1 <= c <= 1) or \
                -.42 <= i <= -.4 and (-1 <= c <= 1) \
                ) or \
                .01 <= d <= .0125 and ( \
                -.52 <= i <= -.51 and .4 <= c <= 1 or \
                -.51 <= i <= -.43 and -1 <= c <= 1 or \
                -.41 <= i <= -.4 and -1 <= c <= 1 \
                ) or \
                .0125 <= d <= .015 and ( \
                -.5 <= i <= -.49 and .5 <= c <= 1 or \
                -.49 <= i <= -.48 and -.9 <= c <= 1 or \
                -.48 <= i <= -.44 and -1 <= c <= 1 or \
                -.44 <= i <= -.43 and -.6 <= c <= 1 or \
                -.41 <= i <= -.4 and -1 <= c <= 1 \
                ) or \
                .015 <= d <= .0175 and ( \
                -.47 <= i <= -.46 and 0 <= c <= 1 or \
                -.46 <= i <= -.45 and .9 <= c <= 1 or \
                -.45 <= i <= -.44 and -.1 <= c <= 1 or \
                -.41 <= i <= -.4 and -1 <= c <= 1 \
                ) or \
                .0175 <= d <= .02 and ( \
                -.41 <= i <= -.4 and -1 <= c <= 1 \
                ) 

        self.viewer = None

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action,
                                       shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
                                            dtype=np.float32)

        # noise functions
        self.noise_pos = lambda pos, vel : self.truec*vel
        self.noise_vel = lambda pos, vel : self.trued*pos
        self.noise_pos_gen = lambda pos, vel, c: c*vel
        self.noise_vel_gen = lambda pos, vel, d: d*pos

        self.seed() # seed 1 is pretty bad, poor convergence
                    # seed 2 is worse
                    # 5-11 are kinda accurate but late
                    # seed 9 is accurate
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.seed_saved = seed#self.np_random.get_state()[1][0] 
        return [seed]

    def reset(self):
        # randomly pick the true simulation parameters
        self.truec = self.np_random.uniform(low=self.TRUECLEFT, high=self.TRUECRIGHT)
        self.trued = self.np_random.uniform(low=self.TRUEDLEFT, high=self.TRUEDRIGHT)
        self.trueinitpos = self.np_random.uniform(low=self.INITPOSLEFT, high=self.INITPOSRIGHT)
        self.trueinitvel = self.np_random.uniform(low=self.INITVELLEFT, high=self.INITVELRIGHT)
        self.steepness = self.np_random.choice(self.steepness_vals, 1, p=self.steepness_probs) [0] # originally 0.0025, challenging = 0.0035

        # initialize the Gym-related vars
        self.state = np.array([self.trueinitpos, self.trueinitvel])
        self.time = 0

        # Remember the initial observation: 
        self.init_pos_obs = self.state[0] + self.noise_pos(self.state[0], self.state[1])
        self.init_vel_obs = self.state[1] + self.noise_vel(self.state[0], self.state[1])

        # Initialize particles  
        #self.particles_pos = self.np_random.uniform(low=-0.6, high=-0.4, size=self.PARTCT)
        #self.particles_pos_init = self.particles_pos.copy() # save for traceability 
        #self.particles_vel = np.array([0.]*self.PARTCT)
        self.weights = np.array([1./self.PARTCT]*self.PARTCT)
        self.particles_c = self.np_random.uniform(low=self.PFINITCLEFT, high=self.PFINITCRIGHT, size=self.PARTCT)
        self.particles_d = self.np_random.uniform(low=self.PFINITDLEFT, high=self.PFINITDRIGHT, size=self.PARTCT)
        #self.particles_d = self.np_random.uniform(low=self.trued, high=self.trued, size=self.PARTCT)

        # Create initial GTs from initial observations
        self.part_pos_gt, self.part_vel_gt = self.obs_to_true(self.init_pos_obs, self.init_vel_obs, self.particles_c, self.particles_d) 

        #print(self.part_pos_gt)
        #print(self.part_vel_gt)

        # print("Resetting particles and weights, initial observations: ", self.init_pos_obs, self.init_vel_obs)

        # returning the observation
        return np.array([self.init_pos_obs, self.init_vel_obs])

        # returning the true state 
        #return np.array(self.state)

    def step(self, action):
        # print("____________________________________________")
        # print("Step for time =", self.time)
        # print("Steepness:", self.steepness)

        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], -1.0), 1.0)

        #if self.time > 77: 
        #    done = True

        # process noise for the true model
        #proc_noise = [0, 0]   
        proc_noise = self.np_random.normal([0, 0], [self.POSNOISESTD, self.VELNOISESTD]) 
        #print(proc_noise)

        # GT state propagation
        position, velocity = self.model_step(position, velocity, force, proc_noise) 
        self.state = np.array([position, velocity])

        # updating the simulation state 
        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        reward = 0
        if done:
            reward = 100.0
        reward-= math.pow(action[0],2)*0.1

        # new observed values
        # obs_pos = position + self.noise_pos(position[0], velocity[0])
        # obs_vel = velocity + self.noise_vel(position[0], velocity[0])
        obs_pos = position + self.noise_pos(position, velocity)
        obs_vel = velocity + self.noise_vel(position, velocity)
        obs_state = np.array([obs_pos, obs_vel])

        # print("True values:")
        # print(self.state)
        #
        # print("Observed values:")
        # print(obs_state)

        # Propagating particle's GT
        for i in range(self.PARTCT): 
            # get the supposed GT at the first step 
            self.part_pos_gt[i], self.part_vel_gt[i] = self.model_step(np.array([self.part_pos_gt[i]]), np.array([self.part_vel_gt[i]]), force, [0,0])

            # get the expected observations from the new GT 
            pred_pos = self.part_pos_gt[i] + self.noise_pos_gen(self.part_pos_gt[i], self.part_vel_gt[i], self.particles_c[i]) 
            pred_vel = self.part_vel_gt[i] + self.noise_vel_gen(self.part_pos_gt[i], self.part_vel_gt[i], self.particles_d[i]) 
            
            # update weights exponentially to observation deltas
            # L1 norm
            self.weights[i] = self.weights[i]*math.exp( self.OVERALLWEIGHT*(-abs(obs_pos - pred_pos) - self.PFVELWEIGHT*abs(obs_vel - pred_vel)))
            
            # L2 norm
            # self.weights[i] = self.weights[i]*math.exp( self.OVERALLWEIGHT*(-pow(abs(obs_pos - pred_pos),2) - self.PFVELWEIGHT*pow(abs(obs_vel - pred_vel),2)))

            # additive 
            #self.weights[i] = self.DISCOUNT*self.weights[i] + (1-self.DISCOUNT)*math.exp( self.OVERALLWEIGHT*(-abs(obs_pos - pred_pos) - self.PFVELWEIGHT*abs(obs_vel - pred_vel)))
            # L1 norm, one point at a time 
            #self.weights[i] = math.exp( self.OVERALLWEIGHT*(-abs(obs_pos - pred_pos) - self.PFVELWEIGHT*abs(obs_vel - pred_vel)))

            #print("For particle", self.particles_c[i], self.particles_d[i])
            #print("GT estimates", self.part_pos_gt[i], self.part_vel_gt[i]) 
            #print("Obs predictions", pred_pos, pred_vel) 
            #print("Weight:", self.weights[i])

        # normalize weights
        #self.weights = self.weights/np.sum(self.weights)

        # debug print particles over the threshold
        thresh = 0.01
        normweights = self.weights/np.sum(self.weights)
        # print("Weights:", len(normweights[ normweights > thresh ]))
        # print("Weight Values:", normweights[ normweights > thresh ])
        # print("C Values:", self.particles_c[ normweights > thresh ])
        # print("D Values:", self.particles_d[ normweights > thresh ])


        # do not resample particles 
        #assert(np.abs(np.sum(self.weights) - 1) < 1e-6) # make sure there are no precision issues

#        self.particle_log_c.append(self.particles_c)
#        self.particle_log_d.append(self.particles_d)
#        self.weight_log.append(self.weights) 

        self.time += 1 
        #return self.state, reward, done, {}
        return obs_state, reward, done, self.compute_outputs()

    # prints and computes the monitoring outputs
    def compute_outputs(self):
        normweights = self.weights/np.sum(self.weights)
        # compute the means/stds, weighted
        avgc = np.average(self.particles_c, weights=normweights)
        avgd = np.average(self.particles_d, weights=normweights)
        stdc = math.sqrt(np.average((self.particles_c-avgc)**2, weights=normweights))
        stdd = math.sqrt(np.average((self.particles_d-avgd)**2, weights=normweights))

        uncertc = stdc/((self.PFINITCRIGHT-self.PFINITCLEFT) / math.sqrt(12))
        uncertd = stdd/((self.PFINITDRIGHT-self.PFINITDLEFT) / math.sqrt(12))
        # cap the uncertainties 
        uncertc = min(max(uncertc, 0), 1) 
        uncertd = min(max(uncertd, 0), 1) 

        # aggregate the uncertainty
        # uncertagg = uncertc*uncertd  # the "area" covered by the estimates -- too narrow 
        uncertagg1 = (uncertc+uncertd)/2 # arithmetic mean
        uncertagg2 = math.sqrt(uncertc*uncertd) # geometric mean

        #prob_c = self.estimate_part_in_interval(self.particles_c, self.MONINTC) 
        #prob_d = self.estimate_part_in_interval(self.particles_d, self.MONINTD) 
        # get an array of estimated positions, velocities
        poss, vels = self.obs_to_true(self.init_pos_obs, self.init_vel_obs, self.particles_c, self.particles_d) 
        avgi = np.average(poss, weights=normweights)
        #prob_init_pos = self.estimate_part_in_interval(poss, self.MONINTINITPOS) # implicitly uses weights, too
        prob_assn_pre = self.estimate_part_in_assn()

        # adjust for uncertainty preserving the odds
        if prob_assn_pre < 0.99999: # finite odds
            odds = prob_assn_pre/(1-prob_assn_pre)
            prob_assn_post1 = (1-uncertagg1)/(1+1/odds)
        else : # infinite odds
            prob_assn_post1  = 1 - uncertagg1

        if prob_assn_pre < 0.99999: # finite odds
            odds = prob_assn_pre/(1-prob_assn_pre)
            prob_assn_post2 = (1-uncertagg2)/(1+1/odds)
        else : # infinite odds
            prob_assn_post2  = 1 - uncertagg2


        #print(prob_c, prob_d, prob_init_pos)
        #assert(0 <= prob_c <= 1) # TODO DEBUG THIS
        #assert(0 <= prob_d <= 1) # gets triggered for some reason
        #assert(0 <= prob_init_pos <= 1)

        #print("Stepped particles:") 
        #print(self.particles_c) 
        #print(self.particles_d) 
        
        #print("Weights:") 
        #print(self.weights) 

        # print("Mean c:", avgc)
        # print("Mean d:", avgd)
        # print("Mean i:", avgi)
        # print("Stdev c:", stdc)
        # print("Stdev d:", stdd)
        #
        # print("Uncertainty in c:", uncertc)
        # print("Uncertainty in d:", uncertd)
 
        #print("Prob of c in ", self.MONINTC, ": ", prob_c) 
        #print("Prob of d in ", self.MONINTD, ": ", prob_d) 
        #print("Prob of init pos in ", self.MONINTINITPOS, ": ", prob_init_pos) 
        # print("Prob of assn (pre-uncert): ", prob_assn_pre )
        # print("Prob of assn (post-uncert, arithm mean): ", prob_assn_post1 )
        # print("Prob of assn (post-uncert, geom mean): ", prob_assn_post2 )

        assert(0 <= prob_assn_pre <= 1) 

        # this is the "info" variable in the outer loop
        return [avgc, avgd, stdc, stdd, uncertc, uncertd, uncertagg1, uncertagg2, prob_assn_pre, prob_assn_post1, prob_assn_post2]

    # computes a step of the dynamical 
    # no measurement noise here, just the pure dynamics
    # input: ([position], [velocity], [control action], 2-dim process noise) 
    # output: ([new pos], [new vel]) 
    def model_step(self, oldpos, oldvel, act, proc_noise): 

        # updating state values 
        newvel = oldvel + act*self.power - self.steepness * math.cos(3*oldpos) + proc_noise[1]
        if (newvel > self.max_speed): newvel = np.array([self.max_speed])
        if (newvel < -self.max_speed): newvel = np.array([-self.max_speed]) 

        newpos = oldpos + newvel + proc_noise[0]
        if (newpos > self.max_position): newpos = np.array([self.max_position])
        if (newpos < self.min_position): newpos = np.array([self.min_position]) 
        if (newpos == self.min_position and newvel<0): newvel = np.array([0])

        return newpos, newvel

    # computes the supposed true state given the observation and the C/D parameters
    def obs_to_true(self, obspos, obsvel, c, d): 
            gtpos = (c*obsvel - obspos)/(c*d - 1)
            gtvel = (d*obspos - obsvel)/(c*d - 1)
            return gtpos, gtvel

    # estimates the probability of particles in the assumption, using the current weight
    # implicitly uses all the particles and their 
    def estimate_part_in_assn(self): 
        poss, vels = self.obs_to_true(self.init_pos_obs, self.init_vel_obs, self.particles_c, self.particles_d) 
        #res = self.VERIFASSN(self.particles_c, self.particles_d, poss) 
        #print("___________________")
        #print(range(len(vels)))
        idx = [i for i in range(len(vels)) if self.VERIFASSN3(self.particles_c[i], self.particles_d[i], poss[i])]
        #print(idx) 
        #print("___________________")
        return min(max(np.sum(self.weights[idx])/np.sum(self.weights), 0), 1) # cap the probs

    # estimates the probability of particles in an interval, using the current weights 
    # inputs: 
        # list of 1D particles 
        # interval [a, b], where a <= b
    def estimate_part_in_interval(self, particles, interval): 
        idxRight = np.where(particles  >= interval[0])[0]
        idxLeft = np.where(particles <= interval[1])[0]
        idxInside = np.intersect1d(idxRight, idxLeft) 
        return min(max(np.sum(self.weights[idxInside])/np.sum(self.weights), 0), 1) # cap the probs

    # estimate the chance the initial position was in an interval, from C/D particles 
    # inputs: [a, b], where a <= b
    def estimate_init_prob_from_cd(self, interval): 
        poss, vels = self.obs_to_true(self.init_pos, self.init_vel, self.particles_c, self.particles_d) 
        self.estimate_part_in_interval(self, poss, interval) 
 
    # estimate the probability about the initial position based on the current particle weights 
    # inputs: [a, b], where a <= b
    # outputs: float between 0 and 1
    def estimate_init_prob(self, interval):
        #print("Initial particles:", self.particles_pos_init)
        idxRight = np.where(self.particles_pos_init >= interval[0])[0]
        idxLeft = np.where(self.particles_pos_init  <= interval[1])[0]
        idxInside = np.intersect1d(idxRight, idxLeft) 

        #print("IDs of selected particles:", idxInside)
        #idxInside = np.where(self.particles_pos_init >= interval[0] and self.particles_pos_init <= interval[1])[0]
        return np.sum(self.weights[idxInside])/np.sum(self.weights)

    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width
        carwidth=40
        carheight=20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (self.goal_position-self.min_position)*scale
            flagy1 = self._height(self.goal_position)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation((pos-self.min_position)*scale, self._height(pos)*scale)
        self.cartrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
