#--------------------------------- ALL
import h5py 
import uproot
import time
import pandas as pd
import numpy as np
import os
#---------------------------------------------- JET IMAGE
from pyjet import cluster
from pyjet.testdata import get_event
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import glob
import matplotlib.cm as cm
from scipy.stats import binned_statistic_2d
from matplotlib.colors import LogNorm

#-------------------------------- JSS
import pyjet
import argparse
import glob
import math
import operator
import re
import time
from functools import lru_cache, reduce
from itertools import combinations




def flatten(event): #my update
    fp = np.concatenate((np.expand_dims(event[0], axis=-1),
                         np.expand_dims(event[1], axis=-1),
                         np.expand_dims(event[2], axis=-1),
                         np.expand_dims(event[3], axis=-1)), axis=-1)
    fp = fp.transpose((1,0))
    fp = np.core.records.fromarrays( [fp[:][0],fp[:][1],fp[:][2],fp[:][3]], names= 'pT, eta, phi, mass' , formats = 'f8, f8, f8,f8')

    return fp
## Taylor's root to jet image code
def root_2_jets(root_file):
    # Takes root_file as produces a jet-image
#     events = root2array(root_file, "Delphes;1", branches=[
#         "Tower.ET", "Tower.Eta", "Tower.Phi", "Tower.E"])

    file = uproot.open(root_file)
    events = np.array([np.array(file["Delphes;1"]["Tower.ET"].array()), #assum E>>m
                       np.array(file["Delphes;1"]["Tower.Eta"].array()),
                       np.array(file["Delphes;1"]["Tower.Phi"].array()),
                       np.array(file["Delphes;1"]["Tower.E"].array())*0   #assume m<<1
                      ])
    events = np.expand_dims(events, axis=-1)
    events = events.transpose((1,0,2))
    events = np.squeeze(events,axis=(2,))

    jet_images = []
    for ix, event in enumerate(tqdm(events)):
        # create trimmed jet event (also centered and rotated)
        event = jet_trimmer(event=event, R0=1.2, R1=0.2, pt_cut=0.03) #paper setting
        

        # pixelize the trimmed jet
        jet_image = pixelize(event)

        # include jet-image as long as it isn't blank
        # blank jets occur only if the event fails a cut
        # during trimming
        if np.sum(jet_image) != 0:
            jet_images.append(jet_image)
            
    return jet_images
def JSS(root_file): #jet substructure
    file = uproot.open(root_file)
    events = np.array([np.array(file["Delphes;1"]["Tower.ET"].array()), #assum E>>m
                       np.array(file["Delphes;1"]["Tower.Eta"].array()),
                       np.array(file["Delphes;1"]["Tower.Phi"].array()),
                       np.array(file["Delphes;1"]["Tower.E"].array())*0  #assume m<<1
                      ])
    events = np.expand_dims(events, axis=-1)
    events = events.transpose((1,0,2))
    events = np.squeeze(events,axis=(2,))

    jss = []
    for ix, event in enumerate(tqdm(events)):
        # create trimmed jet event (also centered and rotated)
        event = jet_trimmer(event=event, R0=0.8, R1=0.4, pt_cut=0.03) #paper setting
        jss.append(event)
            
    return jss

def struc2arr(x):
    # pyjet outputs a structured array. This converts
    # the 4 component structured array into a simple
    # 4xN numpy array
    x = x.view((float, len(x.dtype.names)))
    return x


def rotate(x, y, a):
    xp = x * np.cos(a) - y * np.sin(a)
    yp = x * np.sin(a) + y * np.cos(a)
    return xp, yp


def jet_trimmer(event, R0, R1, pt_cut):
    # R0 = Clustering radius for the main jets
    # R1 = Clustering radius for the subjets in the primary jet
    # pt_cut = Threshold for subjets (relative to the primary jet it's a subjet of)

    trim_pt, trim_eta, trim_phi, trim_mass = [], [], [], []
    
    flattened_event = flatten(event)

#     flattened_event = stretch(event.reshape(-1))
    sequence = cluster(flattened_event, R=R0, p=-1)

    # Main jets
    jets = sequence.inclusive_jets()
#     print("check")
    # In case we are missing a leading jet, break early
    if len(jets) == 0:
        return np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)

    # Take just the leading jet
    jet0 = jets[0]

    # Define a cut threshold that the subjets have to meet (i.e. 5% of the original jet pT)
    jet0_max = jet0.pt
    jet0_cut = jet0_max*pt_cut

    # Grab the subjets by clustering with R1
    subjets = cluster(jet0.constituents_array(), R=R1, p=1)
    subjet_array = subjets.inclusive_jets()

    # Require at least 2 subjets. Otherwise, break early
    if len(subjet_array) <= 1:
        return np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)

    for subjet in subjet_array:
        if subjet.pt < jet0_cut:
            # subjet doesn't meet the percentage cut on the original jet pT
            pass
        else:
            # Get the subjets pt, eta, phi constituents
            subjet_data = subjet.constituents_array()
            subjet_data = struc2arr(subjet_data)
            pT = subjet_data[:, 0]
            eta = subjet_data[:, 1]
            phi = subjet_data[:, 2]
            mass = subjet_data[:, 3]

            # Shift all data such that the leading subjet
            # is located at (eta,phi) = (0,0)
            eta -= subjet_array[0].eta
#             eta -= jet0.eta
            phi = np.array( [deltaPhi(i,subjet_array[0].phi) for i in phi])
#             phi = np.array( [deltaPhi(i,jet0.phi) for i in phi])
            

            # Rotate the jet image such that the second leading
            # jet is located at -pi/2
#             s1x, s1y = subjet_array[1].eta, subjet_array[1].phi
            s1x, s1y = subjet_array[1].eta - subjet_array[0].eta, deltaPhi(subjet_array[1].phi,subjet_array[0].phi)
#             s1x, s1y = subjet_array[1].eta - jet0.eta, deltaPhi(subjet_array[1].phi, jet0.phi)
            

            theta = np.arctan2(s1y, s1x)
            if theta < 0.0:
                theta += 2 * np.pi
            eta, phi = rotate(eta, phi, np.pi - theta)

            # Collect the trimmed subjet constituents
            trim_pt.append(pT)
            trim_eta.append(eta)
            trim_phi.append(phi)
            trim_mass.append(mass)
    return np.concatenate(trim_pt), np.concatenate(trim_eta), np.concatenate(trim_phi), np.concatenate(trim_mass)


def pixelize(event):
    pt, eta, phi = event[0], event[1], event[2]

    # Define the binning for the complete calorimeter
#     bins = np.arange(-1.2, 1.2, 0.1)
    bins = 64
    ranges = np.array([[-1.2,1.2],[-1.2,1.2]])

    # Sum energy deposits in each bin
    digitized = binned_statistic_2d(eta, phi, pt, statistic="sum", bins=bins, range= ranges)
#     digitized = binned_statistic_2d(eta, phi, pt, statistic="sum", bins=bins)
    
    jet_image = digitized.statistic
    return jet_image
#-----------------------------------------------------------------------------------------------------------
def find_decayratio(event,n, mode = 'daughter'):
    d1, d2 = -1, -1
    d0 = -1
    ID = 4900101 #Xd PID
    for j in  range(len(event[n][0])):
        if(event[n][5][j]==ID)&(event[n][0][j]==23): #23 : hardest outgoing particles
            d1 = event[n][_D1][j]
            d2 = event[n][_D2][j]
            while((event[n][_PID][d1]==ID)or(event[n][5][d2]==ID) ):
                if(event[n][_PID][d1]==ID):
                    d0 = d1
                    d1 = event[n][_D1][d0]
                    d2 = event[n][_D2][d0]
                    if(event[n][_PID][d2]==ID):
                        d0 = d2
                        d1 = event[n][_D1][d0]
                        d2 = event[n][_D2][d0]
                        
    if mode=='daughter':
        return d1, d2
    if mode=='id':
        return event[n][5][d1], event[n][5][d2]
    
def find_subdecayratio(event, n, j, ID=4900111): 
    d1, d2 = -1, -1
    d0 = -1
    d1 = event[n][_D1][j]
    d2 = event[n][_D2][j]
    if((abs(event[n][_PID][d1])!=ID)&(abs(event[n][_PID][d2])!=ID)):
        d1=0
        d2=0
        return d1, d2
    else:
        while((abs(event[n][_PID][d1])==ID)or(abs(event[n][_PID][d2])==ID) ):
            if(abs(event[n][_PID][d1])==ID):
                d0 = d1
                d1 = event[n][_D1][d0]
                d2 = event[n][_D2][d0]
            if(abs(event[n][_PID][d2])==ID):
                d0 = d2
                d1 = event[n][_D1][d0]
                d2 = event[n][_D2][d0]
        return event[n][_PID][d1], event[n][_PID][d2]
def find_daughterid(event, n, j, ID=4900111):
    sid = event[n][_PID][j]
    if(sid!=ID):
        return 0, 0
    else:
        d1 = event[n][_D1][j]
        d2 = event[n][_D2][j]
        return event[n][_PID][d1], event[n][_PID][d2]

def deltaPhi(phi1,phi2):
    x = phi1-phi2
    while x>= np.pi: x -= np.pi*2.
    while x< -np.pi: x += np.pi*2.
    return x        
#----------------------------------------------------------------------------------------------------------
def jet_trimmer_1J(event, R0, R1, pt_cut):
    # R0 = Clustering radius for the main jets
    # R1 = Clustering radius for the subjets in the primary jet
    # pt_cut = Threshold for subjets (relative to the primary jet it's a subjet of)    
    flattened_event = flatten(event)
    sequence = cluster(flattened_event, R=R0, p=-1)
    # Main jets
    jets = sequence.inclusive_jets()
    # In case we are missing a leading jet, break early
    if len(jets) == 0:
        return jets
    

    # Take just the leading jet
    jet0 = jets[0]

    # Define a cut threshold that the subjets have to meet (i.e. 5% of the original jet pT)
    jet0_max = jet0.pt
    jet0_cut = jet0_max*pt_cut

    # Grab the subjets by clustering with R1
    subjets = cluster(jet0.constituents_array(), R=R1, p=1)
    subjet_array = subjets.inclusive_jets()
    j0 = []
    if (subjet_array[0].pt >= jet0_cut):
#         j0 = subjet_array[0].constituents_array()
        for ij, subjet in enumerate(subjet_array):
#             if (ij == 0):
#                 continue
            if subjet.pt < jet0_cut:
                # subjet doesn't meet the percentage cut on the original jet pT
                continue
            if subjet.pt >= jet0_cut:
                # Get the subjets pt, eta, phi constituents
                subjet_data = subjet.constituents_array()
                j0.append(subjet_data)
#                 j0 = np.append(j0, subjet_data)
    else:
        j0 = subjet_array[0].constituents_array()*0
    jet = j0[0]
    for i, subjet in enumerate(j0):
        if i==0 :
            continue
        jet = np.append(jet, subjet)
        
    sequence = cluster(jet, R=R0, p=-1)
    jet = sequence.inclusive_jets()
    return jet

def jet_clustering(event, R0):
    # R0 = Clustering radius for the main jets
    # R1 = Clustering radius for the subjets in the primary jet
    # pt_cut = Threshold for subjets (relative to the primary jet it's a subjet of)    
#     flattened_event = flatten(event)
    sequence = cluster(flattened_event, R=R0, p=-1)
    # Main jets
    jets = sequence.inclusive_jets(ptmin=300)
    # In case we are missing a leading jet, break early
    if len(jets) == 0:
        return jets

    # Take just the leading jet
#     jet0 = jets[0]

    return jet0
#--------------------------------------------------------------------------------------------------
def angle(jet, particles):
    ptot2 = (jet.px**2 + jet.py**2 + jet.pz**2) * \
        (particles['px']**2 + particles['py']**2 + particles['pz']**2)
    arg = (jet.px * particles['px'] + jet.py *
           particles['py'] + jet.pz * particles['pz']) / ptot2**(1/2)
    arg[np.isnan(arg)] = 1.0
    arg[arg > 1.0] = 1.0
    arg[arg < -1.0] = -1.0
    return np.arccos(arg)
def t0(jet):
    return sum(p.pt * CalcDeltaR(p, jet) for p in jet.constituents())
def tn(jet, n): #t1 t2 t3 t21 t32
    assert n >= 0
    if n == 0:
        return t0(jet)
    particles = jet.constituents_array()
    if len(particles) < n:
        return -1
    subjets = pyjet.cluster(particles, R=1.0, p=1).exclusive_jets(n)
    subjets_array = [subjet.constituents_array() for subjet in subjets]
    wta_axes = [a[np.argmax(a['pT'])] for a in subjets_array]
    wta_axes = np.array(wta_axes, dtype=subjets_array[0].dtype)
    return np.sum(particles['pT']*CalcDeltaRArray(particles, wta_axes).min(axis=0)) / t0(jet)
def CalcDeltaRArray(p, a):
    dEta = p['eta'] - \
        a['eta'].repeat(p.shape[0]).reshape(a.shape[0], p.shape[0])
    dPhi = np.abs(p['phi'] - a['phi'].repeat(p.shape[0]
                                             ).reshape(a.shape[0], p.shape[0]))
    mask = dPhi > np.pi
    dPhi[mask] *= -1
    dPhi[mask] += 2 * np.pi
    return (dPhi**2 + dEta**2)**0.5


def CalcDeltaR(j1, j2):
    eta1 = j1.eta
    phi1 = j1.phi
    eta2 = j2.eta
    phi2 = j2.phi

    dEta = eta1-eta2
    dPhi = abs(phi1-phi2)
    if dPhi > np.pi:
        dPhi = 2*np.pi - dPhi

    dR = (dPhi**2 + dEta**2)**0.5

    return dR

# energy correlators
# https://arxiv.org/pdf/1411.0665.pdf


def CalcEECorr(jet, n=1, beta=1.0):

    assert n == 2 or n == 3, 'fn must be in [2, 3] but is n'

    jet_particles = jet.constituents()

    if len(jet_particles) < n:
        return -1

    currentSum = 0

    if n == 2:
        for p1, p2 in combinations(jet_particles, 2):
            # get the terms of the triplet at hand
            pt1 = p1.pt
            pt2 = p2.pt
            dr12 = CalcDeltaR(p1, p2)

            # calculate the partial contribution
            thisterm = pt1*pt2 * (dr12)**beta

            # sum it up
            currentSum += thisterm

        eec = currentSum/(jet.pt)**2

    elif n == 3:
        dr = {(p1, p2): CalcDeltaR(p1, p2)
              for p1, p2 in combinations(jet_particles, 2)}
        for p1, p2, p3 in combinations(jet_particles, 3):
            # get the terms of the triplet at hand
            dr12 = dr[(p1, p2)]
            dr13 = dr[(p1, p3)]
            dr23 = dr[(p2, p3)]

            # calculate the partial contribution
            thisterm = p1.pt*p2.pt*p3.pt * (dr12*dr13*dr23)**beta

            # sum it up
            currentSum += thisterm

        eec = currentSum/(jet.pt)**3
    return eec

def calc_angularity(jet):
    jet_particles = jet.constituents_array(ep=True)

    if jet_particles.shape[0] == 0:
        return -1
    if jet.mass < 1.e-20:
        return -1

    theta = angle(jet, jet_particles)
    e_theta = jet_particles['E'] * np.sin(theta)**-2 * (1 - np.cos(theta))**3

    return np.sum(e_theta) / jet.mass

def calc_KtDeltaR(jet):
    particles = jet.constituents_array()
    if particles.shape[0] < 2:
        return 0.0

    subjets = pyjet.cluster(particles, R=0.4, p=1).exclusive_jets(2)

    return CalcDeltaR(subjets[0], subjets[1])

#---------------------my im-----------------------------------
def myrotate(x, y, a):
    xp = x * np.cos(a) + y * np.sin(a)
    yp = y * np.cos(a) - x * np.sin(a)
    return xp, yp
def root_2_jetim(root_file):
    # Takes root_file as produces a jet-image
#     events = root2array(root_file, "Delphes;1", branches=[
#         "Tower.ET", "Tower.Eta", "Tower.Phi", "Tower.E"])

    file = uproot.open(root_file)
    events = np.array([np.array(file["Delphes;1"]["Tower.ET"].array()), #assum E>>m
                       np.array(file["Delphes;1"]["Tower.Eta"].array()),
                       np.array(file["Delphes;1"]["Tower.Phi"].array()),
                       np.array(file["Delphes;1"]["Tower.E"].array())*0   #assume m<<1
                      ])
    events = np.expand_dims(events, axis=-1)
    events = events.transpose((1,0,2))
    events = np.squeeze(events,axis=(2,))

    jet_images = []
    for ix, event in enumerate(tqdm(events)):
        # create trimmed jet event (also centered and rotated)
        event = event2image(event=event, R0=1.2, R1=0.2, pt_cut=0.03) #paper setting
        

        # pixelize the trimmed jet
        jet_image = pixelize(event)

        # include jet-image as long as it isn't blank
        # blank jets occur only if the event fails a cut
        # during trimming
        if np.sum(jet_image) != 0:
            jet_images.append(jet_image)
            
    return jet_images
def event2image(event, R0, R1, pt_cut,ptmin=300,ptmax=400):
    # R0 = Clustering radius for the main jets
    # R1 = Clustering radius for the subjets in the primary jet
    # pt_cut = Threshold for subjets (relative to the primary jet it's a subjet of)

    trijet = jet_trimmer_1J(event, R0, R1, pt_cut)
    if (len(trijet) < 1):
        return np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
    # Take just the leading jet
    jet0 = trijet[0]
    jc = jet0.constituents_array()
    if (jet0.pt < ptmin)or(jet0.pt>ptmax):
        return np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)

    theta = np.arctan(get_tan(jet0))
#     if theta < 0.0:
#         theta += 2*np.pi
#     eta, phi = rotate(jc['eta'], jc['phi'], np.pi - theta)
    eta, phi = myrotate(jc['eta']-jet0.eta, deltaPhi_np(jc['phi'],jet0.phi), theta)

    # Collect the trimmed subjet constituents
#     trim_pt.append(pT)
#     trim_eta.append(eta)
#     trim_phi.append(phi)
#     trim_mass.append(mass)
    return [jc['pT'], eta, phi, jc['mass']]
def get_tan(jet): ## input one jet
    jc = jet.constituents_array() #leading jet
    jp = p_pteta(jc['pT'],jc['eta'])
    jE = (jp**2+jc['mass']**2)**0.5
    delphi = deltaPhi_np(jc['phi'],jet.phi)
    deleta = jc['eta']-jet.eta
    delR = (delphi**2+deleta**2)**0.5
    if sum(delR) ==0:
        return 0
    sumx = np.sum(np.select([delR>0],[deleta*jE/delR]) )
    sumy = np.sum(np.select([delR>0],[delphi*jE/delR]) )
#     sumx = np.sum(deleta*jE/delR )
#     sumy = np.sum(delphi*jE/delR )


#     if sumx == 0:
#         print("smx ==0")
#         if sumy == 0:
#             print("all==0")
#             return 0
#         else:
#             return np.sign(sumy)*np.Inf
    
#     if sumx!=0 :
    return sumy/sumx
    
def p_pteta(pt,eta): #np.array
    return pt*np.cosh(eta)

def deltaPhi_np(phi1,phi2):
    x = phi1-phi2
    x = x - (x>=np.pi).astype(np.int)*np.pi*2
    x = x + (x<-np.pi).astype(np.int)*np.pi*2
    return x  

#=======================================================================================================
import sys
nevent = int(sys.argv[1]) # N of events
inputfile = sys.argv[2] # input root file
outpath = sys.argv[3] # output .h5 file path

#---------------------------Jet image---------------------------------------
outfile = outpath + 'jetim.h5' # output jetim .h5 file 
jet_im = root_2_jets(inputfile)
hf = h5py.File(outfile, 'w')
hf.create_dataset('jetim', data=jet_im)
hf.close()

#---------------------------JSS---------------------------------------
def JSS_V(events): ##take too long time (about 10 min need to speed up)
#     T1, T2, T3, T21, T22, T32, EE2, EE3, D2, ANGU, KTDEL, PT, ETA, PHI, MASS = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    T21, MASS, D21, D22, C21, C22 = [], [], [], [], [], []
    for event in tqdm(events):
        
        
        found_jet = jet_trimmer_1J(event=event, R0=1.2, R1=0.2, pt_cut=0.03)
        if len(found_jet)==0: 
            continue
        
        found_jet = found_jet[0]
#         if (found_jet.pt> 400)or(found_jet.pt<300): # if  I add this condition, I don't have enough jet, since SVJ has invisible meson in jet
#             continue
        if (found_jet.pt> 400)or(found_jet.pt<300): # so I increase upper bound to XXX GeV
            continue
            
#         origin_jet = jet_clustering(event, 1.2)
#         if len(origin_jet)==0: 
#             continue
#         mass = origin_jet.mass - found_jet.mass
        t1 = tn(found_jet, n=1)
        t2 = tn(found_jet, n=2)
#         t3 = tn(found_jet, n=3)
        t21 = t2 / t1 if t1 > 0.0 else 0.0
#         t32 = t3 / t2 if t2 > 0.0 else 0.0
        ee2 = CalcEECorr(found_jet, n=2, beta=1.0)
        ee3 = CalcEECorr(found_jet, n=3, beta=1.0)
        d21 = ee3/(ee2**3) if ee2>0 else 0
        d22 = ee3**2/((ee2**2)**3) if ee2>0 else 0
        c21 = ee3/(ee2**2) if ee2>0 else 0
        c22 = ee3**2/((ee2**2)**2) if ee2>0 else 0
#         angularity = calc_angularity(found_jet)
#         KtDeltaR = calc_KtDeltaR(found_jet)
#         T1.append(t1)
#         T2.append(t2)
#         T3.append(t3)
        T21.append(t21)
#         T32.append(t32)
#         EE2.append(ee2)
#         EE3.append(ee3)
#         D2.append(d2)
#         ANGU.append(angularity)
#         KTDEL.append(KtDeltaR)
#         PT.append(found_jet.pt)
#         ETA.append(found_jet.eta)
#         PHI.append(found_jet.phi)
        MASS.append(found_jet.mass)
#         MASS.append(mass)
        D21.append(d21)
        D22.append(d22)
        C21.append(c21)
        C22.append(c22)
#     return({'T1':T1, 'T2':T2, 'T3':T3, 'T21':T21, 'T22':T22, 'T32':T32, 'EE2':EE2, 'EE3':EE3, 'D2':D2, 'angularity':ANGU,
#             'KtDeltaR':KTDEL, 'PT':PT, 'ETA':ETA, 'PHI':PHI, 'MASS':MASS})
    return({'T21':T21, 'D21':D21, 'MASS':MASS, 'D22':D22, 'C22':C22, 'C21':C21})


file = uproot.open(inputfile)
events = np.array([np.array(file["Delphes;1"]["Tower.ET"].array()),
                   np.array(file["Delphes;1"]["Tower.Eta"].array()),
                   np.array(file["Delphes;1"]["Tower.Phi"].array()),
                   np.array(file["Delphes;1"]["Tower.E"].array())*0  #assume m<<1
                  ])
events = np.expand_dims(events, axis=-1)
events = events.transpose((1,0,2))
events = np.squeeze(events,axis=(2,))

jss = JSS_V(events)
j_im_my = root_2_jetim(inputfile)
outfile = outpath + 'JSS_Jim.h5' # output jetim .h5 file 
hf = h5py.File(outfile, 'w')
for key in jss.keys():
    hf.create_dataset(key, data=jss[key])
hf.create_dataset('Jet_im_my', data=j_im_my)
hf.close()


#-----------------------------------------------------------------------------------------
