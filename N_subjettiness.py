from pyjet import cluster
from pyjet.testdata import get_event
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import pyjet
import argparse
import glob
import math
import operator
import re
from scipy.stats import binned_statistic_2d
import pathlib

path = pathlib.Path.cwd()



def flatten(event): # for single event in order to reshape data
    z = np.zeros((event.shape[0],1))
    fp = np.append(event, z, axis=-1)
    fp = fp.transpose((1,0))
    fp = np.core.records.fromarrays( [fp[:][0],fp[:][1],fp[:][2],fp[:][3]], names= 'pT, eta, phi, mass' , formats = 'f8, f8, f8,f8')

    return fp

def jet_clustering(event, R0, p = -1):
    # R0 = Clustering radius for the main jets
    flattened_event = flatten(event)
    ## p = -1, 0, 1 => anti-kt, C/A, kt Algorithm
    sequence = cluster(flattened_event, R=R0, p= p)
    # List of jets
    jets = sequence.inclusive_jets()
    return jets
def t0(jet):
    return sum(p.pt * CalcDeltaR(p, jet) for p in jet.constituents())
def tn(jet, n, beta = 1, R=1): #t1 t2 t3 t21 t32
    assert n >= 0
    if n == 0:
        return t0(jet)
    particles = jet.constituents_array()
    if len(particles) < n:
        return -1
    subjets = pyjet.cluster(particles, R=1.0, p=1/beta).exclusive_jets(n) #Common choices include using exclusive kt axes or using “minimal” axes
                                                                        #generalised-kt algorithm with p = 1/2.1
    wta_axes = subjets
    wta_axes = np.array(wta_axes)
    if len(wta_axes)<n:
        return -1
    
    return np.sum((particles['pT']*(CalcDeltaRA(particles, wta_axes))**beta).min(axis=0)) / (jet.pt*(R**beta))

def CalcDeltaRArray(p, a):
    dEta = p['eta'] - \
        a['eta'].repeat(p.shape[0]).reshape(a.shape[0], p.shape[0])
    dPhi = np.abs(p['phi'] - a['phi'].repeat(p.shape[0]
                                             ).reshape(a.shape[0], p.shape[0]))
    mask = dPhi > np.pi
    dPhi[mask] *= -1
    dPhi[mask] += 2 * np.pi
    return (dPhi**2 + dEta**2)**0.5

def CalcDeltaRA(p, j2): ##FOR Taylor's code
    Rs = []
    for i in j2:
        ep = np.array([i.eta, i.phi])
        pa = np.append(np.expand_dims(p['eta'], axis=-1), np.expand_dims(p['phi'], axis=-1), axis = -1) 
        pa = np.transpose(pa, [1,0])
        dEP =pa - ep.repeat(p.shape[0]).reshape(ep.shape[0], p.shape[0])
        dEP[:][1] = np.abs(dEP[:][1])
        mask = dEP[:][1] > np.pi
        dEP[:][1][mask] *= -1
        dEP[:][1][mask] += 2 * np.pi
        Rs.append((dEP[:][1]**2 + dEP[:][0]**2)**0.5)
    return np.array(Rs)

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

def T21(event, beta=1):
    jet = jet_clustering(event, 1)[0]
    t1 = tn(jet, n=1, beta = beta)
    t2 = tn(jet, n=2, beta = beta)
    #         t3 = tn(found_jet, n=3)
    t21 = t2 / t1 if t1 > 0.0 else 0.0
    if t21<0:
        t21 = 0
    return t21

def generate_NSJ(file_path, NUM_Ev= None, beta=1, target = None):
    NSJ = []
    df = pd.read_hdf(file_path, "features")
    y = pd.read_hdf(file_path, "targets")
    
    if NUM_Ev == None:
        NUM_Ev = len(df.iloc[:])
    for i in tqdm(range(NUM_Ev)):
        if target != None:
            if y.iloc[i].values[0] == 1-target:
                continue
        event = df.iloc[i].values[0]
        NSJ.append(T21(event, beta=beta))
    return(NSJ)
