# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# %%
import networkx as nx
import matplotlib.pyplot as plt
import random
import scipy as sp
from statistics import mean
import numpy as np
import functools
import simpy
from SimComponents import PacketGenerator, PacketSink, SwitchPort, PortMonitor
import tempfile
import itertools as IT
import os
import pandas as pd


def uniquify(path, sep=''):
    def name_sequence():
        count = IT.count()
        yield ''
        while True:
            yield '{s}{n:d}'.format(s=sep, n=next(count))

    orig = tempfile._name_sequence
    with tempfile._once_lock:
        tempfile._name_sequence = name_sequence()
        path = os.path.normpath(path)
        dirname, basename = os.path.split(path)
        filename, ext = os.path.splitext(basename)
        fd, filename = tempfile.mkstemp(dir=dirname, prefix=filename, suffix=ext)
        tempfile._name_sequence = orig
    return filename


def adj_list_to_matrix(adj_list):
    n = len(adj_list)
    adj_matrix = np.zeros((119, 119))

    for i in range(n):
        a = adj_list[i][0]
        b = adj_list[i][1]
        adj_matrix[a][b] = 1
        adj_matrix[b][a] = 1

    return adj_matrix


##GRAPH AND ADJACENCY MATRIX GENERATOR########################################

first_column = [1, 1, 4, 3, 5, 6, 8, 8, 8, 9, 4, 5, 11, 2, 3, 7, 11, 12, 13, 14, 12, 15, 16, 17, 18, 19, 15, 20, 21, 22,
                23, 23, 26, 25, 27, 28, 30, 8, 26, 17, 29, 23, 31, 27, 15, 19, 35, 35, 33, 34, 34, 38, 37, 37, 30, 39,
                40, 40, 41, 43, 34, 44, 45, 46, 46, 47, 42, 42, 45, 48, 49, 49, 51, 52, 53, 49, 49, 54, 54, 55, 56, 50,
                56, 51, 54, 56, 56, 55, 59, 59, 60, 61, 63, 63, 63, 64, 38, 64, 49, 49, 62, 62, 63, 66, 65, 47, 49, 68,
                69, 24, 70, 24, 71, 71, 70, 70, 69, 74, 76, 69, 75, 77, 78, 77, 77, 79, 68, 81, 77, 82, 83, 83, 84, 85,
                86, 85, 85, 88, 89, 89, 90, 89, 89, 91, 92, 92, 93, 94, 80, 82, 94, 80, 80, 92, 94, 95, 96, 98, 99, 100,
                92, 101, 100, 100, 103, 103, 100, 104, 10, 105, 105, 106, 108, 103, 109, 110, 110, 17, 32, 32, 27, 114,
                68, 12, 75, 76]
first_column = np.array(first_column)
second_column = [2, 3, 5, 5, 6, 7, 9, 5, 10, 11, 11, 12, 12, 12, 12, 13, 14, 15, 15, 16, 17, 17, 18, 19, 20, 19, 21, 22,
                 23, 24, 25, 25, 27, 28, 29, 17, 30, 30, 31, 31, 32, 32, 32, 33, 34, 36, 37, 37, 36, 37, 37, 39, 40, 38,
                 40, 41, 42, 42, 44, 43, 45, 46, 47, 48, 49, 49, 49, 49, 49, 50, 51, 52, 53, 54, 54, 54, 55, 56, 56, 57,
                 57, 58, 58, 59, 59, 59, 59, 60, 61, 61, 62, 62, 59, 64, 61, 65, 65, 66, 66, 66, 67, 59, 67, 68, 69, 69,
                 69, 70, 70, 71, 72, 72, 73, 74, 75, 75, 75, 77, 77, 77, 78, 79, 80, 80, 80, 81, 80, 82, 83, 84, 85, 85,
                 86, 87, 88, 89, 89, 90, 90, 91, 92, 92, 92, 93, 94, 94, 95, 96, 96, 96, 97, 98, 99, 100, 100, 96, 97,
                 100, 100, 101, 102, 102, 103, 104, 104, 105, 106, 105, 106, 107, 108, 107, 109, 110, 110, 111, 112,
                 113, 113, 114, 115, 115, 116, 117, 118, 118]
second_column = np.array(second_column)
ad_list = np.vstack((first_column, second_column))
ad_list = ad_list.T
adj_matrix = adj_list_to_matrix(ad_list)
print(np.count_nonzero(adj_matrix > 0))
################################################
# %%
##THROUGHPUT VALUES###########################################################

norm_throughputs = np.linspace(12.75, 19.0, 1000)
# l_mal_throughputs = np.linspace(3.75,7.75,1000)
# le_mal_throughputs = np.linspace(8.00,11.25,1000)
mal_throughputs = np.linspace(5, 12.75, 101)
# male_throughputs = np.linspace(1.75,3.50,1000)


##LATENCY VALUES##############################################################

# exponential arrival distribution for generator

adist_list = np.zeros(shape=(1000))
for i in range(1000):
    adist_list[i] = random.expovariate(10)

global step_num
step_num = 100
adist_ramp = np.linspace(.01, .5, 101)


def adistramp():
    global step_num
    val = random.choice(adist_list) + adist_ramp[step_num]
    print(val)
    return val


def adist():
    # adist = functools.partial(random.expovariate, 10)
    return random.choice(adist_list) + adist_ramp[100]


# mean size 100 bytes
sdist_list = np.zeros(shape=(1000))
for i in range(1000):
    sdist_list[i] = random.expovariate(0.01)


def sdist():
    # sdist = functools.partial(random.expovariate, 0.1)
    return random.choice(sdist_list)


samp_dist_list = np.zeros(shape=(1000))
for i in range(1000):
    samp_dist_list[i] = random.expovariate(1.0)


def samp_dist():
    # samp_dist = functools.partial(random.expovariate, 1.0)
    return random.choice(samp_dist_list)


def att_values_check(last):
    if not last:
        return random.sample(range(1, 118), 5)
    else:
        global step_num
        step_num = 100
        return []


# Variables for switch function in simulation
port_rate_norm = 900.0
port_rate_mal = 900.0
qlimit_norm = 10000
qlimit_mal = 10000

# %%
##MISC VALUES##############################################################

BIG_TD = []
BIG_BW = []
FLIPS = []
ABUSES = []
Flip = 1
generated_IAT = []
BIG_PACKETS = []

# Execute the simulation
if __name__ == '__main__':
    att_values = []

    for step in range(230):
        # print(step)

        TD_list = []
        Bw_list = []
        inter_arrivals = []
        p_count = []

        if (step % 100) == 99:
            att_values = att_values_check(att_values)
        elif att_values:
            step_num -= 1

        print(att_values)

        for i in range(len(ad_list)):

            if (ad_list[i][0] not in att_values) and (ad_list[i][1] not in att_values):
                env = simpy.Environment()  # Create the SimPy environment
                # Create the packet generators and sink
                ps = PacketSink(env, debug=False, rec_arrivals=True, absolute_arrivals=False)
                pg = PacketGenerator(env, "Greg", adist, sdist)
                switch_port = SwitchPort(env, port_rate_norm, qlimit_norm)
                # Using a PortMonitor to track queue sizes over time
                # pm = PortMonitor(env, switch_port, samp_dist)
                # Wire packet generators, switch ports, and sinks together
                pg.out = switch_port
                switch_port.out = ps
                # Run it
                env.run(until=15)

                inter_arrivals.append(mean(ps.arrivals))
                p_count.append(pg.packets_sent)
                TD_list.append(mean(ps.waits))
                Bw_list.append(random.choice(norm_throughputs))

            if (ad_list[i][0] in att_values) or (ad_list[i][1] in att_values):
                # val = random.randint(6,25)
                env = simpy.Environment()  # Create the SimPy environment
                # Create the packet generators and sink
                ps = PacketSink(env, debug=False, rec_arrivals=True, absolute_arrivals=False)
                pg = PacketGenerator(env, "Greg", adistramp, sdist)
                switch_port = SwitchPort(env, port_rate_mal, qlimit_mal)
                # Using a PortMonitor to track queue sizes over time
                # pm = PortMonitor(env, switch_port, samp_dist)
                # Wire packet generators, switch ports, and sinks together
                pg.out = switch_port
                switch_port.out = ps
                # Run it
                env.run(until=15)

                inter_arrivals.append(mean(ps.arrivals))
                p_count.append(pg.packets_sent)
                TD_list.append(mean(ps.waits))
                Bw_list.append(mal_throughputs[step_num])

        # print(step)
        BIG_BW.append(Bw_list)
        BIG_TD.append(TD_list)
        ABUSES.append(att_values)
        FLIPS.append(Flip)
        generated_IAT.append(inter_arrivals)
        BIG_PACKETS.append(p_count)

    path = 'C:/Users/Morgan/Dropbox/Morgan/GRADUATE/EEL6935 Distributed Computing/slowddos/results/truth.csv'
    df = pd.DataFrame(data=FLIPS)
    df.to_csv(uniquify(path))

    path = 'C:/Users/Morgan/Dropbox/Morgan/GRADUATE/EEL6935 Distributed Computing/slowddos/results/attacked_buses.csv'
    df = pd.DataFrame(data=ABUSES)
    df.to_csv(uniquify(path))

    path = 'C:/Users/Morgan/Dropbox/Morgan/GRADUATE/EEL6935 Distributed Computing/slowddos/results/throughput.csv'
    df = pd.DataFrame(data=BIG_BW)
    df.to_csv(uniquify(path))

    path = 'C:/Users/Morgan/Dropbox/Morgan/GRADUATE/EEL6935 Distributed Computing/slowddos/results/td.csv'
    df = pd.DataFrame(data=BIG_TD)
    df.to_csv(uniquify(path))

    path = 'C:/Users/Morgan/Dropbox/Morgan/GRADUATE/EEL6935 Distributed Computing/slowddos/results/adj_mat.csv'
    df = pd.DataFrame(data=adj_matrix)
    df.to_csv(uniquify(path))

    path = 'C:/Users/Morgan/Dropbox/Morgan/GRADUATE/EEL6935 Distributed Computing/slowddos/results/z.csv'
    df = pd.DataFrame(data=ad_list)
    df.to_csv(uniquify(path))

    path = 'C:/Users/Morgan/Dropbox/Morgan/GRADUATE/EEL6935 Distributed Computing/slowddos/results/IAT.csv'
    df = pd.DataFrame(data=generated_IAT)
    df.to_csv(uniquify(path))

    path = 'C:/Users/Morgan/Dropbox/Morgan/GRADUATE/EEL6935 Distributed Computing/slowddos/results/pcount.csv'
    df = pd.DataFrame(data=BIG_PACKETS)
    df.to_csv(uniquify(path))
