"""
Brad Rafferty
brjrafferty@gmail.com

S: state space
s: current state
s_p: s', next state
a: current action
a_p: a', next action
r: current reward
Q: value / utility
"""


from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from bisect import bisect_left
import time


def plot_grid(data, policies):
    # Parse data
    s_vector = data[:, 0]
    r_vector = data[:, 2]
    S = np.unique(s_vector)
    mean_reward = {}
    for s in S:
        idcs = list(np.where(s == s_vector))[0]
        rewards = [r_vector[i] for i in idcs]
        mean_reward[s] = np.mean(rewards)

    fig, ax = plt.subplots()
    for pol in policies:
        i = pol-1
        if mean_reward[pol] == 0:
            col = 'red'
            fs = 8
            if policies[pol] == 1:
                plt.text(np.unravel_index(i, (10, 10))[1]+0.5, np.unravel_index(i, (10, 10))[0]+0.5,
                         "<" , color=col, fontsize=fs)
            elif policies[pol] == 2:
                plt.text(np.unravel_index(i, (10, 10))[1]+0.5, np.unravel_index(i, (10, 10))[0]+0.5,
                         ">" % (mean_reward[pol]), color=col, fontsize=fs)
            elif policies[pol] == 3:
                plt.text(np.unravel_index(i, (10, 10))[1]+0.5, np.unravel_index(i, (10, 10))[0]+0.5,
                         "^" % (mean_reward[pol]), color=col, fontsize=fs)
            else:
                plt.text(np.unravel_index(i, (10, 10))[1]+0.5, np.unravel_index(i, (10, 10))[0]+0.5,
                         "v" % (mean_reward[pol]), color=col, fontsize=fs)
        else:
            col = 'green'
            fs = 10
            plt.text(np.unravel_index(i, (10, 10))[1]+0.5, np.unravel_index(i, (10, 10))[0]+0.5, "%d" % (mean_reward[pol]), color=col, fontsize=fs)

        major_ticks = np.arange(0, 11, 1)
        ax.set_xticks(major_ticks)
        ax.set_yticks(major_ticks)
        ax.grid(linestyle='-', linewidth=0.5, color='gray')


def initialize_q(data):
    q = defaultdict(dict)
    max_state = max(data[:, 0])
    states = np.array([i for i in range(1, max_state+1)])  # state space S contains the unique entries within the data; some states may be missing, so go up to the max state in the data in increments of 1
    actions = list(range(1, 10))
    for s in states:
        for a in actions:
            q[s][a] = 0
    return q, states, actions


def initialize_n(states, actions):
    n = defaultdict(dict)
    for s in states:
        for a in actions:
            n[s][a] = 0
    return n


def define_hidden_states(data_states_list):
    max_state = max(data_states_list)
    all_states = np.array([i for i in range(1, max_state + 1)])
    observed_states = list(np.unique(data_states_list))
    hidden_states = list(set(all_states) - set(observed_states))
    return hidden_states, observed_states


def sarsa(data, gam, num_episodes):
    [Q, all_states, all_actions] = initialize_q(data)  # initialize Q(s,a)
    episode = 1  # initialize episode
    start_times = []
    end_times = []
    while episode <= num_episodes:
        print("Episode {}".format(episode))
        t = 0  # initialize time/step
        s = data[t, 0]  # initialize state
        N = initialize_n(all_states, all_actions)  # initialize counts of each (s, a) pair
        start_times.append(time.time_ns())
        while t in range(len(data)-1):
            a = data[t, 1]  # action
            r = data[t, 2]  # reward

            N[s][a] += 1  # counts of state-action pair
            alph = 1 / N[s][a]  # learning rate

            s_p = data[t+1, 0]  # sp = s' (next state)
            a_p = data[t+1, 1]  # ap = a' (next action)

            Q[s][a] = Q[s][a] + alph*(r + gam*Q[s_p][a_p] - Q[s][a])  # Sarsa update
            t += 1
            s = s_p

            if t % 10000 == 0:
                print('    Step {}'.format(t))

        episode += 1
        end_times.append(time.time_ns())
    return Q, start_times, end_times


def determine_policies(q):
    policies = {}
    states = list(q.keys())
    for s in states:
        actions = list(q[s].keys())
        vals = list(q[s].values())
        best_action = actions[vals.index(max(vals))]
        policies[s] = best_action
    return policies


def write_policy_file(filename, policies):
    with open(filename, 'w') as f:
        for s in policies.keys():
            f.write("{}\n".format(policies[s]))


def take_closest(myList, myNumber):
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before


def max_val_in_key(d):
    v = list(d.values())
    return max(v)


def key_with_min_val(d, comp_val):
    v = [abs(x - comp_val) for x in list(d.values())]
    k = list(d.keys())
    return k[v.index(min(v))]




# Input
filename = "example.csv"

# Read data
startTime = time.time_ns()
data_matrix = np.array(read_csv(filename))
endTime = time.time_ns()

# Hyperparameters:
gamma = 0.95
episodes = 3

# Execute Sarsa
[Q, startTimes, endTimes] = sarsa(data_matrix, gamma, episodes)
print("Average time per episode: %0.4f seconds" % (np.mean((np.array(endTimes) - np.array(startTimes)) / 1e9)))

# Interpolate if necessary
[S_hidden, S_observed] = define_hidden_states(data_matrix[:, 0])

# For each hidden state, apply the policy of the closest observed state
for s_hidden in S_hidden:
    s_observed_closest = min(S_observed, key=lambda x: abs(x-s_hidden))
    a_observed_closest = max(Q[s_observed_closest], key=Q[s_observed_closest].get)
    Q[s_hidden][a_observed_closest] = 1  # as long as it is nonzero, it will be higher than all other actions and will be converted to a policy

# Write out policy based on Q
policy = determine_policies(Q)

# Plot policy for gridworld
plot_grid(data_matrix, policy)

# Save policy to file
write_policy_file(filename[0:-4] + '.policy', policy)

print('Done!')
