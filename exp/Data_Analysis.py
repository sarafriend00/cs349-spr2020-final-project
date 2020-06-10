import sys
sys.path.insert(0, '..')

from utils import data
import os
import sklearn
import numpy as np
import json
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')


# ------------ HYPERPARAMETERS -------------
BASE_PATH = '../COVID-19/csse_covid_19_data/'
MIN_CASES = 1000
# ------------------------------------------
lockdown = os.path.join(
    BASE_PATH,
    'csse_covid_19_time_series',
    'Covid-states-lockdown.csv')
lockdown = data.load_csv_data(lockdown)
states_lock = []
cases_lock = []
no_lock = []
i = 0
for val in np.unique(lockdown["State"]):
    df = data.filter_by_attribute(
        lockdown, "State", val)
    idx, labels = data.get_lock_idx(df)
    idx = idx.sum(axis=0)
    if idx == -1:
        no_lock.append(i)
    states_lock.append(labels[0])
    cases_lock.append(idx)
    i=i+1

confirmed = os.path.join(
    BASE_PATH, 
    'csse_covid_19_time_series',
    'time_series_covid19_confirmed_US_states.csv')
confirmed = data.load_csv_data(confirmed)
states_conf = []
cases_conf = []
cases_diff = []
cases_peak = []
cases_first = []
cases_sum = []
cases_sum_lock = []
i = 0
for val in np.unique(confirmed["Province_State"]):
    df = data.filter_by_attribute(
        confirmed, "Province_State", val)
    cases, labels = data.get_cases_chronologically(df)
    cases = cases.sum(axis=0)
    states_conf.append(labels[0][6])
    cases_conf.append(cases)
    for c in range(len(cases)):
        if c == 0:
            c_diff = [cases[c]]
        else:
            c_diff.append(cases[c]-cases[c-1])
    cases_diff.append(c_diff)
    cases_peak.append(np.argmax(c_diff))
    cases_first.append(np.argmax(np.array(c_diff)>0))
    cases_sum.append(cases[-1])
    cases_sum_lock.append(sum(cases[cases_lock[i]:]))
    i = i+1

cases_lock = [i for j, i in enumerate(cases_lock) if j not in no_lock]
cases_peak = [i for j, i in enumerate(cases_peak) if j not in no_lock]
cases_first = [i for j, i in enumerate(cases_first) if j not in no_lock]
cases_sum = [i for j, i in enumerate(cases_sum) if j not in no_lock]
cases_sum_lock = [i for j, i in enumerate(cases_sum_lock) if j not in no_lock]

plt.scatter(cases_lock, cases_peak)
plt.xlabel("lockdown date")
plt.ylabel("Peak data")
plt.tight_layout()
plt.savefig('results/lockdown_v_peak.png')
plt.close()

plt.figure
plt.scatter(cases_lock, cases_first)
plt.xlabel("lockdown date")
plt.ylabel("First data")
plt.tight_layout()
plt.savefig('results/lockdown_v_first.png')
plt.close()

plt.figure
plt.scatter(cases_lock, cases_sum)
plt.xlabel("lockdown date")
plt.ylabel("sum cases")
plt.tight_layout()
plt.savefig('results/lockdown_v_sum.png')
plt.close()

plt.figure
plt.scatter(cases_lock, cases_sum_lock)
plt.xlabel("lockdown date")
plt.ylabel("sum cases after lockdown")
plt.tight_layout()
plt.savefig('results/lockdown_v_sum_lock.png')
plt.close()

plt.figure
for s in range(len(states_lock)):
    plt.plot([i for i in range(len(cases_diff[s]))], cases_diff[s], label=states_lock[s])
    plt.xlabel("date")
    plt.ylabel("new cases")
    # plt.legend()
    plt.tight_layout()
    plt.savefig('results/lockdown_v_new_cases.png')

plt.figure
for s in range(len(states_lock)):
    plt.plot([i for i in range(len(cases_conf[s]))], cases_conf[s], label=states_lock[s])
    plt.xlabel("date")
    plt.ylabel("total cases")
    # plt.legend()
    plt.tight_layout()
    plt.savefig('results/lockdown_v_total_cases.png')