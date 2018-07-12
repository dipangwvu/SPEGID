#!/usr/bin/env python
"""
Created on Thu Jun 20 2018

@author: di
"""
# use _SPEG_all.csv as start
# DM range of the central part of some SPEG, needs input
# remove if some SPEGs are too close < 100ms
# tolerance 0.99

import os
import time
import pandas as pd
import numpy as np
import itertools
from copy import deepcopy

os.chdir("/home/di/Documents/Paper_Module/test_data/56475/p2030.20130702.G33.79+00.82.N.b3.00000")
# center overlap and contain the peak of the brightest pulse
# find current working directory full path
cwd = os.getcwd()
print cwd
last_sub_dir = cwd.rfind('/')
# find out the name of current directory
start_idx = last_sub_dir + 1
cur_dir = cwd[start_idx:]
print cur_dir

# find out the current parent directory
parent_dir = cwd[:last_sub_dir]
# find out name of the parent dir
last_sub_dir = parent_dir.rfind('/')
cur_parent_dir = parent_dir[last_sub_dir + 1:]
print cur_parent_dir

error_log = "/home/di/Documents/AstroData/ErrorLog/PALFA_Period.txt"
error_log_fp = open(error_log, 'a')

# tolerance 0.05 period
tolerance = 0.01

# read in SPEGs file
SPEG_file = '_SPEG_all'
cur_SPEGs = cur_dir + SPEG_file + '.csv'
print cur_SPEGs

# read in SPEG data frame
try:
    SPEG_DF = pd.read_table(cur_SPEGs, sep=",", skipinitialspace=True)
    print SPEG_DF.shape

except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
    error_msg = cur_parent_dir + '/' + cur_dir + ',' + 'reading SPEGs failed!\n'
    error_log_fp.write(error_msg)
    exit('reading SPEGs failed!')

# read in the single pulse events(spe) file
cur_singlepulses = cur_dir + 'singlepulses.csv'

try:
    spe_DF_full = pd.read_table(cur_singlepulses, sep=",", skipinitialspace=True, skiprows=1, header=None,
                                engine="c", names=["DM", "SNR", "time", "sampling", "downfact"],
                                dtype={"DM": np.float64, "SNR": np.float64, "time": np.float64,
                                       "sampling": np.uint32, "downfact": np.uint16})

    # remove single-pulse events too bright or too dim
    spe_DF_clean = spe_DF_full.loc[(spe_DF_full['SNR'] < 10000) & (spe_DF_full['SNR'] > 4.99)].copy()

except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
    error_msg = cur_parent_dir + '/' + cur_dir + ',' + 'reading singlepulses failed!\n'
    error_log_fp.write(error_msg)
    exit('reading singlepulses failed!')


# the classes for single-pulse event
class TimeRange(object):
    __slots__ = ['SPEG_rank', 'min_time', 'max_time']

    def __init__(self, current_list):
        """
        This is to get time ranges from SPEGs
        """
        self.SPEG_rank = current_list[0]
        self.min_time = current_list[1]
        self.max_time = current_list[2]


class SPEV(object):
    """
    A single-pulse event belonging to a SPEG is a valid single-pulse event.
    """
    __slots__ = ['DM', 'SNR', 'time', 'SPEG_rank', 'shadow']

    def __init__(self, current_list):
        self.DM = current_list[0]
        self.SNR = current_list[1]
        self.time = current_list[2]
        self.SPEG_rank = -1
        self.shadow = False

    def __str__(self):
        # print current pulse
        s = ["\n\tSPEG_rank:   %4d" % self.SPEG_rank,
             "\t  DM:   %5.2f" % self.DM,
             "\ttime: %3.6f" % self.time,
             "\t SNR:  %3.2f" % self.SNR,
             "\tshadow pulse:  %s" % self.shadow,
             "--------------------------------"
             ]
        return '\n'.join(s)


class SPEArray(object):
    """
    A group of single pulse events in the same DM channel
    """
    __slots__ = ['period', 'DM', 'spes', 'toa', 'num_of_spes', 'max_SNR', 'SPEG_rank', 'error', 'num_of_periods']

    def __init__(self, current_list):
        self.spes = current_list
        self.num_of_spes = len(current_list)
        self.period = -1
        self.error = -1
        self.max_SNR = -1
        self.SPEG_rank = None
        self.toa = None
        self.num_of_periods = None

    def __str__(self):
        # print current group
        s = ["\n\t            DM:   %5.2f" % self.DM,
             "\tNumber of spes:   %4d" % self.num_of_spes,
             "\t  Pulses ranks:   %s" % self.SPEG_rank,
             "\t        Period: %3.6f" % self.period,
             "\t         Error:  %3.5f" % self.error,
             "---------------------------------------"
             ]
        return '\n'.join(s)


cur_option = raw_input("Select what you want to do(1/2/3)?\n1. find the periodicity \n2. find the periodicity of a subset\
of spes \n3. verify the known periodicity\nYou choice: ")

if cur_option == '1' or cur_option == '2' or cur_option == '3':
    print cur_option
else:
    exit("You should input 1/2/3 ONLY!")

if cur_option == '1':
    # specify the SPEG(s) which have to be included in the the array
    str_input = raw_input("enter the rank(s) of key SPEG(s) (separated by comma): ")
    SPEGs_included_ranks = map(int, str_input.split(','))

    SPEGs_included_ranks.sort()
    # sort in ascending order
    print "included key SPEGs: ", SPEGs_included_ranks

    # the brightest SPEG that has to be included
    key_SPEG = SPEG_DF.loc[SPEG_DF.SPEG_rank == SPEGs_included_ranks[0]]
    print key_SPEG

    # Only check the peak DM of key_SPEG
    peak_DM = float(key_SPEG.peak_DM)
    # get all the points in the DM range
    print "peak_DM: ", peak_DM

    # group rank: all SPEGs having the same rank as key_SPEG
    group_rank = int(key_SPEG.group_rank)
    # print group_rank

    excluded_SPEG_rank = int(raw_input("enter the rank of the SPEG to be excluded: "))
    print "excluded SPEG:", excluded_SPEG_rank
    # get all the single-pulse events in the DM range and also in SPEGs
    SPEG_group = SPEG_DF.loc[SPEG_DF.group_rank == group_rank]
    print "to start with: ", SPEG_group.shape[0]

    time_ranges_list = []

    # generate a list of time intervals
    for i in range(SPEG_group.shape[0]):
        cur_SPEG = SPEG_group.iloc[i]
        cur_SPEG_rank = cur_SPEG.SPEG_rank
        if cur_SPEG_rank != excluded_SPEG_rank:
            time1 = cur_SPEG.min_time
            time2 = cur_SPEG.max_time
            cur_times = [cur_SPEG_rank, time1, time2]
            # TimeRange class
            cur_TimeRange = TimeRange(cur_times)
            time_ranges_list.append(cur_TimeRange)

    # all single-pulse events in this DM channel
    spes_in_Channel = spe_DF_clean.loc[spe_DF_clean['DM'] == peak_DM, ]
    print spes_in_Channel.shape

    # select single-pulse events belonging to SPEGs
    spe_list = []
    for i in range(spes_in_Channel.shape[0]):
        cur_spe = spes_in_Channel.iloc[i]
        # make a list
        cur_list = [cur_spe.DM, cur_spe.SNR, cur_spe.time]
        # a valid single-pulse event
        cur_SPEV = SPEV(cur_list)
        in_SPEG = False
        # check if the spe is in any valid time range of SPEGs
        for cur_TimeRange in time_ranges_list:
            if cur_TimeRange.min_time <= cur_SPEV.time <= cur_TimeRange.max_time:
                in_SPEG = True
                cur_SPEV.SPEG_rank = cur_TimeRange.SPEG_rank
                break
        # add to list if spe in the time interval of an SPEG
        if in_SPEG:
            spe_list.append(cur_SPEV)

    # total number of spes to start with
    spe_number = len(spe_list)

    # store good group
    good_arrays = []

    # single-pulse events at least this much (50ms) away from each other
    TOA_diff_bound = 0.05

    # if there are more than 2 spes, remove the shadow SPEG if found
    if spe_number > 2:
        for i in range(spe_number - 1):
            cur_spe = spe_list[i]
            for j in range((i + 1), spe_number):
                ano_spe = spe_list[j]
                # two spes that are really close
                if abs(cur_spe.time - ano_spe.time) < TOA_diff_bound:
                    if cur_spe.SNR > ano_spe.SNR:
                        ano_spe.shadow = True
                    else:
                        cur_spe.shadow = True
    # remove shadow spe
    spe_candidates = [each_spe for each_spe in spe_list if not each_spe.shadow]

    # if there are more than 2 spes, check periodicity
    spe_number = len(spe_candidates)
    if spe_number > 2:
        # get all combinations
        # no period found and spe number at least 3, subset_size is the length of subgroup
        print "total spes in group:", spe_number
        # time.sleep(1)
        # check all combinations of 3 or more in reversed order
        for subset_size in reversed(range(3, spe_number + 1)):
            # a list of candidates
            for cur_spe_list in itertools.combinations(spe_candidates, subset_size):
                # initialize a single-pulse event array
                cur_spe_array = SPEArray(cur_spe_list)

                # create a list of time
                cur_spe_time_list = []
                #           # empty dictionary
                #           curCluRankTimeDict = {}
                # store the ranks of the list
                SPEG_rank_in_list = []
                # store the SNRs of the list
                SNR_in_list = []

                # find the ranks of the single-pulses in the list
                for each_spe in cur_spe_list:
                    # print each_spe.time
                    cur_TOA = round(each_spe.time, 7)
                    cur_spe_time_list.append(cur_TOA)
                    # the SPEG rank
                    SPEG_rank_in_list.append(each_spe.SPEG_rank)
                    SNR_in_list.append(each_spe.SNR)

                cur_spe_array.SPEG_rank = SPEG_rank_in_list

                is_legit_group = True
                # required SPEGs have to be in the group
                if not set(SPEGs_included_ranks).issubset(set(SPEG_rank_in_list)):
                    is_legit_group = False

                valid_array_subset = False
                # check if these ranks are subset of a good group, if they are, skip the rest of the loop
                if len(good_arrays) > 0:
                    for each_array in good_arrays:
                        periodic_spes_ranks = each_array.SPEG_rank
                        if set(SPEG_rank_in_list).issubset(periodic_spes_ranks):  # is a subset()
                            valid_array_subset = True
                            break

                if not valid_array_subset and is_legit_group:
                    cur_max_SNR = max(SNR_in_list)
                    # the brightest spd in the array has to be bright
                    if cur_max_SNR >= 6:
                        # calculate period
                        # find difference
                        diffs = np.diff(cur_spe_time_list)
                        print "diffs: ", diffs
                        # total difference
                        total_diff = cur_spe_time_list[-1] - cur_spe_time_list[0]

                        denom = 2

                        # period is at most half of the total diff
                        cur_period = total_diff / denom

                        cur_min_diff = diffs.min()

                        # as long as the period is greater than 0.05 second
                        if cur_min_diff > TOA_diff_bound:
                            print "number of spes in current group:", len(cur_spe_list)
                            print "cur_min_diff", cur_min_diff
                            while cur_period > TOA_diff_bound:
                                # each diff is how many periods
                                period_number_list = []
                                all_good = True
                                # error sum
                                error_sum = 0
                                # current period should be less than twice the min diff to avoid rounding up problem
                                if cur_period < cur_min_diff * 2:
                                    for cur_diff in diffs:
                                        periods_number = round(cur_diff / cur_period, 4)
                                        period_number_list.append(periods_number)

                                        res = cur_diff / cur_period % 1
                                        # the error is small
                                        error_sum = error_sum + min(res, 1 - res)
                                        # check if all diffs good
                                        if tolerance < res < 1 - tolerance:
                                            all_good = False
                                            break

                                    # if all spes are good
                                    if all_good:
                                        # update period if it is found
                                        cur_spe_array.period = round(cur_period, 7)
                                        cur_spe_array.error = round(error_sum, 7)
                                        cur_spe_array.num_of_periods = period_number_list
                                        cur_spe_array.max_SNR = cur_max_SNR
                                        cur_spe_array.toa = cur_spe_time_list
                                        # a good array is found
                                        good_arrays.append(deepcopy(cur_spe_array))

                                # decrease period if needed
                                denom = denom + 1
                                # print denom, cur_period
                                cur_period = total_diff / denom

    header = 'numOfGoodClu' + '\t' + 'max_SNR' + '\t' + 'period' + '\t' + 'DM' + '\t' + \
             'error' + '\t' + 'SPEGRank' + '\t' + 'timeOfArrival' + '\t' + 'num_of_periods'
    good_arrays = list(set(good_arrays))

    if len(good_arrays) > 0:
        # sort good_arrays
        def mixed_order(cur_group):
            return -cur_group.max_SNR, -cur_group.num_of_spes, -cur_group.period, cur_group.error

        good_arrays.sort(key=mixed_order)

        # write to file
        periodicity_result = 'NewPeriods' + str(tolerance) + '-' + cur_parent_dir + '-' + cur_dir \
                             + '_' + str_input + SPEG_file + '.txt'

        periodicity_result_fp = open(periodicity_result, 'w')
        
        periodicity_result_fp.write(header + '\n')
        
        for each_array in good_arrays:
            #        print each_array
            cur_line = str(each_array.num_of_spes) + '\t' + str(each_array.max_SNR) + '\t' + str(each_array.period) + '\t' \
                      + str(peak_DM) + '\t' + str(each_array.error) + '\t' + str(each_array.SPEG_rank).strip('[]') + '\t' \
                      + str(each_array.toa).strip('[]') + '\t' + str(each_array.num_of_periods).strip('[]')

            periodicity_result_fp.write(cur_line + '\n')

        periodicity_result_fp.close()

    print "number of good groups: ", len(good_arrays)
    error_log_fp.close()

# find the periodicity of a subset of specified  SPEGs
elif cur_option == '2':
    # specify the SPEGs of interest
    str_input = raw_input("enter the ranks of SPEGs of interest (separated by comma): ")
    SPEGs_included_ranks = map(int, str_input.split(','))

    # sort in ascending order
    SPEGs_included_ranks.sort()
    print "SPEGs included: ", SPEGs_included_ranks

    key_SPEG = SPEG_DF.loc[SPEG_DF.SPEG_rank == SPEGs_included_ranks[0]]
    print key_SPEG

    # only check the peak DM
    key_peak_DM = float(key_SPEG.peak_DM)
    # get all the points in the DM range
    print "key_peak_DM: ", key_peak_DM

    time_ranges_list = []

    # generate a class of time range
    for i in SPEGs_included_ranks:
        # the i-1th row is SPEG i
        cur_SPEG = SPEG_DF.loc[SPEG_DF.SPEG_rank == i]
        cur_SPEG_rank = int(cur_SPEG.SPEG_rank)

        time1 = float(cur_SPEG.min_time)
        time2 = float(cur_SPEG.max_time)
        cur_times = [cur_SPEG_rank, time1, time2]
        # time range class
        cur_TimeRange = TimeRange(cur_times)
        time_ranges_list.append(cur_TimeRange)

    # all spes in this DM channel
    spes_in_Channel = spe_DF_clean.loc[(spe_DF_clean['DM'] == key_peak_DM),]
    print spes_in_Channel.shape

    # select only spes in SPEGs
    spe_list = []
    for i in range(spes_in_Channel.shape[0]):
        cur_spe = spes_in_Channel.iloc[i]
        # make a list
        cur_list = [cur_spe.DM, cur_spe.SNR, cur_spe.time]
        cur_SPEV = SPEV(cur_list)
        in_SPEG = False
        # check if the pulse is in time range
        for cur_TimeRange in time_ranges_list:
            if cur_TimeRange.min_time <= cur_SPEV.time <= cur_TimeRange.max_time:
                in_SPEG = True
                cur_SPEV.SPEG_rank = cur_TimeRange.SPEG_rank
                break
        if in_SPEG:
            spe_list.append(cur_SPEV)

    # count the spes
    spe_number = len(spe_list)

    # store good array
    good_arrays = []

    # single-pulse events at least this much (50ms) away
    TOA_diff_bound = 0.05

    # if there are more than 2 points, remove the shadow SPEG if found
    if spe_number > 2:
        for i in range(spe_number - 1):
            cur_spe = spe_list[i]
            for j in range((i + 1), spe_number):
                ano_spe = spe_list[j]
                # two spes that are really close
                if abs(cur_spe.time - ano_spe.time) < TOA_diff_bound:
                    if cur_spe.SNR > ano_spe.SNR:
                        ano_spe.shadow = True
                    else:
                        cur_spe.shadow = True

    # remove shadow single-pulse events
    spe_candidates = [each_spe for each_spe in spe_list if not each_spe.shadow]

    # if there are more than 2 spes, check periodicity
    spe_number = len(spe_candidates)
    if spe_number > 2:
        # get all combinations
        # no period found and spe number at least 3, subset_size is the length of subgroup
        print "total spes in group:", spe_number
        # time.sleep(1)
        # check all combinations of 3 or more in reversed order
        for subset_size in reversed(range(3, spe_number + 1)):
            for cur_spe_list in itertools.combinations(spe_candidates, subset_size):
                # initialize an spe array
                cur_spe_array = SPEArray(cur_spe_list)

                # create a list of time
                cur_spe_time_list = []
                # store the ranks of current list
                SPEG_rank_in_list = []
                # store the SNR of the list
                SNR_in_list = []

                # find the ranks of the SPEGs that spes belonging to
                for each_spe in cur_spe_list:
                    # round to 7 decimal places
                    cur_TOA = round(each_spe.time, 7)
                    cur_spe_time_list.append(cur_TOA)
                    SPEG_rank_in_list.append(each_spe.SPEG_rank)
                    SNR_in_list.append(each_spe.SNR)

                # update the ranks of the SPEG array
                cur_spe_array.SPEG_rank = SPEG_rank_in_list

                valid_array_subset = False
                # check if these ranks are subset of a good group
                if len(good_arrays) > 0:
                    for each_array in good_arrays:
                        periodic_spes_ranks = each_array.SPEG_rank
                        if set(SPEG_rank_in_list).issubset(periodic_spes_ranks):  # is a subset()
                            valid_array_subset = True
                            break

                if not valid_array_subset:
                    cur_max_SNR = max(SNR_in_list)
                    if cur_max_SNR >= 6:
                        # calculate period
                        # find difference
                        diffs = np.diff(cur_spe_time_list)
                        # total difference
                        total_diff = cur_spe_time_list[-1] - cur_spe_time_list[0]
                        denom = 2

                        # period is at most half of the total
                        cur_period = total_diff / denom

                        # as long as the period is greater than 0.05 second
                        cur_min_diff = diffs.min()

                        if cur_min_diff > TOA_diff_bound:
                            print "number of spes in current group:", len(cur_spe_list)
                            print "cur_min_diff", cur_min_diff
                            while (cur_period > TOA_diff_bound):
                                period_number_list = []
                                all_good = True
                                # error sum
                                error_sum = 0
                                if cur_period < cur_min_diff * 2:
                                    for cur_diff in diffs:
                                        periods_number = round(cur_diff / cur_period, 4)
                                        period_number_list.append(periods_number)

                                        res = cur_diff / cur_period % 1
                                        # the error is small
                                        error_sum = error_sum + min(res, 1 - res)
                                        # if they are all good
                                        if tolerance < res < 1 - tolerance:
                                            all_good = False
                                            break

                                    # if all spes are good
                                    if all_good:
                                        # update period if it is found
                                        cur_spe_array.period = round(cur_period, 7)
                                        cur_spe_array.error = round(error_sum, 7)
                                        cur_spe_array.num_of_periods = period_number_list
                                        cur_spe_array.max_SNR = cur_max_SNR
                                        cur_spe_array.toa = cur_spe_time_list
                                        # found a good group
                                        good_arrays.append(deepcopy(cur_spe_array))

                                denom = denom + 1
                                # print denom, cur_period
                                cur_period = total_diff / denom

    header = 'numOfGoodClu' + '\t' + 'max_SNR' + '\t' + 'period' + '\t' + 'DM' + '\t' + \
             'error' + '\t' + 'SPEGRank' + '\t' + 'timeOfArrival' + '\t' + 'num_of_periods'
    good_arrays = list(set(good_arrays))

    if len(good_arrays) > 0:
        # sort good_arrays by pulseNum and error
        def mixed_order(cur_group):
            return -cur_group.max_SNR, -cur_group.num_of_spes, -cur_group.period, cur_group.error

        good_arrays.sort(key=mixed_order)

        periodicity_result = 'NewPeriodsSubset' + str(tolerance) + '-' + cur_parent_dir + '-' + cur_dir \
                       + '_' + str_input + SPEG_file + '.txt'
        periodicity_result_fp = open(periodicity_result, 'w')
        periodicity_result_fp.write(header + '\n')
        for each_array in good_arrays:
            #        print each_array
            cur_line = str(each_array.num_of_spes) + '\t' + str(each_array.max_SNR) + '\t' + str(each_array.period)\
                       + '\t' + str(key_peak_DM) + '\t' + str(each_array.error)\
                       + '\t' + str(each_array.SPEG_rank).strip('[]') + '\t'\
                       + str(each_array.toa).strip('[]') + '\t' \
                       + str(each_array.num_of_periods).strip('[]')
            #        print cur_line
            periodicity_result_fp.write(cur_line + '\n')

        periodicity_result_fp.close()

    print "number of good groups: ", len(good_arrays)
    error_log_fp.close()

# verify all SPEGs with known period and known true spes
elif cur_option == '3':
    # specify the key SPEG
    str_input = raw_input("enter the rank(s) of two SPEG(s) to be confirmed (separated by comma): ")

    SPEGs_included_ranks = map(int, str_input.split(','))
    SPEGs_included_ranks.sort()
    # sort in ascending order
    print "SPEGs to be confirmed: ", SPEGs_included_ranks

    str_input2 = raw_input("Enter the predefined period: ")
    cur_period = float(str_input2)

    print "Known Period: ", cur_period

    # add the two know spes
    time_ranges_list = []
    SPEG_one_rank = SPEGs_included_ranks[0]
    SPEG_one = SPEG_DF.loc[SPEG_DF.SPEG_rank == SPEG_one_rank]
    time1 = float(SPEG_one.min_time)
    time2 = float(SPEG_one.max_time)

    SPEG_two_rank = SPEGs_included_ranks[1]
    SPEG_two = SPEG_DF.loc[SPEG_DF.SPEG_rank == SPEG_two_rank]
    time3 = float(SPEG_two.min_time)
    time4 = float(SPEG_two.max_time)

    # do you want to check one or check all
    str_input3 = raw_input("Do you want to check all SPEGs or just check some SPEGs? Input all or some.")
    if str_input3 == "all":
        verified_result = 'verifiedPeriodsAll' + str(tolerance) + '-' + cur_parent_dir + '-' + cur_dir + '-' \
                          + str_input + SPEG_file + '.csv'
        verified_result_fp = open(verified_result, 'w')

        # verify other SPEGs one by one
        for i in range(SPEG_DF.shape[0]):
            ano_SPEG = SPEG_DF.loc[i]
            SPEG_rank = ano_SPEG.SPEG_rank
            # python index starts from 0
            if SPEG_rank in SPEGs_included_ranks:
                astro_pulse = "YES"
                output_str = str(SPEG_rank) + ',' + astro_pulse
                verified_result_fp.write(output_str + '\n')
            else:
                # verify the known period of another SPEG
                key_peak_DM = float(ano_SPEG.peak_DM)
                print "key_peak_DM: ", key_peak_DM
                astro_pulse = "NO"

                # single-pulse events between time 1 and time 2
                spe_in_SPEG_one = spe_DF_clean.loc[(spe_DF_clean['DM'] == key_peak_DM) &
                                                   (spe_DF_clean['time'] >= time1) & (spe_DF_clean['time'] <= time2), ]

                # if spe_in_SPEG_one in peak_DM channel not empty
                if spe_in_SPEG_one.shape[0] > 0:
                    # sort by SNR
                    spe_in_SPEG_one.sort_values(by=['SNR'], ascending=False, inplace=True)
                    # get the TOA
                    time_one = float(spe_in_SPEG_one.iloc[0, 2])

                    # spes in this DM channel in SPEG_two
                    spe_in_SPEG_two = spe_DF_clean.loc[(spe_DF_clean['DM'] == key_peak_DM) &
                                                       (spe_DF_clean['time'] >= time3) &
                                                       (spe_DF_clean['time'] <= time4), ]
                    # if spe_in_SPEG_two in peak_DM channel not empty
                    if spe_in_SPEG_two.shape[0] > 0:
                        spe_in_SPEG_two.sort_values(by=['SNR'], ascending=False, inplace=True)
                        # get the TOA
                        time_two = float(spe_in_SPEG_two.iloc[0, 2])

                        # this time range of current SPEG
                        time5 = float(ano_SPEG.min_time)
                        time6 = float(ano_SPEG.max_time)

                        # spes in this DM channel in current Clu
                        spe_in_ano_SPEG = spe_DF_clean.loc[(spe_DF_clean['DM'] == key_peak_DM) &
                                                           (spe_DF_clean['time'] >= time5) &
                                                           (spe_DF_clean['time'] <= time6), ]

                        spe_in_ano_SPEG.sort_values(by=['SNR'], ascending=False, inplace=True)
                        # get the TOA
                        time_three = float(spe_in_ano_SPEG.iloc[0, 2])

                        # all three TOAs exist
                        TOAs = [time_one, time_two, time_three]
                        TOAs.sort()

                        diff1 = TOAs[1] - TOAs[0]
                        diff2 = TOAs[2] - TOAs[1]
                        res1 = diff1 / cur_period % 1
                        res2 = diff2 / cur_period % 1
                        
                        # both good periodicity
                        if (res1 < tolerance or res1 > 1 - tolerance) and (res2 < tolerance or res2 > 1 - tolerance):
                            astro_pulse = "YES"
                        # one of the two is good
                        elif (res1 < tolerance or res1 > 1 - tolerance) or (res2 < tolerance or res2 > 1 - tolerance):
                            astro_pulse = "MAY"
                            print "TOAs: ", TOAs
                        else:
                            astro_pulse = "NO"

                output_str = str(SPEG_rank) + ',' + astro_pulse
                verified_result_fp.write(output_str + '\n')
        verified_result_fp.close()
        error_log_fp.close()

    elif str_input3 == "some":
        astro_pulse = "NO"

        str_input4 = raw_input("which SPEG(s) do you want to verify? ")

        other_SPEGs = map(int, str_input4.split(','))
        other_SPEGs.sort()
        # sort in ascending order
        print "SPEGs included: ", other_SPEGs

        verified_result = 'verifiedPeriodsSome' + str(tolerance) + '-' + cur_parent_dir + '-' + cur_dir + '-' + str_input \
                         + '_' + str_input4 + SPEG_file + '.txt'
        verified_result_fp = open(verified_result, 'w')

        for ano_SPEG_rank in other_SPEGs:
            # verify other SPEGs one by one
            ano_SPEG = SPEG_DF.loc[SPEG_DF.SPEG_rank == ano_SPEG_rank]

            # python index starts from 0
            if ano_SPEG_rank not in SPEGs_included_ranks:
                # verify the known period
                key_peak_DM = float(ano_SPEG.peak_DM)
                print "key_peak_DM: ", key_peak_DM

                # get spes in SPEG one in key_peak_DM channel
                spe_in_SPEG_one = spe_DF_clean.loc[(spe_DF_clean['DM'] == key_peak_DM) &
                                                   (spe_DF_clean['time'] >= time1) & (spe_DF_clean['time'] <= time2),]

                print spe_in_SPEG_one

                # if spe_in_SPEG_one in key_peak_DM channel not empty
                if spe_in_SPEG_one.shape[0] > 0:
                    spe_in_SPEG_one.sort_values(by=['SNR'], ascending=False, inplace=True)
                    # get the time
                    time_one = float(spe_in_SPEG_one.iloc[0, 2])

                    # spes in this DM channel in SPEG_two
                    spe_in_SPEG_two = spe_DF_clean.loc[(spe_DF_clean['DM'] == key_peak_DM) &
                                                 (spe_DF_clean['time'] >= time3) & (spe_DF_clean['time'] <= time4),]
                    # if spe_in_SPEG_two not empty
                    if spe_in_SPEG_two.shape[0] > 0:
                        spe_in_SPEG_two.sort_values(by=['SNR'], ascending=False, inplace=True)
                        # get the TOA
                        time_two = float(spe_in_SPEG_two.iloc[0, 2])

                        # this time range of current SPEG
                        time5 = float(ano_SPEG.min_time)
                        time6 = float(ano_SPEG.max_time)

                        # spes in key_peak_DM  channel in current SPEG
                        spe_in_ano_SPEG = spe_DF_clean.loc[(spe_DF_clean['DM'] == key_peak_DM) &
                                                           (spe_DF_clean['time'] >= time5) &
                                                           (spe_DF_clean['time'] <= time6), ]

                        spe_in_ano_SPEG.sort_values(by=['SNR'], ascending=False, inplace=True)
                        # get the TOA
                        time_three = float(spe_in_ano_SPEG.iloc[0, 2])

                        # all there TOAs exist
                        TOAs = [time_one, time_two, time_three]
                        TOAs.sort()

                        diff1 = TOAs[1] - TOAs[0]
                        diff2 = TOAs[2] - TOAs[1]
                        res1 = diff1 / cur_period % 1
                        res2 = diff2 / cur_period % 1
                        print res1, res2

                        if (res1 < tolerance or res1 > 1 - tolerance) and (res2 < tolerance or res2 > 1 - tolerance):
                            astro_pulse = "YES"
                        elif (res1 < tolerance or res1 > 1 - tolerance) or (res2 < tolerance or res2 > 1 - tolerance):
                            astro_pulse = "MAY"
                            print "TOAs: ", TOAs
                        # else no

                output_str = str(ano_SPEG_rank) + ',' + astro_pulse
                verified_result_fp.write(output_str + '\n')
            else:
                exit("Choose another SPEG to verify!")
        verified_result_fp.close()
        error_log_fp.close()
    else:
        exit("You should input all/some ONLY!")