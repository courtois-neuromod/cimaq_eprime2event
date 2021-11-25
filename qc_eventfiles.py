#!/usr/bin/env python
# encoding: utf-8

import os
import sys

import argparse
import glob
import numpy as np
from numpy import nan as NaN
import pandas as pd


def get_arguments():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="",
        epilog="""
        Perform quality check on event files outputed by
        cimaq_convert_eprime_to_bids.py
        """)

    parser.add_argument(
        "-d", "--idir",
        required=True,
        help="Directory that contains event files (.tsv)")

    parser.add_argument(
        "-o", "--odir",
        required=True,
        help="Output directory - if doesn\'t exist it will be created.")

    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    else:
        return args


def get_all_tsvs(in_dir):
    '''
    Return sorted list of paths to in-scan event files outputed by
    cimaq_convert_eprime_to_bids_event.py
    (file format: sub-subid_ses-sesid_task-memory_events.tsv)
    '''
    if not os.path.exists(in_dir):
        sys.exit('This folder doesn\'t exist: {}'.format(in_dir))
        return

    tsv_list = sorted(glob.glob(os.path.join(in_dir, 'sub*task-memory_events.tsv')))

    return tsv_list


def qc_tsv_file(tsv_path):
    '''
    This is where the magic happens!!
    '''
    try:
        # load .tsv as pandas dataframe
        dframe = pd.read_csv(tsv_path, sep = '\t')

        qc_summary = ''

        # check that number of rows = 120 - 3 (removed first 3 "blank" trials)
        if not (dframe.shape[0]==117):
            qc_summary += 'Row count is not 117; '

        # check for missing / mislabelled columns
        missing_cols = []
        col_list = ['trial_number', 'onset', 'duration', 'offset', 'trial_type',
                    'response',  'response_time', 'stim_id', 'stim_file', 'stim_category', 'stim_name',
                    'recognition_accuracy', 'recognition_responsetime', 'position_correct',
                    'position_response', 'position_accuracy', 'position_responsetime']
        for col in col_list:
            if col not in dframe.columns:
                missing_cols.append(col)
        if len(missing_cols) > 0:
            qc_summary += 'Missing columns: '
            for i in range(len(missing_cols)):
                qc_summary += missing_cols[i] + ', '
            qc_summary += '; '

        # check range  and order of trial numbers
        trial_num = np.array_equal(dframe['trial_number'].tolist(), range(4, 121))
        if not trial_num:
            qc_summary += 'Invalid trial numbers; '

        # check for missing onset times
        if np.sum(np.isnan(dframe['onset'].tolist())) > 0:
            qc_summary += 'Onset times contain NaN; '

        # check range of onset times
        onset_min = (np.min(dframe['onset']) > 10 and np.min(dframe['onset']) < 30)
        onset_max = (np.max(dframe['onset']) > 700 and np.max(dframe['onset']) < 750)
        if not (onset_min and onset_max):
            qc_summary += 'Invalid onset time range; '

        # check number of trials from each type (control vs image encoding)
        trial_types, trial_counts = np.unique(dframe['trial_type'], return_counts=True)
        check_ctl = trial_types[np.argmin(trial_counts)]=='CTL' and trial_counts[np.argmin(trial_counts)]==39
        check_enc = trial_types[np.argmax(trial_counts)]=='Enc' and trial_counts[np.argmax(trial_counts)]==78
        if not (check_ctl and check_enc):
            qc_summary += 'Invalid number of trials per condition; '

        # split dataframe per trial type
        dframe_enc = dframe[dframe['trial_type']=='Enc']
        dframe_ctl = dframe[dframe['trial_type']=='CTL']

        # Check assigment of stimulus names and categories from post-scan output file
        for cname in ['stim_name', 'stim_category']:
            if np.sum(dframe_enc[cname]=='None') > 0:
                qc_summary += 'Missing ' + cname + ' values; '
            if np.sum(dframe_ctl[cname]=='None') != dframe_ctl.shape[0]:
                qc_summary += cname + 'values assigned to CTL condition; '

        # Check recognition accuracy values
        if not np.sum(np.isin(dframe_enc['recognition_accuracy'], [0, 1])) == dframe_enc.shape[0]:
            qc_summary += 'Invalid recognition accuracy entries for Enc trials; '
        if not np.sum(dframe_ctl['recognition_accuracy']==-1) == dframe_ctl.shape[0]:
            qc_summary += 'Invalid recognition accuracy entries for CTL trials; '

        # Check recognition reaction time values
        rt_nan = np.sum(np.isnan(dframe_enc['recognition_responsetime']))
        rt_min = np.sum(dframe_enc['recognition_responsetime'] < 0.5)
        rt_max = np.sum(dframe_enc['recognition_responsetime'] > 12.0) > 0
        if rt_nan or rt_min or rt_max:
            qc_summary += 'Invalid or missing recognition reaction times check rt_nan = {}, rt_min = {}, rt_max = {}; '.format(rt_nan, rt_min, rt_max)
        if np.sum(np.isnan(dframe_ctl['recognition_responsetime'])) != dframe_ctl.shape[0]:
            qc_summary += 'Recognition reaction times misattributed to control trials; '

        # Check position correct entries
        if np.sum(np.isin(dframe['position_correct'], [5, 6, 8, 9])) < dframe.shape[0]:
            qc_summary += 'Missing correct position entries; '

        # Check position response, accuracy and reaction time entries
        if np.sum(np.isin(dframe['position_response'], [-1, 5, 6, 8, 9])) < dframe.shape[0]:
            qc_summary += 'Invalid position response entries; '
        if np.sum(np.isin(dframe_enc['position_accuracy'], [0, 1, 2])) < dframe_enc.shape[0]:
            qc_summary += 'Invalid position accuracy entries for encoding trials; '
        # Check that no position response (CTL and missed trials)=> no position reaction time entered
        dframe_nopos = dframe[dframe['position_response']==-1]
        if np.sum(np.isnan(dframe_nopos['position_responsetime'])) != dframe_nopos.shape[0]:
            qc_summary += 'Position reaction time entered without position response entry; '
        if np.sum(np.isnan(dframe_nopos['position_responsetime'])) != np.sum(np.isnan(dframe['position_responsetime'])):
            qc_summary += 'Position reaction times missing despite given position response; '
        # Check position responses, accuracy and rt in control trials
        if np.sum(dframe_ctl['position_response'] == -1) < dframe_ctl.shape[0]:
            qc_summary += 'Position responses misattributed to control trials; '
        if np.sum(dframe_ctl['position_accuracy'] == -1) < dframe_ctl.shape[0]:
            qc_summary += 'Position accuracy entries misattributed to control trials; '
        if np.sum(np.isnan(dframe_ctl['position_responsetime'])) < dframe_ctl.shape[0]:
            qc_summary += 'Position reaction time entered for control trials; '

        # Check hit and miss encoding trials for esponses, accuracy and rt in control trials
        dframe_hitenc = dframe_enc[dframe_enc['recognition_accuracy']==1]
        dframe_missenc = dframe_enc[dframe_enc['recognition_accuracy']==0]
        if np.sum(np.isin(dframe_hitenc['position_response'], [5, 6, 8, 9])) < dframe_hitenc.shape[0]:
            qc_summary += 'Hit trials missing position responses; '
        if np.sum(np.isin(dframe_hitenc['position_accuracy'], [1, 2])) < dframe_hitenc.shape[0]:
            qc_summary += 'Hit trials with invalid position accuracy responses; '
        if np.sum(np.isnan(dframe_hitenc['position_responsetime'])) > 0:
            qc_summary += 'Hit trials with missing position response times; '

        if np.sum(dframe_missenc['position_response'] == -1) < dframe_missenc.shape[0]:
            qc_summary += 'Missed trials with position responses; '
        if np.sum(dframe_missenc['position_accuracy']==0) < dframe_missenc.shape[0]:
            qc_summary += 'Missed trials with invalid position accuracy entries; '
        if np.sum(np.isnan(dframe_missenc['position_responsetime'])) < dframe_missenc.shape[0]:
            qc_summary += 'Position reaction time entered for missed trials; '

        # Check that correctly positioned trials are well identified
        dframe_corpos = dframe_enc[dframe_enc['position_accuracy']==2]
        if np.sum(np.equal(dframe_corpos['position_correct'], dframe_corpos['position_response'])) != dframe_corpos.shape[0]:
            qc_summary += 'Incorrect labelling of trials with correctly remembered positions; '
        dframe_incorpos = dframe_enc[dframe_enc['position_accuracy']==1]
        if np.sum(np.equal(dframe_incorpos['position_correct'], dframe_incorpos['position_response'])) > 0:
            qc_summary += 'Incorrect labelling of trials with incorrectly remembered positions; '
    except:
        qc_summary = 'Failed to process subject'

    return qc_summary


def main():
    '''
    Script performs quality check on event files
    outputed by cimaq_convert_eprime_to_bids.py script
    and saves QC eport (eventfiles_QCreport.txt) in
    output directory
    '''
    args = get_arguments()

    input_dir = args.idir
    output_dir = args.odir

    # Create output_dir if not exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    tsv_list = get_all_tsvs(input_dir)

    # text file that documents trials with no response, per run
    qc_report = open(os.path.join(output_dir, 'eventfiles_QCreport.txt'), 'w+')

    for tsv_path in tsv_list:
        sub_id = os.path.basename(tsv_path).split('_')[0].split('-')[1]
        qc_rep = qc_tsv_file(tsv_path)
        if len(qc_rep) == 0:
            qc_rep = 'All good'
        qc_report.write('subject ' + sub_id + ': ' + qc_rep + '\n')

    qc_report.close()


if __name__ == '__main__':
    sys.exit(main())
