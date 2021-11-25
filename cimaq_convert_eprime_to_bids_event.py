#!/usr/bin/env python
# encoding: utf-8

import os
import re
import sys

import argparse
import glob
import logging
from numpy import nan as NaN
import numpy as np
import pandas as pd
import shutil
import zipfile


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="",
        epilog="""
        Convert behavioural data from cimaq to bids format
        """)

    parser.add_argument("in_dir",
                        help="Folder with all zip files.")

    parser.add_argument("out_dir",
                        help='Output folder - if doesn\'t exist it'
                             ' will be created.')

    parser.add_argument('--log_level', default='WARNING',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Log level of the logging class.')
    return parser


def get_all_ids(in_folder):
    """ List all ZipFile and get all IDs
    Parameters:
    ----------
    iFolder: string (input folder)

    Return:
    ----------
    ids: list of tuple (behavioral ID, IRM ID)
    """
    if not os.path.exists(in_folder):
        sys.exit('This folder doesn\'t exist: {}'.format(in_folder))
        return
    ids = []
    allZipFiles = glob.glob(os.path.join(in_folder, '*.zip'))
    for currZipFile in np.sort(allZipFiles):
        currZipFile = os.path.basename(currZipFile)
        ids.append((currZipFile.split('_')[0], currZipFile.split('_')[1]))

    if not ids:
        sys.exit('This folder doesn\'t contain any zip files.')
        return
    else:
        return ids


def set_subject_data(bID, iFolder, oFolder):
    """
    Parameters:
    ----------
    bID: string (PSCID used to identify participants during data collection)
    datadir: string (input folder)
    oFolder: string (output folder)

    Return:
    ----------
    sub_files: list (three input files)
    """
    logging.debug('Subject PSCID": {}'.format(bID))

    # prefix = ['Output-Responses-Encoding_CIMAQ_*',
    #           'Onset-Event-Encoding_CIMAQ_*',
    #           'Output_Retrieval_CIMAQ_*']

    prefix = ['Output-Responses-Encoding_*',
              'Onset-Event-Encoding_*',
              'Output_Retrieval_*']

    sub_files = []
    s_dir = glob.glob(os.path.join(iFolder, bID+'*task.zip'))
    if len(s_dir) != 1:
        logging.error(' Multiple directories match \
                       this subject PSCID: {}'.format(bID))
    else:
        s_path = os.path.join(oFolder, bID+'*')
        s_out = glob.glob(s_path)
        if not s_out:
            z_ref = zipfile.ZipFile(s_dir[0], 'r')
            z_ref.extractall(oFolder)
            z_ref.close()
            s_out = glob.glob(s_path)

        if len(s_out) == 1:
            s_out = s_out[0]
            for nPrefix in prefix:
                file = glob.glob(os.path.join(s_out, nPrefix))
                if len(file) == 1:
                    sub_files.append(file[0])
                elif len(file) == 0:
                    logging.error("No file with prefix {} "
                                  "found".format(nPrefix))
                else:
                    logging.error('Multiple files found for {} '
                                  'with prefix {} wih files {}'.format(bID,
                                                                       nPrefix,
                                                                       file))

        else:
            logging.error('Multiple folders found for {} found {}'.format(bID,
                                                                          s_out))

    return sub_files


def cleanMain(mainFile):
    """
    Parameters:
    ----------
    mainFile: pandas object

    Return:
    ----------
    mainFile: pandas object
    """
    # remove first three junk rows (blank trials): CTL0, Enc00 and ENc000
    mainFile.drop([0, 1, 2], axis=0, inplace=True)
    # re-label columns
    mainFile.rename(columns={'TrialNumber': 'trial_number',
                             'Category': 'trial_type',
                             'OldNumber': 'stim_id',
                             'CorrectSource': 'position_correct',
                             'Stim_RESP': 'response',
                             'Stim_RT': 'response_time'}, inplace=True)
    # remove redundant columns
    mainFile.drop(['TrialCode', 'Stim_ACC'], axis=1, inplace=True)
    # re-order columns
    cols = ['trial_number', 'trial_type', 'response', 'response_time',
            'stim_id', 'position_correct']
    mainFile = mainFile[cols]
    # change in-scan reaction time from ms to s
    mainFile['response_time'] = mainFile['response_time'].astype('float64',
                                                                 copy=False)
    mainFile['response_time'] = mainFile['response_time'].div(1000)
    # insert new columns
    colNames = ['onset', 'duration', 'offset', 'stim_file', 'stim_category',
                'stim_name', 'recognition_accuracy',
                'recognition_responsetime', 'position_response',
                'position_accuracy', 'position_responsetime']
    dtype = [NaN, NaN, NaN, 'None', 'None', 'None', -1, NaN, -1, -1, NaN]
    colIndex = [0, 1, 2, 8, 9, 10, 11, 12, 14, 15, 16]
    for i in range(0, 11):
        mainFile.insert(loc=colIndex[i],
                        column=colNames[i],
                        value=dtype[i],
                        allow_duplicates=True)
    return mainFile  # modified in-place


def cleanOnsets(onsets):
    """
    Description:
        Label columns and remove first six junk rows
        (3 junk trials; 2 rows per trial).

    Parameters:
    ----------
    onsets: pandas object

    Return:
    ----------
    onsets: pandas object
    """
    # add column headers
    onsets.columns = ["TrialNum", "Condition", "TrialNum_perCondi",
                      "ImageID", "Trial_part", "onsetSec", "durationSec"]
    onsets.drop([0, 1, 2, 3, 4, 5], axis=0, inplace=True)
    return onsets


def cleanRetriev(ret):
    """
    Parameters:
    ----------
    ret: pandas object

    Return:
    ----------
    ret: pandas object
    """
    # Change column headers
    ret.rename(columns={'category': 'old_new',
                        'Stim': 'stim_file',
                        'OldNumber': 'stim_id',
                        'Recognition_ACC': 'recognition_accuracy',
                        'Recognition_RESP': 'recognition_response',
                        'Recognition_RT': 'recognition_responsetime',
                        'Spatial_RESP': 'position_response',
                        'Spatial_RT': 'position_responsetime',
                        'Spatial_ACC(à corriger voir output-encodage)': 'position_accuracy'},
               inplace=True)
    # re-order columns
    cols = ['old_new', 'stim_file', 'stim_id', 'recognition_response',
            'recognition_accuracy', 'recognition_responsetime',
            'position_response', 'position_accuracy', 'position_responsetime']
    ret = ret[cols]
    # Transform reaction time columns from ms to s
    ret[['recognition_responsetime']] = ret[['recognition_responsetime']].astype('float64', copy=False)  # string is object in pandas, str in Python
    ret[['position_responsetime']] = ret[['position_responsetime']].astype('float64', copy=False)
    ret['recognition_responsetime'] = ret['recognition_responsetime'].div(1000)
    ret['position_responsetime'] = ret['position_responsetime'].div(1000)
    # Clean up eprime programming mistake: replace position_response and position_responsetime values
    # with NaN if subject perceived image as 'new' (the image was not probed for position).
    # There should be no response or RT value there, values were carried over from previous trial (not reset in eprime)
    # CONFIRMED w Isabel: subject must give a position answer when probed (image considered OLD) before eprime moves to the next trial.
    i = ret[ret['recognition_response'] == 2].index
    ret.loc[i, 'position_responsetime'] = NaN
    ret.loc[i, 'position_response'] = -1
    # clean up eprime mistake (change Old67 condition ('old_new') from New to OLD)
    q = ret[ret['stim_id'] == 'Old67'].index
    ret.loc[q, 'old_new'] = 'OLD'
    # insert new columns
    colNames = ['trial_number', 'stim_category', 'stim_name',
                'recognition_performance', 'position_correct']
    dtype = [-1, 'None', 'None', 'None', -1]
    colIndex = [0, 4, 5, 9, 10]
    for j in range(0, 5):
        ret.insert(loc=colIndex[j], column=colNames[j], value=dtype[j],
                   allow_duplicates=True)
    # Extract info and fill trial_number, stim_category and stim_name columns
    k = ret.index
    ret.loc[k, 'trial_number'] = k+1
    # format: category_imageName.bmp w some space, _ and - in image names
    stimInfo = ret.loc[k, 'stim_file']
    for s in k:
        ret.loc[s, 'stim_category'] = re.findall('(.+?)_', stimInfo[s])[0]
        ret.loc[s, 'stim_name'] = re.findall('_(.+?)[.]', stimInfo[s])[0]
    # Fill recognition_performance column based on actual and perceived novelty
    m = ret[ret['old_new'] == 'OLD'].index.intersection(ret[ret['recognition_accuracy'] == 1].index)
    ret.loc[m, 'recognition_performance'] = 'Hit'
    n = ret[ret['old_new'] == 'OLD'].index.intersection(ret[ret['recognition_accuracy'] == 0].index)
    ret.loc[n, 'recognition_performance'] = 'Miss'
    o = ret[ret['old_new'] == 'New'].index.intersection(ret[ret['recognition_accuracy'] == 1].index)
    ret.loc[o, 'recognition_performance'] = 'CR'
    p = ret[ret['old_new'] == 'New'].index.intersection(ret[ret['recognition_accuracy'] == 0].index)
    ret.loc[p, 'recognition_performance'] = 'FA'
    # return cleaned up input Dataframe
    return ret


def addOnsets(main, enc):
    """
    Parameters:
    ----------
    main:
    enc: pandas objects

    Return:
    ----------
    main: pandas object
    """

    # make main file indexable by trial number:
    main.set_index('trial_number', inplace=True)
    # copy trial onset and offset times from enc into main
    # note: fixation's onset time is the trial task's offset time
    for i in enc.index:
        trialNum = enc.loc[i, 'TrialNum']
        if enc.loc[i, 'Trial_part'] == 'Fixation':
            main.loc[trialNum, 'offset'] = enc.loc[i, 'onsetSec']
        else:
            main.loc[trialNum, 'onset'] = enc.loc[i, 'onsetSec']
    # Calculate trial duration time from onset and offset times
    main['duration'] = main['offset']-main['onset']
    # reset main's searchable index to default
    main.reset_index(level=None, drop=False, inplace=True)
    return main


def addPostScan(main, ret):
    """
    Parameters:
    ----------
    main: panda object
    ret: panda object

    Return:
    ----------
    mainMerged: pandas object
    """
    # split main's rows (trials) into sublist based on Condition
    mainEnc = main[main['trial_type'] == 'Enc'].copy()
    mainCTL = main[main['trial_type'] == 'CTL'].copy()
    # make mainEnc indexable by picture id
    mainEnc.set_index('stim_id', inplace=True)
    # import post-scan data from ret into mainEnc
    for i in ret[ret['old_new'] == 'OLD'].index:
        stimID = ret.loc[i, 'stim_id']
        mainEnc.loc[stimID, 'stim_category'] = ret.loc[i, 'stim_category']
        mainEnc.loc[stimID, 'stim_name'] = ret.loc[i, 'stim_name']
        mainEnc.loc[stimID, 'recognition_accuracy'] = ret.loc[i, 'recognition_accuracy']
        mainEnc.loc[stimID, 'recognition_responsetime'] = ret.loc[i, 'recognition_responsetime']
        mainEnc.loc[stimID, 'position_response'] = ret.loc[i, 'position_response']
        mainEnc.loc[stimID, 'position_responsetime'] = ret.loc[i, 'position_responsetime']
    # calculate post-scan source (position) accuracy;
    #  -1 = control task; 0 = missed trial; 1 = wrong source (image recognized but wrong quadrant remembered);
    # 2 = image recognized with correct source
    mainEnc['position_accuracy'] = 0
    for j in mainEnc[mainEnc['recognition_accuracy'] == 1].index:
        if mainEnc.loc[j, 'position_correct'] == mainEnc.loc[j, 'position_response']:
            mainEnc.loc[j, 'position_accuracy'] = 2
        else:
            mainEnc.loc[j, 'position_accuracy'] = 1
    # import source accuracy info from mainEnc into ret (in-place)
    for i in ret[ret['old_new'] == 'OLD'].index:
        picID = ret.loc[i, 'stim_id']
        ret.loc[i, 'position_correct'] = mainEnc.loc[picID, 'position_correct']
        ret.loc[i, 'position_accuracy'] = mainEnc.loc[picID,
                                                      'position_accuracy']
    # reset mainEnc searchable index to default
    # and re-order columns to match order in mainCTL
    mainEnc.reset_index(level=None, drop=False, inplace=True)
    cols = ['trial_number', 'onset', 'duration', 'offset', 'trial_type',
            'response', 'response_time', 'stim_id', 'stim_file',
            'stim_category', 'stim_name', 'recognition_accuracy',
            'recognition_responsetime', 'position_correct',
            'position_response', 'position_accuracy', 'position_responsetime']
    mainEnc = mainEnc[cols]
    # Re-merge mainEnc and mainCTL and re-order by trial number
    mainMerged = mainEnc.append(mainCTL, ignore_index=True)
    mainMerged.sort_values('trial_number', axis=0, ascending=True,
                           inplace=True)
    return mainMerged


def extract_taskFile(bID, sID, file_list, output):
    """
    Parameters:
    ----------
    bID: string (subject PSCID, id used during data collection)
    sID: string (subject DCCID, id used in Loris)
    file_list: list (three input files)
    output: string (output Folder)

    Return:
    ----------
    None
    """
    # import data from three text files into pandas DataFrames
    encMain = pd.read_csv(file_list[0], sep='\t')
    manualEdits = ['3303819', '5477234', '6417837', '7674650']
    if bID in manualEdits:
        encOnsets = pd.read_csv(file_list[1], sep='\t', header=None)
    else:
        encOnsets = pd.read_fwf(file_list[1], infer_nrows=210,
                                delim_whitespace=True,
                                header=None)
    retriev = pd.read_csv(file_list[2], sep='\t', encoding='ISO-8859-1')
    # clean up each file
    encMain = cleanMain(encMain)
    #print(encOnsets)
    encOnsets = cleanOnsets(encOnsets)
    retriev = cleanRetriev(retriev)
    # import onset times from encOnset into encMain
    encMain = addOnsets(encMain, encOnsets)
    # import post-scan performance data from retriev into encMain
    encMain = addPostScan(encMain, retriev)
    # export encMain and retriev into tsv files (output directorty)
    encMain.to_csv(output+'/sub-'+bID+'_ses-sID_task-memory_events.tsv',
                   sep='\t', header=True, index=False)
    retriev.to_csv(output+'/PostScanBehav_pscid'+bID+'_dccid'+sID+'.tsv',
                   sep='\t', header=True, index=False)


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)

    oFolder = args.out_dir
    iFolder = args.in_dir

    # Create oFolder if not exists
    if not os.path.exists(oFolder):
        os.mkdir(oFolder)

    all_ids = get_all_ids(iFolder)
    # Create tmp folder to temporaly store unziped files
    tmpFolder = os.path.join(oFolder, 'tmp')
    if not os.path.exists(tmpFolder):
        os.mkdir(tmpFolder)

    # Create taskFiles folder where all output files will be saved
    fileFolder = os.path.join(oFolder, 'taskfiles')
    if not os.path.exists(fileFolder):
        os.mkdir(fileFolder)

    # loop over zip files
    for (idBEH, idMRI) in all_ids:
        print("Running {}-{}".format(idBEH, idMRI))
        s_files = set_subject_data(idBEH, iFolder, tmpFolder)
        if(len(s_files) == 3):
            extract_taskFile(idBEH, idMRI, s_files, fileFolder)
            shutil.rmtree(tmpFolder, ignore_errors=True)
        else:
            logging.info('missing files for subject ({},{})'.format(idBEH,
                                                                    idMRI))


if __name__ == '__main__':
    sys.exit(main())
