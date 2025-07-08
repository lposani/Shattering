import os.path
import pickle
import numpy as np
from math import floor
from brainbox.io.one import SpikeSortingLoader
from brainbox.io.one import SessionLoader
from brainbox.singlecell import calculate_peths
from one.api import ONE
import os
import shutil
from source import mkdir
from typing import Optional
import itertools
from iblatlas.regions import BrainRegions
import pdb
from tqdm import tqdm
import pandas as pd
from source.utilities import *
from hashlib import md5


def find_good_sessions(sessions, variables, filters, min_trials, min_n_neurons=4):
    good_sessions = []
    for session in sessions:
        conditioned_trials = divide_into_conditions(session, variables, filters=filters)
        enough_trials = [len(conditioned_trials[key]) > min_trials for key in conditioned_trials]
        # print(enough_trials)
        if np.nanmean(enough_trials) == 1.0 and session['n_neurons'] > min_n_neurons:
            good_sessions.append(session)
    return good_sessions


def download_sessions(region: str,
                      pre_time: float = 1.0,
                      post_time: float = 3.0,
                      bin_size: float = 0.01,
                      good_neurons: bool = False,
                      temp_dir: Optional[str] = None,
                      cache_sessions: bool = False,
                      uuids_list_path: Optional[str] = None):
    """

    :param region: brain region in IBL notation
    :param pre_time: time (seconds) before stimulus onset
    :param post_time: time (seconds) after stimulus onset
    :param bin_size: size (seconds) of each time bin of spike counts
    :param good_neurons: if True, only neurons flagged as good neurons will be downloaded
    :param temp_dir: directory where temporary ONE API files are downloaded. This directory is automatically deleted after completing the download.
    :param cache_sessions: if True, sessions after download are cached in the datasets/IBL/cache directory
    :param uuids_list_path: if specified, only neruons in the list will be loaded

    :return: list of sessions
    """
    if temp_dir is None:
        temp_dir = './datasets/IBL/temp'
    if uuids_list_path == 'old':
        uuids_str = md5(str.encode(str(None))).hexdigest()
    elif uuids_list_path is not None:
        uuids_str = md5(str.encode(str(uuids_list_path))).hexdigest()
    else:
        uuids_str = 'no_selection'
    mkdir(temp_dir)
    mkdir('./datasets/IBL/cache')
    one = ONE(base_url='https://openalyx.internationalbrainlab.org', password='international', cache_dir=temp_dir)

    # get data
    br = BrainRegions()
    ses = one.alyx.rest('insertions', 'list', atlas_acronym=region)
    pids = [i['id'] for i in ses]  # ID of each specific probe

    # Selected neurons:
    if uuids_list_path is not None and uuids_list_path != 'old':
        uuids_selected = pd.read_csv(uuids_list_path)['uuids'].values

    sessions = []
    # load sessions
    for PID in pids:
        filename = f'./datasets/IBL/cache/{region}_{PID}_{pre_time}_{post_time}_{bin_size}_{good_neurons}_{uuids_str}.pck'

        if os.path.exists(filename) and cache_sessions:
            session = pickle.load(open(filename, 'rb'))
            if session['n_neurons']:
                sessions.append(session)
        else:
            print("Loading", PID)
            sl = SpikeSortingLoader(pid=PID, one=one)
            try:
                spikes, clusters, channels = sl.load_spike_sorting()
            except:
                print(f"something wrong with sl.load_spike_sorting(); but we ignore...")
            clusters = sl.merge_clusters(spikes, clusters, channels)

            # sub-select region neurons
            if clusters is not None:
                # we use Swanson mapping instead of clusters['acronym'] for several reasons
                #   cannot use clusters['acronym'] in region
                #     to distinguish region VISp from region VISpm
                #   cannot use clusters['acronym'] == region
                #     because clusters['acronym'] have layer information (e.g. ORBm2/3)

                swanson_names = [br.id2acronym(id, mapping='Swanson') for id in clusters['atlas_id']]

                if uuids_list_path is not None:
                    print("SeLECTING NEURONSSSSS")
                    selected_neurons_index = [i for i in range(len(clusters['uuids'])) if
                                              clusters['uuids'][i] in uuids_selected]
                    print(len(selected_neurons_index))
                else:
                    selected_neurons_index = [i for i in range(len(clusters['uuids']))]

                if good_neurons:
                    neurons = [i for i, acronym in enumerate(swanson_names)
                               if (region == acronym)
                               and (clusters['label'][i] == 1)
                               and (i in selected_neurons_index)]
                    neurons_left = np.array([clusters['x'][i] < 0. for i in neurons])
                else:
                    neurons = [i for i, acronym in enumerate(swanson_names)
                               if (region == acronym)
                               and (i in selected_neurons_index)]
                    neurons_left = np.array([clusters['x'][i] < 0. for i in neurons])

                if len(neurons) == 0:
                    print(f"no neurons of {region} in this pid with {np.unique(swanson_names, return_counts=True)}.")
                    session = {'n_neurons': 0}
                    if cache_sessions:
                        print('SAVING IT')
                        pickle.dump(session, open(filename, 'wb'))
                    continue

                # let's download trial info
                EID, _ = one.pid2eid(PID)
                # trials = one.load_object(EID, 'trials')
                sessionONE = SessionLoader(eid=EID, one=one)
                try:
                    sessionONE.load_motion_energy()
                except:
                    continue
                sessionONE.load_trials()
                trials = sessionONE.trials

                lick_times = one.load_object(EID, 'licks', collection='alf')['times']
                left_whisk = sessionONE.motion_energy['leftCamera']
                right_whisk = sessionONE.motion_energy['rightCamera']

                # choose contrast
                position = (trials['contrastLeft'] >= 0).astype(float) - (trials['contrastRight'] >= 0).astype(float)
                contrast = trials['contrastRight']
                contrast[np.isnan(trials['contrastLeft']) == 0] = trials['contrastLeft'][
                    np.isnan(trials['contrastLeft']) == 0]
                choice = trials['choice']
                reward = trials['rewardVolume']
                context = trials['probabilityLeft']
                times = trials['stimOn_times']
                first_movement_intrial = trials['firstMovement_times'] - trials['stimOn_times']
                response_time_intrial = trials['response_times'] - trials['stimOn_times']

                # compute spike counts
                peth, spike_counts = calculate_peths(
                    spikes.times, spikes.clusters, neurons,
                    times, pre_time=pre_time, post_time=post_time, bin_size=bin_size, smoothing=0)

                ntrials = spike_counts.shape[0]
                nneurons = spike_counts.shape[1]
                nbins_pertrial = spike_counts.shape[2]
                time_intrial = np.linspace(-pre_time, post_time, nbins_pertrial, endpoint=False)

                session = {
                    'n_neurons': nneurons,
                    'region': region,
                    'left_hemi': neurons_left,
                    'PID': PID,
                    'raster': np.hstack([spike_counts[i] for i in range(ntrials)]).T,
                    'trial': np.repeat(np.arange(ntrials), nbins_pertrial),
                    'time_from_onset': np.hstack(ntrials * list(time_intrial)),
                    'context': np.repeat(context, nbins_pertrial),
                    'position': np.repeat(position, nbins_pertrial),
                    'choice': np.repeat(choice, nbins_pertrial),
                    'contrast': np.repeat(contrast, nbins_pertrial),
                    'reward': np.repeat(reward, nbins_pertrial),
                    'first_movement_time': np.repeat(first_movement_intrial, nbins_pertrial),
                    'response_time': np.repeat(response_time_intrial, nbins_pertrial),
                }

                # Compute movement and first movement time, TODO: is there a more elegant way?
                movement = []
                licks = []
                whisking_power_left = []
                whisking_power_right = []
                whisking_powerall = []
                time_from_movement = []

                for i in range(ntrials):
                    m = np.zeros(nbins_pertrial)

                    if np.isnan(first_movement_intrial[i]) == 0:
                        tfm = time_intrial - bin_size * (int(first_movement_intrial[i] / bin_size))
                        time_from_movement.append(np.around(tfm, 4))
                        movetime = floor((first_movement_intrial[i] + pre_time) / bin_size)
                        m[movetime:] = 1
                    else:
                        time_from_movement.append(np.nan * np.zeros(nbins_pertrial))
                    movement.append(m)

                    trial_timebins = np.linspace(times[i] - pre_time, times[i] + post_time, nbins_pertrial + 1)
                    licks.append(divide_into_bins(lick_times, trial_timebins))
                    whisking_power_left.append(divide_into_bins(left_whisk.times.values, trial_timebins,
                                                                left_whisk.whiskerMotionEnergy.values))
                    whisking_power_right.append(divide_into_bins(right_whisk.times.values, trial_timebins,
                                                                 right_whisk.whiskerMotionEnergy.values))

                movement = np.hstack(movement)
                time_from_movement = np.hstack(time_from_movement)
                session['movement'] = movement
                session['time_from_firstmove'] = time_from_movement

                session['lick'] = np.hstack(licks)
                session['whisking_left'] = np.hstack(whisking_power_left)
                session['whisking_right'] = np.hstack(whisking_power_right)

                response = []
                time_from_response = []
                for i in range(ntrials):
                    r = np.zeros(nbins_pertrial)
                    if np.isnan(response_time_intrial[i]) == 0:
                        tfr = time_intrial - bin_size * (int(response_time_intrial[i] / bin_size))
                        time_from_response.append(np.around(tfr, 4))
                        responsetime = floor((response_time_intrial[i] + pre_time) / bin_size)
                        r[responsetime:] = 1
                    else:
                        time_from_response.append(np.nan * np.zeros(nbins_pertrial))
                    response.append(r)
                response = np.hstack(response)
                time_from_response = np.hstack(time_from_response)
                session['response'] = response
                session['time_from_response'] = time_from_response

                sessions.append(session)
                print(filename)
                if cache_sessions:
                    print('SAVING IT')
                    pickle.dump(session, open(filename, 'wb'))
            else:
                print("No clusters here.")
                session = {'n_neurons': 0}
                if cache_sessions:
                    print('SAVING IT')
                    pickle.dump(session, open(filename, 'wb'))

    # Always remove temp folder contents
    shutil.rmtree(temp_dir)

    return sessions


def load_sessions(region: str,
                  variables: dict,
                  filters: Optional[list] = None,
                  good_neurons: bool = False,
                  cache_full_sessions: bool = False,
                  cache_name: Optional[str] = None,
                  uuids_list_path: Optional[str] = None):
    """
    :param region: brain region in IBL notation
    :param variables: list of dictionaries, one per variable, each containing a series of functions to select one specific value of that specific variable.
    :param filters: a list of masking functions whose intersection will be used to select the data.
    :param good_neurons: if True, only good neurons are downloaded
    :param cache_full_sessions: if True, the complete downloaded sessions are cached locally
    :param cache_name: if specified, the resulting sessions will be saved into a pickle file with the following file path: "datasets/IBL/trials/{cache_name}_{region}_{var1}_{var2}_..._{varN}_{good_neurons}".
    :param uuids_list_path: if specified, only neruons in the list will be loaded

    :return: list of session dictionaries with squeezed trials divided by condition.
    """

    mkdir('./datasets/IBL/trials')

    if cache_name is not None:
        filename = f'./datasets/IBL/trials/{cache_name}_{region}_' + '_'.join(
            list(variables.keys())) + f'_{good_neurons}.pck'
        # print(filename)
        if os.path.exists(filename):
            sessions = pickle.load(open(filename, 'rb'))
            return sessions

    if filters is None:
        filters = []

    sessions_raw = download_sessions(region, good_neurons=good_neurons, cache_sessions=cache_full_sessions)
    sessions = []
    all_condition_keys = [list(variables[key].keys()) for key in variables]
    condition_keys = list(itertools.product(*all_condition_keys))

    for session in tqdm(sessions_raw):
        # Compute filters mask on the session
        filters_mask = np.ones(len(session['trial']), dtype=bool)
        for filter_func in filters:
            mask = filter_func(session)
            filters_mask = filters_mask & mask

        # Compute all condition masks in the session
        condition_masks = {}
        for condition in condition_keys:
            condition_mask = np.ones(len(session['trial']), dtype=bool)
            for i, var_key in enumerate(variables):
                mask_func = variables[var_key][condition[i]]
                condition_mask = condition_mask & mask_func(session)
            condition_masks[condition] = condition_mask

        # Isolate individual trials and squeeze them
        conditioned_trials = {condition: [] for condition in condition_keys}
        trial_numbers = np.unique(session['trial'])

        for trial in trial_numbers:
            trial_mask = session['trial'] == trial

            # Check that trials have a unique condition
            _trial_lenghts = []
            for condition in condition_masks:
                mask = trial_mask & filters_mask & condition_masks[condition]
                _trial_lenghts.append(np.sum(mask))

            if np.sum(np.asarray(_trial_lenghts) > 0) > 1:
                raise ValueError("More than one condition was found within the same trial. Check variables to avoid "
                                 "overlapping conditions.")

            # Take the (only!) condition that works and squeeze it into a nice 1xN vector
            for condition in condition_masks:
                mask = trial_mask & filters_mask & condition_masks[condition]
                if np.sum(mask):
                    conditioned_raster = session['raster'][mask]
                    # convert to Hz assuming 0.01 seconds time bins, take the mean, add to conditioned_trials dict
                    conditioned_trials[condition].append(np.nanmean(conditioned_raster, 0) * 100)

        for condition in condition_keys:
            conditioned_trials[condition] = np.asarray(conditioned_trials[condition])

        # Save some meta data
        conditioned_trials["_meta_"] = {'neuron_left_hemi': session['left_hemi']}

        sessions.append(conditioned_trials)

        if cache_name is not None:
            pickle.dump(sessions, open(filename, 'wb'))

    return sessions


def divide_into_conditions(session, variables, filters, allow_multiple_conditions_pertrial=True):
    all_condition_keys = [list(variables[key].keys()) for key in variables]
    condition_keys = list(itertools.product(*all_condition_keys))
    condition_keys_string = [''.join(k) for k in condition_keys]

    # Compute filters mask on the session
    filters_mask = np.ones(len(session['trial']), dtype=bool)
    for filter_func in filters:
        mask = filter_func(session)
        filters_mask = filters_mask & mask

    # Compute all condition masks in the session
    condition_masks = {}
    for condition in condition_keys:
        condition_mask = np.ones(len(session['trial']), dtype=bool)
        for i, var_key in enumerate(variables):
            mask_func = variables[var_key][condition[i]]
            condition_mask = condition_mask & mask_func(session)
        condition_masks[condition] = condition_mask

    # Isolate individual trials and squeeze them
    conditioned_trials = {condition: [] for condition in condition_keys}

    if allow_multiple_conditions_pertrial:
        for condition in condition_keys:
            trial_numbers = np.unique(session['trial'][condition_masks[condition]])

            for trial in trial_numbers:
                trial_mask = session['trial'] == trial
                mask = trial_mask & filters_mask & condition_masks[condition]
                if np.sum(mask):
                    conditioned_raster = session['raster'][mask]
                    # convert to Hz assuming 0.1 seconds time bins, take the mean, add to conditioned_trials dict
                    conditioned_trials[condition].append(np.nanmean(conditioned_raster, 0) * 10)
    else:
        trial_numbers = np.unique(session['trial'])

        for trial in trial_numbers:
            trial_mask = session['trial'] == trial

            # Check that trials have a unique condition
            _trial_lenghts = []
            for condition in condition_masks:
                mask = trial_mask & filters_mask & condition_masks[condition]
                _trial_lenghts.append(np.sum(mask))

            if np.sum(np.asarray(_trial_lenghts) > 0) > 1:
                raise ValueError("More than one condition was found within the same trial. Check variables to avoid "
                                 "overlapping conditions.")

            # Take the (only!) condition that works and squeeze it into a nice 1xN vector
            for condition in condition_masks:
                mask = trial_mask & filters_mask & condition_masks[condition]
                if np.sum(mask):
                    conditioned_raster = session['raster'][mask]
                    # convert to Hz assuming 0.01 seconds time bins, take the mean, add to conditioned_trials dict
                    conditioned_trials[condition].append(np.nanmean(conditioned_raster, 0) * 100)

    for condition in condition_keys:
        conditioned_trials[condition] = np.asarray(conditioned_trials[condition])

    return conditioned_trials


def get_conditions_nt(region, neurons_path, min_trials, variables, filters):
    bin_size = 0.05
    pre_time = 0.5
    post_time = 3.0
    uuids_str = md5(str.encode(str(neurons_path))).hexdigest()
    filename = f'./datasets/IBL/cache/sessions_{region}_{pre_time}_{post_time}_{bin_size}_{uuids_str}.pck'

    if os.path.exists(filename):
        sessions = pickle.load(open(filename, 'rb'))
    else:
        sessions = download_sessions(region=region, uuids_list_path=neurons_path,
                                     bin_size=bin_size, pre_time=pre_time,
                                     post_time=post_time, cache_sessions=True)
        pickle.dump(sessions, open(filename, 'wb'))

    good_sessions = find_good_sessions(sessions, variables, filters, min_trials)
    results = []
    all_condition_keys = [list(variables[key].keys()) for key in variables]
    condition_keys = list(itertools.product(*all_condition_keys))

    for session in good_sessions:
        # Compute filters mask on the session
        filters_mask = np.ones(len(session['trial']), dtype=bool)
        for filter_func in filters:
            mask = filter_func(session)
            filters_mask = filters_mask & mask

        # Compute all condition masks in the session
        condition_masks = {}
        for condition in condition_keys:
            condition_mask = np.ones(len(session['trial']), dtype=bool)
            for i, var_key in enumerate(variables):
                mask_func = variables[var_key][condition[i]]
                condition_mask = condition_mask & mask_func(session)
            condition_masks[condition] = condition_mask

        n_spikes = {}
        n_timebins = {}
        for condition in condition_keys:
            n_spikes[condition] = np.sum(session['raster'][condition_masks[condition]], 0)
            n_timebins[condition] = np.sum(condition_masks[condition])

        results.append([n_spikes, n_timebins])
    return results


def get_conditioned_trials(region, variables, filters, min_trials, neurons_path=None, n_neurons_megapooled=None):
    bin_size = 0.05
    pre_time = 0.5
    post_time = 3.0
    uuids_str = md5(str.encode(str(neurons_path))).hexdigest()
    filename = f'./datasets/IBL/cache/sessions_{region}_{pre_time}_{post_time}_{bin_size}_{uuids_str}.pck'

    if os.path.exists(filename):
        sessions = pickle.load(open(filename, 'rb'))
    else:
        sessions = download_sessions(region=region, uuids_list_path=neurons_path,
                                     bin_size=bin_size, pre_time=pre_time,
                                     post_time=post_time, cache_sessions=True)
        pickle.dump(sessions, open(filename, 'wb'))

    good_sessions = find_good_sessions(sessions, variables, filters, min_trials)

    conditioned_trials = [divide_into_conditions(session, variables, filters=filters)
                          for session in good_sessions]

    n_n = np.sum([s['n_neurons'] for s in good_sessions])
    n_sessions = len(good_sessions)
    print(n_n)
    if n_neurons_megapooled is not None and n_n > 1:
        megapooling = int(n_neurons_megapooled / n_n) + 1
        good_sessions = megapooling * good_sessions
        conditioned_trials = [divide_into_conditions(session, variables, filters=filters)
                              for session in good_sessions]
    print(len(conditioned_trials), n_n, n_sessions)
    return conditioned_trials, n_n, n_sessions


def seperate_sessions_byHemisphere(sessions):
    """
    :sessions: returns of IBL.load_sessions(..)
    :return: Two (left and right) lists of session dictionaries with squeezed trials divided by condition.
    """
    left_sessions = [];
    right_sessions = []
    for si, data in enumerate(sessions):
        neuron_left_hemi = data['_meta_']['neuron_left_hemi']
        for hemi in ['left', 'right']:
            left_hemi = hemi == 'left'
            conditioned_trials = {}
            for k, v in data.items():
                if type(k) == str:  # avoid the key of '_meta_'
                    continue
                if len(v) == 0:  # no trials of this condition
                    conditioned_trials[k] = v
                else:
                    conditioned_trials[k] = v[:, neuron_left_hemi == left_hemi]
            if hemi == 'left':
                left_sessions.append(conditioned_trials)
            else:
                right_sessions.append(conditioned_trials)
    return left_sessions, right_sessions
