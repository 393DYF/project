"""
Command-line interface for running privacy evaluation with respect to the risk of linkability
"""

import json
import os


from os import mkdir, path
from numpy.random import choice, seed
from argparse import ArgumentParser
from pandas import DataFrame

from utils.datagen import load_s3_data_as_df, load_local_data_as_df
from utils.utils_ import json_numpy_serialzer
from utils.logging_ import LOGGER
from utils.constants import *

from feature_sets.independent_histograms import HistogramFeatureSet
from feature_sets.model_agnostic import NaiveFeatureSet, EnsembleFeatureSet
from feature_sets.bayes import CorrelationsFeatureSet

from sanitisation_techniques.sanitiser import SanitiserNHS

from generative_models.data_synthesiser import (IndependentHistogram,
                                                BayesianNet,
                                                PrivBayes)

from attack_models.mia_classifier import (MIAttackClassifierRandomForest,
                                          generate_mia_shadow_data,
                                          generate_mia_anon_data)

from warnings import simplefilter
simplefilter('ignore', category=FutureWarning)
simplefilter('ignore', category=DeprecationWarning)


cwd = os.path.dirname(__file__)

SEED = 42


def main():
    argparser = ArgumentParser()
    datasource = argparser.add_mutually_exclusive_group()
    datasource.add_argument('--s3name', '-S3', type=str, choices=['adult', 'census', 'credit', 'alarm', 'insurance'], help='Name of the dataset to run on')
    datasource.add_argument('--datapath', '-D', type=str, default=cwd, help='Relative path to cwd of a local data file')
    argparser.add_argument('--runconfig', '-RC', default='runconfig.json', type=str, help='Path relative to cwd of runconfig file')
    argparser.add_argument('--outdir', '-O', default='tests', type=str, help='Path relative to cwd for storing output files')
    args = argparser.parse_args()

    # Load runconfig
    with open(path.join(cwd, args.runconfig)) as f:
        runconfig = json.load(f)
    print('Runconfig:')
    print(runconfig)




    # Load data
    if args.s3name is not None:
        rawPop, metadata = load_s3_data_as_df(args.s3name)
        dname = args.s3name
    else:
        if args.datapath is not None:
            rawPop, metadata = load_local_data_as_df(path.join(cwd, args.datapath))
            dname = args.datapath.split('/')[-1]


    print(f'Loaded data {dname}:')
    print(rawPop.info())

    # Make sure outdir exists
    if not path.isdir(args.outdir):
        mkdir(args.outdir)

    seed(SEED)

    ########################
    #### GAME INPUTS #######
    ########################
    # Pick targets
    targetIDs = choice(list(rawPop.index), size=runconfig['nTargets'], replace=False).tolist()

    # If specified: Add specific target records
    if runconfig['Targets'] is not None:
        targetIDs.extend(runconfig['Targets'])

    targets = rawPop.loc[targetIDs, :]

    # Drop targets from population
    rawPopDropTargets = rawPop.drop(targetIDs)

    # Init adversary's prior knowledge
    rawAidx = choice(list(rawPopDropTargets.index), size=runconfig['sizeRawA'], replace=False).tolist()
    rawA = rawPop.loc[rawAidx, :]

    # List of candidate generative models to evaluate
    gmList = []
    if 'generativeModels' in runconfig.keys():
        for gm, paramsList in runconfig['generativeModels'].items():
            if gm == 'IndependentHistogram':
                for params in paramsList:
                    gmList.append(IndependentHistogram(metadata, *params))
            elif gm == 'BayesianNet':
                for params in paramsList:
                    gmList.append(BayesianNet(metadata, *params))
            elif gm == 'PrivBayes':
                for params in paramsList:
                    gmList.append(PrivBayes(metadata, *params))
            else:
                raise ValueError(f'Unknown GM {gm}')



    ###################################
    #### ATTACK TRAINING #############
    ##################################
    print('\n---- Attack training ----')
    attacks = {}

    for tid in targetIDs:
        print(f'\n--- Adversary picks target {tid} ---')
        target = targets.loc[[tid]]
        attacks[tid] = {}



        for GenModel in gmList:
            LOGGER.info(f'Start: Attack training for {GenModel.__name__}...')

            attacks[tid][GenModel.__name__] = {}

            # Generate shadow model data for training attacks on this target
            synA, labelsSA = generate_mia_shadow_data(GenModel, target, rawA, runconfig['sizeRawT'], runconfig['sizeSynT'], runconfig['nShadows'], runconfig['nSynA'])

            # Train attack on shadow data
            for Feature in [NaiveFeatureSet(GenModel.datatype), HistogramFeatureSet(GenModel.datatype, metadata), CorrelationsFeatureSet(GenModel.datatype, metadata)]:
                Attack  = MIAttackClassifierRandomForest(metadata, Feature)
                Attack.train(synA, labelsSA)
                attacks[tid][GenModel.__name__][f'{Feature.__name__}'] = Attack

            # Clean up
            del synA, labelsSA

            LOGGER.info(f'Finished: Attack training.')

    ##################################
    ######### EVALUATION #############
    ##################################
    resultsTargetPrivacy = {tid: {gm.__name__: {} for gm in gmList } for tid in targetIDs}

    print('\n---- Start the game ----')
    for nr in range(runconfig['nIter']):
        print(f'\n--- Game iteration {nr + 1} ---')
        # Draw a raw dataset
        rIdx = choice(list(rawPopDropTargets.index), size=runconfig['sizeRawT'], replace=False).tolist()
        rawTout = rawPopDropTargets.loc[rIdx]

        for GenModel in gmList:
            LOGGER.info(f'Start: Evaluation for model {GenModel.__name__}...')
            # Train a generative model
            GenModel.fit(rawTout)
            synTwithoutTarget = [GenModel.generate_samples(runconfig['sizeSynT']) for _ in range(runconfig['nSynT'])]
            synLabelsOut = [LABEL_OUT for _ in range(runconfig['nSynT'])]

            for tid in targetIDs:
                LOGGER.info(f'Target: {tid}')
                target = targets.loc[[tid]]
                resultsTargetPrivacy[tid][f'{GenModel.__name__}'][nr] = {}

                rawTin = rawTout.append(target)
                GenModel.fit(rawTin)
                synTwithTarget = [GenModel.generate_samples(runconfig['sizeSynT']) for _ in range(runconfig['nSynT'])]
                synLabelsIn = [LABEL_IN for _ in range(runconfig['nSynT'])]

                synT = synTwithoutTarget + synTwithTarget
                synTlabels = synLabelsOut + synLabelsIn

                # Run attacks
                for feature, Attack in attacks[tid][f'{GenModel.__name__}'].items():
                    # Produce a guess for each synthetic dataset
                    attackerGuesses = Attack.attack(synT)

                    resDict = {
                        'Secret': synTlabels,
                        'AttackerGuess': attackerGuesses
                    }
                    resultsTargetPrivacy[tid][f'{GenModel.__name__}'][nr][feature] = resDict

            del synT, synTwithoutTarget, synTwithTarget

            LOGGER.info(f'Finished: Evaluation for model {GenModel.__name__}.')



    outfile = f"ResultsMIA_{dname}"
    LOGGER.info(f"Write results to {path.join(f'{args.outdir}', f'VAR_result_try3_new')}")

    with open(path.join(f'{args.outdir}', f'VAR_result_try3_new.json'), 'w') as f:
        json.dump(resultsTargetPrivacy, f, indent=2, default=json_numpy_serialzer)


if __name__ == "__main__":
    main()