"""
--------------------------------------------------------------------------------
         Module that parses global parameters from a configuration file
--------------------------------------------------------------------------------

Author: Diogo L.O.C. (locdiogo@gmail.com)


Last Date: 08/2025

Description:
Module that parses global parameters from a configuration file at first import,
to make them available to the other parts of the program.

More information in:
https://wiki.python.org/moin/ConfigParserExamples

Input:
Configuration file, wherein global paths and parameters are defined.

Outputs:
The module provides a parser for simple configuration files consisting of groups
of named values.

"""

import configparser
import os
import glob
import ast

def select_and_parse_config_file(basedir='.', ext='cnf', verbose=True):
    """
    Reads a configuration file and returns an instance of ConfigParser:
    First, looks for files in *basedir* with extension *ext*.
    Asks user to select a file if several files are found,
    and parses it using ConfigParser module.
    @rtype: L{ConfigParser.ConfigParser}
    """
    config_files = glob.glob(os.path.join(basedir, u'*.{}'.format(ext)))

    if not config_files:
        raise Exception("No configuration file found!")

    if len(config_files) == 1:
        # only one configuration file
        config_file = config_files[0]
    else:
        print("Select a configuration file:")
        for i, f in enumerate(config_files, start=1):
            print("{} - {}".format(i, f))
        res = int(input(''))
        config_file = config_files[res - 1]

    if verbose:
        print("Reading configuration file: {}".format(config_file))

    conf = configparser.ConfigParser(allow_no_value=True)
    conf.read(config_file)

    return conf

# ==========================
# parsing configuration file
# ==========================

config = select_and_parse_config_file(basedir='.', ext='cnf', verbose=True)

# -----
# paths
# -----

## ------------------------
## Name of the model

MODEL_NAME = config.get('paths', 'MODEL_NAME')

## -----------------------
## Directory of the output (Figures and Files)

FOLDER_OUTPUT = config.get('paths', 'FOLDER_OUTPUT')

# -----
# model
# -----

## ----------------
## Number of layers:
NUMBER_LAYERS = config.getint('model', 'NUMBER_LAYERS')

## ------------
## Depth ranges (in meters)
DEPTH_RANGES = ast.literal_eval(config.get('model', 'DEPTH_RANGES'))

## --------------
## Density ranges (g/cm³)
DENSITY_RANGES = ast.literal_eval(config.get('model', 'DENSITY_RANGES'))

# ----
# gene
# ----

## ----------------------------------------
## Minimum thickness of an individual layer (m).
MIN_THICK_LAYER = config.getfloat('gene', 'MIN_THICK_LAYER')

## ----------------------------------------
## Maximum thickness of an individual layer  (m)
MAX_THICK_LAYER = config.getfloat('gene', 'MAX_THICK_LAYER')

## -------------------------------------
## Maximum total thickness of all layers (m)
MAX_TOTAL = config.getfloat('gene', 'MAX_TOTAL')

## -----------------------------
## Number of layers of the model (int)
MAX_LAYERS = config.getint('gene', 'MAX_LAYERS')

## ---------------
## Lower bound for Vs values (m/s)
LOW_VELS = config.getfloat('gene', 'LOW_VELS')

## ---------------
## Upper bound for Vs values (m/s).
UP_VELS = config.getfloat('gene', 'UP_VELS')

## ----------------------------------
## Probability of mutating each value (default=0.02)
MUTPB = config.getfloat('gene', 'MUTPB')

## ---------------------------------------
## The probability of performing crossover (default=0.7)
CXBB = config.getfloat('gene', 'CXBB')     

## ---------------------------------------
## Starting best solution estimation:
HOF = config.getint('gene', 'HOF')    

## ---------------------
## Number of generations:
NGEN = config.getint('gene', 'NGEN')

## ---------------
## Number of individuals in population:
POPULATION = config.getint('gene', 'POPULATION')

## --------------------
## Number of inversions:
N_INV = config.getint('gene', 'N_INV')

## ---------------
## MULTIPROCESSING
NUM_PROCESS = config.getint('gene', 'NUM_PROCESS')