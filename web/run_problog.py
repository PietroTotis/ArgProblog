#! /usr/bin/env python
"""
Part of the ProbLog distribution.

Copyright 2015 KU Leuven, DTAI Research Group

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import print_function

import os, sys, subprocess, traceback, json

sys.setrecursionlimit(10000)

sys.path.insert(0, os.path.abspath( os.path.join(os.path.dirname(__file__), '../')))

from problog.parser import DefaultPrologParser
from problog.program import PrologFile, ExtendedPrologFactory
from problog.evaluator import SemiringLogProbability, Evaluator
# from problog.nnf_formula import NNF
from problog.sdd_formula import SDD
from problog.core import process_error, GroundingError, ParseError

def print_result( d, output, precision=8 ) :
    success, d = d
    result = {}
    if success :
        result['SUCCESS'] = True
        result['probs'] = [[str(n),round(p,12),l,c] for n,p,l,c in d]
        print (200, 'application/json', json.dumps(result), file=output)
    else :
        #print (400, 'application/json', json.dumps(d), file=output)
        result['SUCCESS'] = False
        result = d
        print (200, 'application/json', json.dumps(result), file=output)
    return 0

def process_error( err ) :
    """Take the given error raise by ProbLog and produce a meaningful error message."""
    err_type = type(err).__name__
    if err_type == 'MemoryError':
        return { 'message': 'ProbLog exceeded the available memory limit.' }
    elif err_type == 'ParseException' :
        return { 'message': 'Parsing error on %s:%s: %s.\n%s' % (err.lineno, err.col, err.msg, err.line ), 'lineno' : err.lineno, 'col': err.col }
    elif isinstance(err, ParseError) :
        return { 'message': 'Parsing error on %s:%s: %s.' % (err.lineno, err.col, err.msg ), 'lineno' : err.lineno, 'col' : err.col }
    elif isinstance(err, GroundingError) :
        try :
            location = err.location
            if location :
                return { 'message': 'Error during grounding: %s' % err, 'lineno' : location[0], 'col' : location[1] }
            else :
                return { 'message': 'Error during grounding: %s' % err }
        except AttributeError :
            return { 'message': 'Error during grounding: %s' % err }
    else :
        traceback.print_exc()
        return { 'message' : 'Unknown error: %s' % (err_type) }


def main( filename) :

    try :
        model = PrologFile(filename, parser=DefaultPrologParser(ExtendedPrologFactory()))
        formula = SDD.createFrom( model )
        result = formula.evaluate(semiring=SemiringLogProbability())

        new_result= []
        for n,p in result.items():
            new_result.append((str(n.withProbability()),p,) + model.lineno(n.location))

        return True, new_result
    except Exception as err :
        return False, {'err':process_error(err)}

if __name__ == '__main__' :
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', metavar='MODEL')
    parser.add_argument('output', metavar='OUTPUT')

    args = parser.parse_args()

    result = main(args.filename)
    with open(args.output, 'w') as output :
        retcode = print_result( result , output )

    # Always exit with code 0
