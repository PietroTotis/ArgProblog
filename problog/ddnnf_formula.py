"""
problog.nnf_formula - d-DNNF
----------------------------

Provides access to d-DNNF formulae.

..
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

import tempfile
import os
import subprocess
from collections import defaultdict, Counter

from . import system_info
from .evaluator import Evaluator, EvaluatableDSP
from .errors import InconsistentEvidenceError
from .formula import LogicDAG
from .cnf_formula import CNF, CNF_ASP
from .core import transform
from .errors import CompilationError
from .util import Timer, subprocess_check_call
from .logic import Constant


class DSharpError(CompilationError):
    """DSharp has crashed."""

    def __init__(self):
        msg = "DSharp has encountered an error"
        if system_info["os"] == "darwin":
            msg += ". This is a known issue. See KNOWN_ISSUES for details on how to resolve this problem"
        CompilationError.__init__(self, msg)


class DDNNF(LogicDAG, EvaluatableDSP):
    """A d-DNNF formula."""

    transform_preference = 20

    # noinspection PyUnusedLocal,PyUnusedLocal,PyUnusedLocal
    def __init__(self, **kwdargs):
        LogicDAG.__init__(self, auto_compact=False)

    def _create_evaluator(self, semiring, weights, **kwargs):
        return SimpleDDNNFEvaluator(self, semiring, weights)


class SimpleDDNNFEvaluator(Evaluator):
    """Evaluator for d-DNNFs."""

    def __init__(self, formula, semiring, weights=None, **kwargs):
        Evaluator.__init__(self, formula, semiring, weights, **kwargs)
        self.cache_intermediate = {}  # weights of intermediate nodes
        self.keytotal = {}
        self.keyworlds = {}
        self.multi_sm = Counter()
        print(formula.to_dot())
        # print(formula)
        self.multi_stable_models()

    def _initialize(self, with_evidence=True):
        self.weights.clear()

        model_weights = self.formula.extract_weights(self.semiring, self.given_weights)
        self.weights = model_weights.copy()

        if with_evidence:
            for ev in self.evidence():
                self.set_evidence(abs(ev), ev > 0)

        if self.semiring.is_zero(self._get_z()):
            raise InconsistentEvidenceError(context=" during evidence evaluation")

    def propagate(self):
        self._initialize()

    def _get_z(self):
        result = self.get_root_weight()
        return result

    def evaluate_evidence(self, recompute=False):
        return self.semiring.result(
            self._evaluate_evidence(recompute=recompute), self.formula
        )

    # noinspection PyUnusedLocal
    def _evaluate_evidence(self, recompute=False):
        self._initialize(False)
        for ev in self.evidence():
            self._set_value(abs(ev), ev > 0)

        result = self.get_root_weight()
        return result

    def evaluate_fact(self, node):
        return self.evaluate(node)

    def evaluate(self, node):
        if node == 0:
            if not self.semiring.is_nsp():
                result = self.semiring.one()
            else:
                result = self.get_root_weight()
                result = self.semiring.normalize(result, self._get_z())
        elif node is None:
            result = self.semiring.zero()
        else:
            ps = self._get_weight(abs(node))
            p = self._aggregate_weights(ps)
            ns = self._get_weight(-abs(node))
            n = self._aggregate_weights(ns)
            self._set_value(abs(node), (node > 0))
            result = self.get_root_weight()
            self._reset_value(abs(node), p, n)
            if self.has_evidence() or self.semiring.is_nsp():
                # print(result, self._get_z())
                result = self.semiring.normalize(result, self._get_z())
        return self.semiring.result(result, self.formula)

    def _reset_value(self, index, pos, neg):
        self.set_weight(index, pos, neg)

    def get_root_weight(self):
        """
        Get the WMC of the root of this formula.
        :return: The WMC of the root of this formula (WMC of node len(self.formula)), multiplied with weight of True
        (self.weights.get(0)).
        """
        weights = self._get_weight(len(self.formula))
        result = self._aggregate_weights(weights)
        return (
            self.semiring.times(result, self.weights.get(0)[0])
            if self.weights.get(0) is not None
            else result
        )

    # Basic
    def _get_weight(self, index):
        if index == 0:
            return [self.semiring.one()]
        elif index is None:
            return [self.semiring.zero()]
        else:
            abs_index = abs(index)
            w = self.weights.get(abs_index)  # Leaf nodes
            if w is not None:
                return [w[index < 0]]
            w = self.cache_intermediate.get(abs_index)  # Intermediate nodes
            if w is None:
                w = self._calculate_weight(index)
                self.cache_intermediate[abs_index] = w
            return w

    # Bug
    # def _get_weight(self, index):
    #     if index == 0:
    #         return [([], self.semiring.one())]
    #     elif index is None:
    #         return [([], self.semiring.zero())]
    #     else:
    #         abs_index = abs(index)
    #         w = self.weights.get(abs_index)  # Leaf nodes
    #         if w is not None:
    #             if  w[index < 0] != self.semiring.one():
    #                 return [([index], w[index < 0])]
    #             else:
    #                 return [([], w[index < 0])]
    #         w = self.cache_intermediate.get(abs_index)  # Intermediate nodes
    #         if w is None:
    #             w = self._calculate_weight(index)
    #             self.cache_intermediate[abs_index] = w
    #         return w

    def set_weight(self, index, pos, neg):
        # index = index of atom in weights, so atom2var[key] = index
        self.weights[index] = (pos, neg)
        self.cache_intermediate.clear()

    def set_evidence(self, index, value):
        curr_pos_weight, curr_neg_weight = self.weights.get(index)
        pos, neg = self.semiring.to_evidence(
            curr_pos_weight, curr_neg_weight, sign=value
        )

        if (value and self.semiring.is_zero(curr_pos_weight)) or (
            not value and self.semiring.is_zero(curr_neg_weight)
        ):
            raise InconsistentEvidenceError(self._deref_node(index))

        self.set_weight(index, pos, neg)

    def _deref_node(self, index):
        return self.formula.get_node(index).name

    def _set_value(self, index, value):
        """Set value for given node.

        :param index: index of node
        :param value: value
        """
        if value:
            poss = self._get_weight(index)
            pos = self._aggregate_weights(poss)
            self.set_weight(index, pos, self.semiring.zero())
        else:
            negs = self._get_weight(-index)
            neg = self._aggregate_weights(negs)
            self.set_weight(index, self.semiring.zero(), neg)

    # Bug
    # def _aggregate_weights(self, pws):
    #     result = self.semiring.zero()
    #     for w, p in pws:
    #         result = self.semiring.plus(result, p)
    #     return result
    
    # Basic
    def _aggregate_weights(self, weights):
        result = self.semiring.zero()
        for w in weights:
            result = self.semiring.plus(result, w)
        return result

    # list of (world, probability)
    # simplify 0 worlds: bad idea: from 0 worlds can come non-zero pws due to neg cycles
    # def  _calculate_weight(self, key):

    #     node = self.formula.get_node(abs(key))
    #     ntype = type(node).__name__

    #     if ntype == 'atom':
    #         return [([], self.semiring.one())]
    #     else:
    #         assert key > 0
    #         childworlds = [self._get_weight(c) for c in node.children]
    #         # print(key, childworlds)
    #         if ntype == 'conj':  
    #             pw_conj = list(self.pwproduct(childworlds))
    #             print("cws:", pw_conj)
    #             conj = [] 
    #             for pw in pw_conj:
    #                 w, p = pw
    #                 freeze_pw = frozenset(w)
    #                 n = self.multi_sm.get(freeze_pw,1)
    #                 if not self.semiring.is_zero(p):
    #                     p_norm = p
    #                     if n!=1 and key in self.keyworlds:
    #                         norm = self.semiring.value(1/n)
    #                         p_norm = self.semiring.times(p,norm)
    #                     conj.append((w, p_norm))
    #             # print("cws", conj )
    #             return conj
    #         elif ntype == 'disj':
    #             disj = []
    #             for pws in childworlds:
    #                 disj += [(w,p) for w, p in pws if not self.semiring.is_zero(p)]
    #             # print("dws:", disj)
    #             return disj
    #         else:
    #             raise TypeError("Unexpected node type: '%s'." % ntype)

    # Basic: keep 0 worlds
    def _calculate_weight(self, key):
        assert key != 0
        assert key is not None
        # assert(key > 0)

        node = self.formula.get_node(abs(key))
        ntype = type(node).__name__

        if ntype == "atom":
            return [self.semiring.one()]
        else:
            assert key > 0
            childprobs = [self._get_weight(c) for c in node.children]
            # print(key, childprobs, len(self.multi_sm))
            if ntype == "conj":
                if len(self.multi_sm) == 0:
                    c = self.semiring.one()
                    for p in childprobs:
                        c = self.semiring.times(c, p[0])
                    return [c]
                else:  
                    w_conj = list(self.wproduct(childprobs))
                    n_children = len(w_conj)
                    if key in self.keyworlds:
                        worlds = self.keyworlds[key]
                        for c in range(0, n_children):
                            pw = frozenset(worlds[c])
                            n = self.multi_sm.get(pw,1)
                            if n!=1 and not self.semiring.is_zero(w_conj[c]):
                                norm = self.semiring.value(1/n)
                                w_conj[c] = self.semiring.times(w_conj[c],norm)
                    return w_conj
            elif ntype == "disj":
                if len(self.multi_sm) == 0:
                    d = self.semiring.zero()
                    for p in childprobs:
                        d = self.semiring.plus(d, p[0])
                    return [d]
                else:
                    cp_disj = []
                    for weights in childprobs:
                        cp_disj += [w for w in weights]
                    return cp_disj
            else:
                raise TypeError("Unexpected node type: '%s'." % ntype)


    # def get_paths(self, key, atom):
    #     node = self.formula.get_node(abs(key))
    #     ntype = type(node).__name__
    #     if key == atom:
    #         return 1
    #     elif ntype == 'atom':
    #         return 0
    #     elif ntype == 'conj':
    #         p_children = [self.get_paths(c, atom) for c in node.children]
    #         prod = 1
    #         for n in p_children:
    #             if n>0:
    #                 prod = prod*n 
    #         return prod
    #     else:
    #         p_children = [self.get_paths(c, atom) for c in node.children]
    #         return sum(p_children)

    # visits
    # def get_worlds(self, key, pws, n_choices):
    #     if key == 0 or key is None:
    #         return pws
        
    #     node = self.formula.get_node(abs(key))
    #     ntype = type(node).__name__

    #     if ntype == 'atom':
    #         if node.probability != True: #?
    #             return [[key] + pw for pw in pws]
    #         else:
    #             return pws
    #     else:
    #         assert key > 0
                
    #         if ntype == 'conj':
    #             new_pws = pws
    #             for c in node.children:
    #                 new_pws = self.get_worlds(c, new_pws, n_choices)
    #             partial = []
    #             for pw in new_pws:
    #                 if len(pw) == n_choices:
    #                     fpw = frozenset(pw)
    #                     if key in self.keyworlds:
    #                         self.keyworlds[key].append(fpw)
    #                     else:
    #                         self.keyworlds[key] = [fpw]
    #                     self.multi_sm[fpw] += 1
    #                 else:
    #                     partial.append(pw)
    #             return partial
    #         elif ntype == 'disj':
    #             disj = []
    #             for c in node.children:
    #                 disj += self.get_worlds(c, pws, n_choices)
    #             return disj
    #         else:
    #             raise TypeError("Unexpected node type: '%s'." % ntype)

    def get_worlds(self, key, n_choices):
        if key == 0 or key is None:
            return [[]]

        node = self.formula.get_node(abs(key))
        ntype = type(node).__name__

        if ntype == 'atom':
            if node.probability != True and node.probability is not None: #?
                return [[key]]
            else:
                return [[]]
        else:
            assert key > 0
            childworlds = [self.get_worlds(c, n_choices) for c in node.children]
            # print("cws:", key, childworlds)
            if ntype == 'conj':
                cw_conj = list(self.product(childworlds))
                # print("cj:", key,  cw_conj)
                for i, w in enumerate(cw_conj):
                    if len(w) == n_choices:
                        cw_conj[i] = []
                        fw = frozenset(w)
                        if key in self.keyworlds:
                            self.keyworlds[key].append(fw)
                        else:
                            self.keyworlds[key] = [fw]
                # if len(cw_conj) > 0 and len(cw_conj[0]) == n_choices:
                #     self.keyworlds[key] = [frozenset(w) for w in cw_conj]
                return cw_conj
            elif ntype == 'disj':
                disj = []
                for cws in childworlds:
                    disj += [w for w in cws if len(w) < n_choices or n_choices==1]
                # print("dws:", disj)
                return disj
            else:
                raise TypeError("Unexpected node type: '%s'." % ntype)

    def product(self, ar_list):
        if not ar_list:
            yield []
        else:
            for a in ar_list[0]:
                for prod in self.product(ar_list[1:]):
                    yield a+prod
    
    def wproduct(self, ar_list):
        if not ar_list:
            yield self.semiring.one()
        else:
            for w in ar_list[0]:
                for prod in self.wproduct(ar_list[1:]):
                    yield self.semiring.times(w, prod)

    # def pwproduct(self, ar_list):
    #     if not ar_list:
    #         yield ([], self.semiring.one())
    #     else:
    #         for w, p in ar_list[0]:
    #             for wprod, pprod in self.pwproduct(ar_list[1:]):
    #                 yield (w+wprod, self.semiring.times(p, pprod))

    def multi_stable_models(self):
        weights = self.formula.get_weights()
        choices = [key for key in weights if isinstance(weights[key],Constant)]
        n_choices = len(choices)
        # print(choices, n_choices)
        root = len(self.formula._nodes)
        # n_choices = len([w for w in self.formula.get_weights().values() if isinstance(w,Constant)])
        # print(choices)
        # root = len(self.formula._nodes)
        # paths = {key: self.get_paths(root, key) for key in choices}
        # print(paths)

        # self.multi_sm = Counter()
        # pws = [[]]
        # self.get_worlds(root, pws, n_choices)

        self.get_worlds(root, n_choices)
        worlds = [w for ws in self.keyworlds.values() for w in ws]
        self.multi_sm = Counter(worlds)
        # print(self.multi_sm)
        # if len(self.multi_sm) > 0:
        #     _, min_count = self.multi_sm.most_common()[-1]
        self.multi_sm = {k: c for k, c in self.multi_sm.items() if c>1}
            # self.multi_sm = {k: c/min_count for k, c in self.multi_sm.items() if c/min_count>1}
        # print(self.multi_sm, len(self.multi_sm))
        # print("")

class Compiler(object):
    """Interface to CNF to d-DNNF compiler tool."""

    __compilers = {}

    @classmethod
    def get_default(cls):
        """Get default compiler for this system."""
        if system_info.get("c2d", False):
            return _compile_with_c2d
        else:
            return _compile_with_dsharp

    @classmethod
    def get(cls, name):
        """Get compiler by name (or default if name not found).

        :param name: name of the compiler
        :returns: function used to call compiler
        """
        result = cls.__compilers.get(name)
        if result is None:
            result = cls.get_default()
        return result

    @classmethod
    def add(cls, name, func):
        """Add a compiler.

        :param name: name of the compiler
        :param func: function used to call the compiler
        """
        cls.__compilers[name] = func


if system_info.get("c2d", False):
    # noinspection PyUnusedLocal
    @transform(CNF, DDNNF)
    def _compile_with_c2d(cnf, nnf=None, smooth=True, **kwdargs):
        fd, cnf_file = tempfile.mkstemp(".cnf")
        os.close(fd)
        nnf_file = cnf_file + ".nnf"
        if smooth:
            smoothl = ["-smooth_all"]
        else:
            smoothl = []

        cmd = ["cnf2dDNNF", "-dt_method", "0"] + smoothl + ["-reduce", "-in", cnf_file]

        try:
            os.remove(cnf_file)
        except OSError:
            pass
        try:
            os.remove(nnf_file)
        except OSError:
            pass

        return _compile(cnf, cmd, cnf_file, nnf_file)

    Compiler.add("c2d", _compile_with_c2d)


# noinspection PyUnusedLocal
# @transform(CNF, DDNNF)
def _compile_with_dsharp(cnf, nnf=None, smooth=True, **kwdargs):
    result = None
    with Timer("DSharp compilation"):
        fd1, cnf_file = tempfile.mkstemp(".cnf")
        fd2, nnf_file = tempfile.mkstemp(".nnf")
        os.close(fd1)
        os.close(fd2)
        if smooth:
            smoothl = ["-smoothNNF"]
        else:
            smoothl = []
        cmd = ["dsharp", "-Fnnf", nnf_file] + smoothl + ["-disableAllLits", cnf_file]  #

        try:
            result = _compile(cnf, cmd, cnf_file, nnf_file)
        except subprocess.CalledProcessError:
            raise DSharpError()

        try:
            os.remove(cnf_file)
        except OSError:
            pass
        try:
            os.remove(nnf_file)
        except OSError:
            pass

    return result


Compiler.add("dsharp", _compile_with_dsharp)

# noinspection PyUnusedLocal
@transform(CNF_ASP, DDNNF)
def _compile_with_dsharp_asp(cnf, nnf=None, smooth=True, **kwdargs):
    result = None
    with Timer('DSharp compilation'):
        fd1, cnf_file = tempfile.mkstemp('.cnf')
        fd2, nnf_file = tempfile.mkstemp('.nnf')
        os.close(fd1)
        os.close(fd2)
        if smooth:
            smoothl = '-smoothNNF'
        else:
            smoothl = ''
        # cmd = ['dsharp_with_unfounded', '-noIBCP', '-evidencePropagated', '-noPP', '-Fnnf', nnf_file, smoothl, '-disableAllLits', cnf_file]
        # cmd = ['dsharp_with_unfounded', '-noIBCP', '-noPP', '-Fnnf', nnf_file, smoothl, '-disableAllLits', cnf_file]
        cmd = ['dsharp_with_unfounded', '-noIBCP', '-noPP', '-Fnnf', nnf_file, '-smoothNNF', '-disableAllLits', cnf_file]

        try:
            result = _compile(cnf, cmd, cnf_file, nnf_file)
        except subprocess.CalledProcessError:
            raise DSharpError()

        try:
            os.remove(cnf_file)
        except OSError:
            pass
        try:
            os.remove(nnf_file)
        except OSError:
            pass

    return result

Compiler.add('dsharp_asp', _compile_with_dsharp_asp)


def _compile(cnf, cmd, cnf_file, nnf_file):
    names = cnf.get_names_with_label()

    if cnf.is_trivial():
        nnf = DDNNF()
        weights = cnf.get_weights()
        for i in range(1, cnf.atomcount + 1):
            nnf.add_atom(i, weights.get(i))
        or_nodes = []
        for i in range(1, cnf.atomcount + 1):
            or_nodes.append(nnf.add_or((i, -i)))
        if or_nodes:
            nnf.add_and(or_nodes)

        for name, node, label in names:
            nnf.add_name(name, node, label)
        for c in cnf.constraints():
            nnf.add_constraint(c.copy())

        return nnf
    else:
        with open(cnf_file, "w") as f:
            f.write(cnf.to_dimacs())

        attempts_left = 1
        success = False
        while attempts_left and not success:
            try:
                # subprocess_check_call(cmd)
                with open(os.devnull, "w") as OUT_NULL:
                    subprocess_check_call(cmd, stdout=OUT_NULL)
                success = True
            except subprocess.CalledProcessError as err:
                attempts_left -= 1
                if attempts_left == 0:
                    raise err
        return _load_nnf(nnf_file, cnf)


def _load_nnf(filename, cnf):
    nnf = DDNNF()

    weights = cnf.get_weights()

    names_inv = defaultdict(list)
    for name, node, label in cnf.get_names_with_label():
        names_inv[node].append((name, label))

    with open(filename) as f:
        line2node = {}
        rename = {}
        lnum = 0
        for line in f:
            # print(line)
            line = line.strip().split()
            if line[0] == "nnf":
                pass
            elif line[0] == "L":
                name = int(line[1])
                prob = weights.get(abs(name), True)
                node = nnf.add_atom(abs(name), prob)
                rename[abs(name)] = node
                if name < 0:
                    node = -node
                line2node[lnum] = node
                if name in names_inv:
                    for actual_name, label in names_inv[name]:
                        nnf.add_name(actual_name, node, label)
                    del names_inv[name]
                lnum += 1
            elif line[0] == "A":
                children = map(lambda x: line2node[int(x)], line[2:])
                line2node[lnum] = nnf.add_and(children)
                lnum += 1
            elif line[0] == "O":
                children = map(lambda x: line2node[int(x)], line[3:])
                line2node[lnum] = nnf.add_or(children)
                lnum += 1
            else:
                print("Unknown line type")
        for name in names_inv:
            for actual_name, label in names_inv[name]:
                if name == 0:
                    nnf.add_name(actual_name, 0, label)
                else:
                    nnf.add_name(actual_name, None, label)
    for c in cnf.constraints():
        nnf.add_constraint(c.copy(rename))

    return nnf
