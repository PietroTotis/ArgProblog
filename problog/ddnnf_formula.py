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
from problog.ground_gringo import check_evidence

import tempfile
import os
import subprocess
import time
from collections import defaultdict, Counter

from . import system_info
from .evaluator import Evaluator, EvaluatableDSP, SemiringProbability
from .errors import InconsistentEvidenceError
from .formula import LogicDAG
from .cnf_formula import CNF, CNF_ASP
from .core import transform
from .errors import CompilationError
from .util import Timer, subprocess_check_output, subprocess_check_call
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
    def __init__(self, neg_cycles=None, **kwdargs):
        LogicDAG.__init__(self, auto_compact=False, **kwdargs)
        # self.n_models = n_models
        self.neg_cycles = neg_cycles

    def _create_evaluator(self, semiring, weights, **kwargs):
        return SimpleDDNNFEvaluator(self, semiring, weights, self.neg_cycles, **kwargs)


class SimpleDDNNFEvaluator(Evaluator):
    """Evaluator for d-DNNFs."""

    def __init__(self, formula, semiring, weights=None, neg_cycles=None, **kwargs):
        Evaluator.__init__(self, formula, semiring, weights, **kwargs)
        self.cache_intermediate = {}  # weights of intermediate nodes
        self.cache_models = {} # weights of models
        self.neg_cycles = neg_cycles
        self.keytotal = {}
        self.keyworlds = {}
        self.models = []
        self.multi_sm = {}
        self.valid_choices = set()
        self.pasp = kwargs["pasp"]
        print(formula.to_dot())

        if not self.pasp:
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
        result = self.correct_weight(result)
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

    # Basic
    # def evaluate(self, node):
    #     if node == 0:
    #         if not self.semiring.is_nsp():
    #             result = self.semiring.one()
    #         else:
    #             result = self.get_root_weight()
    #             result = self.semiring.normalize(result, self._get_z())
    #     elif node is None:
    #         result = self.semiring.zero()
    #     else:
    #         ps = self._get_weight(abs(node))
    #         p = self._aggregate_weights(ps)
    #         ns = self._get_weight(-abs(node))
    #         n = self._aggregate_weights(ns)
    #         self._set_value(abs(node), (node > 0))
    #         result = self.get_root_weight()
    #         self._reset_value(abs(node), p, n)
    #         if self.has_evidence() or self.semiring.is_nsp():
    #             # print(result, self._get_z())
    #             result = self.semiring.normalize(result, self._get_z())
    #     return self.semiring.result(result, self.formula)

    # Aggregate and correct later
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
            p = self._get_weight(abs(node))
            n = self._get_weight(-abs(node))
            self._set_value(abs(node), (node > 0))
            result = self.get_root_weight()
            self._reset_value(abs(node), p, n)
            # if not abs(node) in self.evidence():
            # if not self.has_evidence():
            result = self.correct_weight(result, node)
            if self.has_evidence() or self.semiring.is_nsp() or self.pasp:
                result = self.semiring.normalize(result, self._get_z())
            result = self.semiring.result(result, self.formula)
        return result
        
    def check_model_evidence(self, model):
        # we overcount only the models that are compatible with evidence
        ok_ev = True
        for e in self.evidence():
            ok_ev = ok_ev and e in model
        return ok_ev
        
    def correct_weight(self, w, node=None):
        """
        compute the unnormalized weight first, then for each 1:many model to which the node belongs
        remove the weight of the other models that the unnormalized weight includes
        """
        for pw in self.multi_sm:
            if pw in self.cache_models:
                w_pw = self.cache_models[pw]
            else:
                w_pw = self.semiring.one()
                for atom in pw:
                    w_at = self._get_weight(atom)
                    w_pw = self.semiring.times(w_pw, w_at)
                n = len(self.multi_sm[pw])
                # consider only models that are possible w.r.t. evidence (but n is w.r.t. all anyway) 
                models = [m for m in self.multi_sm[pw] if self.check_model_evidence(m)]
                if not self.semiring.is_zero(w_pw):
                    for model in models:
                        if node in model or node is None: 
                            # print(">", node, model)
                            extra_norm = self.semiring.value(1-1/n)
                            extra_weight = self.semiring.times(w_pw, extra_norm)
                            # w-extra = 1-(extra+(1-w))
                            a = self.semiring.negate(w)
                            b = self.semiring.plus(extra_weight,a)
                            w = self.semiring.negate(b)
        return w
    
    def query(self, index):
        if self.pasp:
            return self.evaluate(index)
        if len(list(self.evidence()))==0:
            root_weight = self._get_z()
            inconsistent_weight = self.semiring.negate(root_weight)
            true_weight = self.evaluate(index)
            false_weight = self.semiring.negate(self.semiring.plus(inconsistent_weight,true_weight))
            return (true_weight, false_weight, inconsistent_weight)
        else:
            true_weight = self.evaluate(index)
            false_weight = self.semiring.negate(true_weight)
            return (true_weight, false_weight, self.semiring.zero())

        # self._initialize()
        # weights = self.weights.copy()
        # valid_mass = self.semiring.zero()
        # choice = self.valid_choices.pop()
        # self.valid_choices.add(choice)
        # for atom in choice:
        #     weights[abs(atom)] = (self.semiring.zero(), self.semiring.zero())
        # valid_choices_weights = {}
        # for vc in self.valid_choices:
        #     w  = self.semiring.one()
        #     for atom in vc:
        #         aw = self.weights[abs(atom)][atom<0]
        #         w = self.semiring.times(w,aw)
        #     valid_choices_weights[vc] = w
        #     valid_mass = self.semiring.plus(valid_mass,w)
        #     for atom in vc:
        #         val = atom<0
        #         if val:
        #             neg = weights[abs(atom)][val]
        #             weights[abs(atom)] = (weights[abs(atom)][val], self.semiring.plus(neg, w))
        #         else:
        #             pos = weights[abs(atom)][val]
        #             weights[abs(atom)] = (self.semiring.plus(pos, w), weights[abs(atom)][val])

        # p = self.semiring.zero()
        # for vc in self.valid_choices:
        #     self.weights = weights
        #     for atom in vc:
        #         if atom>0:
        #             self.set_weight(abs(atom), self.semiring.one(), self.semiring.zero())
        #         else:
        #             self.set_weight(abs(atom), self.semiring.zero(),  self.semiring.one())
        #     e = self.evaluate(index)
        #     pvc = self.semiring.times(valid_choices_weights[vc], e)
        #     p = self.semiring.plus(p, pvc)
        # i = self.semiring.negate(valid_mass)
        # tot = self.semiring.plus(p, i)
        # n   = self.semiring.negate(tot)
        # return (p, n, i)

    # Aggregate and correct later
    def _reset_value(self, index, pos, neg):
        self.set_weight(index, pos, neg)

    # Basic
    # def get_root_weight(self):
    #     """
    #     Get the WMC of the root of this formula.
    #     :return: The WMC of the root of this formula (WMC of node len(self.formula)), multiplied with weight of True
    #     (self.weights.get(0)).
    #     """
    #     weights = self._get_weight(len(self.formula))
    #     result = self._aggregate_weights(weights)
    #     return (
    #         self.semiring.times(result, self.weights.get(0)[0])
    #         if self.weights.get(0) is not None
    #         else result
    #     )

    # Aggregate and correct
    def get_root_weight(self):
        """
        Get the WMC of the root of this formula.
        :return: The WMC of the root of this formula (WMC of node len(self.formula)), multiplied with weight of True
        (self.weights.get(0)).
        """
        result = self._get_weight(len(self.formula))
        return (
            self.semiring.times(result, self.weights.get(0)[0])
            if self.weights.get(0) is not None
            else result
        )

    # Basic
    # def _get_weight(self, index):
    #     if index == 0:
    #         return [self.semiring.one()]
    #     elif index is None:
    #         return [self.semiring.zero()]
    #     else:
    #         abs_index = abs(index)
    #         w = self.weights.get(abs_index)  # Leaf nodes
    #         if w is not None:
    #             return [w[index < 0]]
    #         w = self.cache_intermediate.get(abs_index)  # Intermediate nodes
    #         if w is None:
    #             w = self._calculate_weight(index)
    #             self.cache_intermediate[abs_index] = w
    #         return w

    # Aggregate and correct later
    def _get_weight(self, index):
        if index == 0:
            return self.semiring.one()
        elif index is None:
            return self.semiring.zero()
        else:
            abs_index = abs(index)
            w = self.weights.get(abs_index)  # Leaf nodes
            if w is not None:
                return w[index < 0]
            w = self.cache_intermediate.get(abs_index)  # Intermediate nodes
            if w is None:
                w = self._calculate_weight(index)
                self.cache_intermediate[abs_index] = w
            return w

    def set_weight(self, index, pos, neg):
        # index = index of atom in weights, so atom2var[key] = index
        self.weights[index] = (pos, neg)
        self.cache_intermediate.clear()
        self.cache_models.clear()

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

    # Aggregate and correct later
    def _set_value(self, index, value):
        """Set value for given node.

        :param index: index of node
        :param value: value
        """
        if value:
            pos = self._get_weight(index)
            self.set_weight(index, pos, self.semiring.zero())
        else:
            neg = self._get_weight(-index)
            self.set_weight(index, self.semiring.zero(), neg)

    # Basic
    # def _set_value(self, index, value):
    #     """Set value for given node.

    #     :param index: index of node
    #     :param value: value
    #     """
    #     if value:
    #         poss = self._get_weight(index)
    #         pos = self._aggregate_weights(poss)
    #         self.set_weight(index, pos, self.semiring.zero())
    #     else:
    #         negs = self._get_weight(-index)
    #         neg = self._aggregate_weights(negs)
    #         self.set_weight(index, self.semiring.zero(), neg)
    
    # # Basic
    # def _aggregate_weights(self, weights):
    #     result = self.semiring.zero()
    #     for w in weights:
    #         result = self.semiring.plus(result, w)
    #     return result

    # Basic: keep 0 worlds
    # def _calculate_weight(self, key):
    #     assert key != 0
    #     assert key is not None
    #     # assert(key > 0)

    #     node = self.formula.get_node(abs(key))
    #     ntype = type(node).__name__

    #     if ntype == "atom":
    #         return [self.semiring.one()]
    #     else:
    #         assert key > 0
    #         childprobs = [self._get_weight(c) for c in node.children]
    #         # print(key, childprobs, len(self.multi_sm))
    #         if ntype == "conj":
    #             if len(self.multi_sm) == 0: # no multiple stable models: aggregate without normalization
    #                 c = self.semiring.one()
    #                 for p in childprobs:
    #                     c = self.semiring.times(c, p[0])
    #                 return [c]
    #             else:  
    #                 w_conj = list(self.wproduct(childprobs))
    #                 n_children = len(w_conj)
    #                 if key in self.keyworlds:   # if we have to normalize something
    #                     worlds = self.keyworlds[key]
    #                     for c in range(0, n_children): # follow the list
    #                         pw = frozenset(worlds[c])
    #                         n = self.multi_sm.get(pw,1) # get normalization constant
    #                         if n!=1 and not self.semiring.is_zero(w_conj[c]):
    #                             norm = self.semiring.value(1/n)
    #                             w_conj[c] = self.semiring.times(w_conj[c],norm) # replace with normalized
    #                 return w_conj
    #         elif ntype == "disj":
    #             if len(self.multi_sm) == 0:
    #                 d = self.semiring.zero()
    #                 for p in childprobs:
    #                     d = self.semiring.plus(d, p[0])
    #                 return [d]
    #             else:
    #                 cp_disj = []
    #                 for weights in childprobs:
    #                     cp_disj += [w for w in weights]
    #                 return cp_disj
    #         else:
    #             raise TypeError("Unexpected node type: '%s'." % ntype)

    # Aggregate and correct later
    def _calculate_weight(self, key):
        assert key != 0
        assert key is not None
        # assert(key > 0)

        node = self.formula.get_node(abs(key))
        ntype = type(node).__name__

        if ntype == "atom":
            return self.semiring.one()
        else:
            assert key > 0
            childprobs = [self._get_weight(c) for c in node.children]
            # print(key, list(zip(node.children, childprobs)))
            if ntype == "conj":
                p = self.semiring.one()
                for c in childprobs:
                    p = self.semiring.times(p, c)
                return p
            elif ntype == "disj":
                p = self.semiring.zero()
                for c in childprobs:
                    p = self.semiring.plus(p, c)
                return p
            else:
                raise TypeError("Unexpected node type: '%s'." % ntype)

    # def get_worlds(self, key):
    #     if key == 0 or key is None:
    #         return [[]]

    #     node = self.formula.get_node(abs(key))
    #     ntype = type(node).__name__

    #     if ntype == 'atom':
    #         # keep track of logical and probabilistic atoms
    #         if abs(key) in self.labelled or abs(key) in self.choices:
    #             return [[key]]
    #         else: #ignore extra stuff from compiler
    #             return [[]]
    #     else:
    #         assert key > 0
    #         childworlds = [self.get_worlds(c) for c in node.children]
    #         # print("cws:", key, childworlds)
    #         if ntype == 'conj':
    #             cw_conj = list(self.product(childworlds))
    #             # print("cj:", key,  cw_conj)
    #             for i, w in enumerate(cw_conj): # if the conjunction corresponds to some pw
    #                 if self.choices.issubset(self.chosen(w)): # and we made all probabilistic choices
    #                     cw_conj[i] = [] # forget about it when handing list to the partent
    #                     pw = [id for id in w if abs(id) in self.choices]
    #                     fw = frozenset(pw)
    #                     if key in self.keyworlds:   # remember that on this node we might need some normalization
    #                         self.keyworlds[key].append(fw)
    #                     else:
    #                         self.keyworlds[key] = [fw]
    #             return cw_conj  # this contains partial worlds
    #         elif ntype == 'disj':
    #             disj = []
    #             for cws in childworlds:
    #                 disj += [w for w in cws if self.partial_choice(w)] # just flatten or 
    #             # print("dws:", disj)
    #             return disj
    #         else:
    #             raise TypeError("Unexpected node type: '%s'." % ntype)

    # Aggregate later     
    # def get_worlds(self, key):
    #     if key == 0 or key is None:
    #         return [[]]

    #     node = self.formula.get_node(abs(key))
    #     ntype = type(node).__name__

    #     if ntype == 'atom':
    #         # keep track of logical and probabilistic atoms
    #         # if abs(key) in self.labelled or abs(key) in self.choices:
    #         #     return [[key]]
    #         # else: #ignore extra stuff from compiler
    #         #     return [[]]
    #         return [[key]]
    #     else:
    #         assert key > 0
    #         childworlds = [self.get_worlds(c) for c in node.children]
    #         # print("cws:", key, childworlds)
    #         if ntype == 'conj':
    #             cw_conj = list(self.product(childworlds))
    #             # print("cj:", key,  cw_conj)
    #             return cw_conj  # this contains partial worlds
    #         elif ntype == 'disj':
    #             disj = []
    #             for cws in childworlds:
    #                 disj += [w for w in cws] # just flatten or 
    #             # print("dws:", disj)
    #             return disj
    #         else:
    #             raise TypeError("Unexpected node type: '%s'." % ntype)

    # Aggregate later with numpy
    def get_worlds(self, key):
        if key == 0 or key is None:
            return ((),)

        node = self.formula.get_node(abs(key))
        ntype = type(node).__name__

        if ntype == 'atom':
            return ((key, ), )
        else:
            assert key > 0
            childworlds = [self.get_worlds(c) for c in node.children]
            # print("cws:", key, childworlds)
            if ntype == 'conj':
                cw_conj = tuple(self.tproduct(childworlds))
                # print("cj:", key,  len(cw_conj), [len(w) for w in cw_conj])
                return cw_conj  # this contains partial worlds
            elif ntype == 'disj':
                # disj = childworlds.flatten()
                disj = sum(childworlds, ())
                # print("dws:", disj)
                return disj
            else:
                raise TypeError("Unexpected node type: '%s'." % ntype)

    def tproduct(self, ar_list):
        if not ar_list:
            yield ()
        else:
            for a in ar_list[0]:
                for prod in self.tproduct(ar_list[1:]):
                    yield a+prod

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

    def subset_diff(self,a,b):
        return a.issubset(b) and a != b

    def chosen(self, world):
        return set([abs(id) for id in world])

    def partial_choice(self, world):
        chosen_facts = self.chosen(world) & self.choices
        return self.subset_diff(chosen_facts, self.choices)

    # def pwproduct(self, ar_list):
    #     if not ar_list:
    #         yield ([], self.semiring.one())
    #     else:
    #         for w, p in ar_list[0]:
    #             for wprod, pprod in self.pwproduct(ar_list[1:]):
    #                 yield (w+wprod, self.semiring.times(p, pprod))

    # Basic
    # def multi_stable_models(self):
    #     self.labelled = [id for _, id, _ in self.formula.labeled()] # logical and probabilistic atoms
    #     weights = self.formula.get_weights()
    #     self.choices = set([key for key in weights if not isinstance(weights[key], bool)])
    #     root = len(self.formula._nodes)
    #     # print(weights)
    #     # print(self.labelled)
    #     # print(self.choices)

    #     ws = self.get_worlds(root)  
    #     n_models = len(ws)  
    #     worlds = [w for ws in self.keyworlds.values() for w in ws]
    #     print(worlds)
    #     self.multi_sm = Counter(worlds)
    #     # if the number of models is a multiple of the total number from the counter
    #     # then there must be some non-probabilistic choice in each world
    #     # then normalize each world w.r.t. that number
    #     n_pws = sum(self.multi_sm.values())
    #     n_logic_choices = n_models / n_pws

    #     self.multi_sm = {k: c*n_logic_choices for k, c in self.multi_sm.items() if c>1 or n_logic_choices>1}

    def multi_stable_models(self):
        self.labelled = [id for _, id, _ in self.formula.labeled()] # logical and probabilistic atoms
        weights = self.formula.get_weights()
        self.choices = set([key for key in weights if not isinstance(weights[key], bool)])
        print(len(self.choices),len(self.formula))
        if self.neg_cycles:
            # print("mh")
            root = len(self.formula._nodes)
            # print(weights)
            # print(self.labelled)
            # print(self.choices)

            start = time.time()
            self.models = self.get_worlds(root)

            # n_models = len(self.models)  
            # worlds = [w for ws in self.keyworlds.values() for w in ws]
            # self.multi_sm = Counter(worlds)
            for model in self.models:
                choices = frozenset([atom for atom in model if abs(atom) in self.choices])
                self.valid_choices.add(choices)
                if choices in self.multi_sm:
                    self.multi_sm[choices].append(model)
                else:
                    self.multi_sm[choices] = [model]
            # self.multi_sm = Counter([frozenset(m) for m in self.models])
            # if the number of models is a multiple of the total number from the counter
            # then there must be some non-probabilistic choice in each world
            # then normalize each world w.r.t. that number
            # n_pws = sum(self.multi_sm.values())
            # n_pws = len(self.multi_sm)
            # self.n_logic_choices = n_models / n_pws

            self.multi_sm = {k:self.multi_sm[k] for k in self.multi_sm if len(self.multi_sm[k])>1}
            # self.multi_sm = {k: c*n_logic_choices for k, c in self.multi_sm.items() if c>1 or n_logic_choices>1}
            # print(self.keyworlds)
            end = time.time()
            print(f"Enumeration: {round(end-start,3)}s")
            # print(self.multi_sm.values())

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


# if system_info.get("c2d", False):

    # noinspection PyUnusedLocal
@transform(CNF_ASP, DDNNF)
def _compile_with_c2d(cnf, nnf=None, smooth=True, **kwdargs):
    fd, cnf_file = tempfile.mkstemp(".cnf")
    os.close(fd)
    nnf_file = cnf_file + ".nnf"
    if smooth:
        smoothl = ["-smooth_all"]
    else:
        smoothl = []

    cmd = ["c2d"] + smoothl + ["-reduce", "-in", cnf_file]

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
# @transform(CNF_ASP, DDNNF)
# def _compile_with_dsharp(cnf, nnf=None, smooth=True, **kwdargs):
#     result = None
#     with Timer("DSharp compilation"):
#         fd1, cnf_file = tempfile.mkstemp(".cnf")
#         fd2, nnf_file = tempfile.mkstemp(".nnf")
#         os.close(fd1)
#         os.close(fd2)
#         if smooth:
#             smoothl = ["-smoothNNF"]
#         else:
#             smoothl = []
#         cmd = ["dsharp", "-Fnnf", nnf_file] + smoothl + ["-disableAllLits", cnf_file]  #

#         try:
#             result = _compile(cnf, cmd, cnf_file, nnf_file)
#         except subprocess.CalledProcessError:
#             raise DSharpError()

#         try:
#             os.remove(cnf_file)
#         except OSError:
#             pass
#         try:
#             os.remove(nnf_file)
#         except OSError:
#             pass

#     return result


# Compiler.add("dsharp", _compile_with_dsharp)

# noinspection PyUnusedLocal
# @transform(CNF_ASP, DDNNF)
# def _compile_with_dsharp_asp(cnf, nnf=None, smooth=True, **kwdargs):
#     result = None
#     with Timer('DSharp compilation'):
#         fd1, cnf_file = tempfile.mkstemp('.cnf')
#         fd2, nnf_file = tempfile.mkstemp('.nnf')
#         os.close(fd1)
#         os.close(fd2)
#         if smooth:
#             smoothl = '-smoothNNF'
#         else:
#             smoothl = ''
#         # cmd = ['dsharp_with_unfounded', '-noIBCP', '-evidencePropagated', '-noPP', '-Fnnf', nnf_file, smoothl, '-disableAllLits', cnf_file]
#         # cmd = ['dsharp_with_unfounded', '-noIBCP', '-noPP', '-Fnnf', nnf_file, smoothl, '-disableAllLits', cnf_file]
#         cmd = ['dsharp_with_unfounded', '-noIBCP', '-noPP', '-Fnnf', nnf_file, '-smoothNNF', '-disableAllLits', cnf_file]

#         try:
#             result = _compile(cnf, cmd, cnf_file, nnf_file)
#         except subprocess.CalledProcessError:
#             raise DSharpError()

#         try:
#             os.remove(cnf_file)
#         except OSError:
#             pass
#         try:
#             os.remove(nnf_file)
#         except OSError:
#             pass

#     return result

# Compiler.add('dsharp_asp', _compile_with_dsharp_asp)


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
                start = time.time()
                # out = subprocess_check_output(cmd)
                # print(out)
                with open(os.devnull, "w") as OUT_NULL:
                    subprocess_check_call(cmd, stdout=OUT_NULL)
                end = time.time()
                print(f"Compilation: {round(end-start,3)}s")
                # i = out.find("# of solutions:")
                # j = out.find("#SAT")
                # n_models = float(out[i+17:j])
                success = True
            except subprocess.CalledProcessError as err:
                attempts_left -= 1
                if attempts_left == 0:
                    raise err
        return _load_nnf(nnf_file, cnf)


def _load_nnf(filename, cnf):
    nnf = DDNNF(cnf.neg_cycles, keep_all=True)

    weights = cnf.get_weights()

    names_inv = defaultdict(list)
    for name, node, label in cnf.get_names_with_label():
        names_inv[node].append((name, label))

    with open(filename) as f:
        line2node = {}
        rename = {}
        lnum = 0
        for line in f:
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
                    nnf.add_name(actual_name, len(nnf), label)
                else:
                    nnf.add_name(actual_name, None, label)

    for c in cnf.constraints():
        nnf.add_constraint(c.copy(rename))

    return nnf
