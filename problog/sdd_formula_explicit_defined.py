"""
problog.sdd_formula_explicit - Sentential Decision Diagrams
--------------------------------------------------

Interface to Sentential Decision Diagrams (SDD) using the explicit encoding representing all models
(similar to d-DNNF encoding except that it is not converted into cnf first).

..
    Part of the ProbLog distribution.

    Copyright 2018 KU Leuven, DTAI Research Group

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
import sys
import os
from collections import namedtuple
sys.path.append(os.path.join(os.path.dirname(__file__), 'aspmc'))
from .aspmc.aspmc.compile.vtree import Vtree as aspmc_Vtree
from .aspmc.aspmc.compile.cnf import CNF as aspmc_CNF
from .aspmc.aspmc.compile.constrained_compile import tree_from_cnf 

from .formula import LogicDAG, LogicFormula
from .core import transform
from .errors import InconsistentEvidenceError
from .util import Timer
from .evaluator import SemiringProbability, SemiringLogProbability
from .sdd_formula import SDDEvaluator, SDD, x_constrained
from .forward import _ForwardSDD, ForwardInference
from .sdd_formula import SDDManager

from .sdd_formula_explicit import SDDExplicit, SDDExplicitManager
from .cnf_formula import CNF
from .logic import Term
try:
    from pysdd.sdd import Vtree
except Exception as err:
    sdd = None

from .util import mktempfile


# noinspection PyBroadException

class SDDExplicitDefined(SDDExplicit):
    """
        This formula is using the cnf-encoding (c :- a,b = {c,a,b} v {-c,(-a v -b)}). This implies there is an
        indicator variable for each derived literal and the circuit consists of a single root node on which we do WMC.
        Evidence and querying is done by modifying the weights.
    """

    def __init__(self, sdd_auto_gc=False, **kwdargs):
        SDDExplicit.__init__(self, sdd_auto_gc=sdd_auto_gc, **kwdargs)

    def _create_manager(self, var_count = 0, vtree=None, separator=None):
        mgr = SDDExplicitDefinedManager(
            auto_gc=self.auto_gc,
            varcount=var_count,
            vtree = vtree,
            separator = separator
        )
        self.inode_manager = mgr
        return mgr

    def build_dd(self, root_key=None):
        """
        Build the SDD structure from the current self.nodes starting at index root_key.
        :param root_key: The key of the root node in self.nodes. None if the root must be True.
        """
        if root_key is None:
            root_node = self.get_manager().true()
        else:
            root_node = self.get_inode(root_key)

        self.build_constraint_dd()
        constrained_node = self.get_manager().constraint_dd
        cycle_constrained_node = self.get_manager().cycle_constraint_dd
        self._root = root_node.conjoin(constrained_node)
        self._root = root_node.conjoin(cycle_constrained_node)

class SDDExplicitDefinedManager(SDDManager):
    """
    Manager for SDDs with one root which use the explicit encoding, for example where c :- a,b is represented as
    {c,a,b} v {-c,(-a v -b)}). !Beware!, the self.nodes() of this class might be inaccessible (empty) when calling
    clean_nodes(self, root_inode).
    """

    def __init__(self, varcount=0, auto_gc=False, vtree=None, separator=None):
        """Create a new SDDExplicitManager.

        :param varcount: number of initial variables
        :type varcount: int
        :param auto_gc: use automatic garbage collection and minimization
        :type auto_gc: bool
        :param var_constraint: A variable ordering constraint. Currently only x_constrained namedtuple are allowed.
        :type var_constraint: x_constrained
        """
        SDDExplicitManager.__init__(self, varcount=varcount, auto_gc=auto_gc, vtree=vtree)
        self.separator = separator
        print(">>",self.separator)

    def _get_wmc_func(self, weights, semiring, perform_smoothing=True, normalize=True):
        """
        Get the function used to perform weighted model counting with the SddIterator. Smoothing supported.

        :param weights: The weights used during computations.
        :type weights: dict[int, tuple[Any, Any]]
        :param semiring: The semiring used for the operations.
        :param perform_smoothing: Whether smoothing must be performed. If false but semiring.is_nsp() then
            smoothing is still performed.
        :return: A WMC function that uses the semiring operations and weights, Performs smoothing if needed.
        """

        smooth_flag = perform_smoothing or semiring.is_nsp()
        smooth_flag = False

        def func_weightedmodelcounting(
            node, rvalues, expected_prime_vars, expected_sub_vars
        ):
            """ Method to pass on to SddIterator's ``depth_first`` to perform weighted model counting."""
            # print(node, rvalues)
            if rvalues is None:
                # Leaf
                if node.is_true():
                    result_weight = semiring.one()

                    # If smoothing, go over literals missed in scope
                    if smooth_flag:
                        missing_literals = (
                            expected_prime_vars
                            if expected_prime_vars is not None
                            else set()
                        )
                        missing_literals |= (
                            expected_sub_vars
                            if expected_sub_vars is not None
                            else set()
                        )

                        for missing_literal in missing_literals:
                            missing_pos_weight, missing_neg_weight = weights[
                                missing_literal
                            ]
                            missing_combined_weight = semiring.plus(
                                missing_pos_weight, missing_neg_weight
                            )
                            result_weight = semiring.times(
                                result_weight, missing_combined_weight
                            )

                    return result_weight

                elif node.is_false():
                    return semiring.zero()

                elif node.is_literal():
                    p_weight, n_weight = weights.get(abs(node.literal))
                    result_weight = p_weight if node.literal >= 0 else n_weight

                    # If smoothing, go over literals missed in scope
                    if smooth_flag:
                        lit_scope = {abs(node.literal)}

                        if expected_prime_vars is not None:
                            missing_literals = expected_prime_vars.difference(lit_scope)
                        else:
                            missing_literals = set()
                        if expected_sub_vars is not None:
                            missing_literals |= expected_sub_vars.difference(lit_scope)

                        for missing_literal in missing_literals:
                            missing_pos_weight, missing_neg_weight = weights[
                                missing_literal
                            ]
                            missing_combined_weight = semiring.plus(
                                missing_pos_weight, missing_neg_weight
                            )
                            result_weight = semiring.times(
                                result_weight, missing_combined_weight
                            )

                    return result_weight

                else:
                    raise Exception("Unknown leaf type for node {}".format(node))
            else:
                # Decision node
                if node is not None and not node.is_decision():
                    raise Exception("Expected a decision node for node {}".format(node))

                normalization = semiring.one()
                if node.vtree().position() in self.separator:
                    mc_decision =  node.model_count()
                    normalization = semiring.value(1/mc_decision)
                
                result_weight = None
                for prime_weight, sub_weight, prime_vars, sub_vars in rvalues:
                    branch_weight = semiring.times(prime_weight, sub_weight)

                    # If smoothing, go over literals missed in scope
                    if smooth_flag:
                        missing_literals = expected_prime_vars.difference(
                            prime_vars
                        ) | expected_sub_vars.difference(sub_vars)
                        for missing_literal in missing_literals:
                            missing_pos_weight, missing_neg_weight = weights[
                                missing_literal
                            ]
                            missing_combined_weight = semiring.plus(
                                missing_pos_weight, missing_neg_weight
                            )
                            branch_weight = semiring.times(
                                branch_weight, missing_combined_weight
                            )

                    # Add to current intermediate result
                    if result_weight is not None:
                        result_weight = semiring.plus(result_weight, branch_weight)
                    else:
                        result_weight = branch_weight

                return semiring.times(result_weight, normalization)

        return func_weightedmodelcounting

def dag_to_aspmc_cnf(dag):
    """
    Encode this SDD into an aspmc cnf formula
    """
    problog_cnf = CNF.create_from(dag)
    # print(problog_cnf.clauses)
    quantify = []
    for i, node, t in dag:
        if t == "atom":
            if isinstance(node.probability, Term) and  node.probability.is_float():
                p = node.probability
                n = 1-node.probability.value # works only with nologspace
                problog_cnf.add_comment(f"p weight {i} {p};{p}")
                problog_cnf.add_comment(f"p weight {-i} {n};{n}")
                quantify.append(str(i))
            else:
                o = "(1.0,1.0)"
                problog_cnf.add_comment(f"p weight {i} {o};{o}")
                problog_cnf.add_comment(f"p weight {-i} {o};{o}")
        else:
            if node.name is not None:
                o = "(1.0,1.0)"
                problog_cnf.add_comment(f"p weight {i} {o};{o}")
                problog_cnf.add_comment(f"p weight {-i} {o};{o}")

    q_comment = "p quantify " + " ".join(quantify)
    s_comment = "p semirings aspmc.semirings.probabilistic aspmc.semirings.two_nat"
    problog_cnf.add_comment(q_comment)
    problog_cnf.add_comment(s_comment)
    problog_cnf.add_comment("p quantify")
    problog_cnf.add_comment('p transform lambda w : w[0]/w[1]')
    cnf_file = mktempfile(".cnf")
    # print(problog_cnf.to_dimacs())
    with open(cnf_file, "w") as f:
        f.write(problog_cnf.to_dimacs())
    aspmc_cnf = aspmc_CNF(path=cnf_file)
    # print(aspmc_cnf)
    return aspmc_cnf

def aspmc_vtree_to_pysdd_vtree(aspmc_vtree):
    """
    Convert vtree from aspmc class to a vtree for pysdd (sdd manager)
    """
    vtree_file = mktempfile(".vtree")
    aspmc_vtree.write(vtree_file)
    pysdd_vtree = Vtree().from_file(vtree_file)
    return pysdd_vtree

def special_cnf(source):
    
    ld = CNF()

    # build
    with Timer("Compiling %s" % ld.__class__.__name__):
        identifier = 0
        line_map = (
            dict()
        )  # line in source mapped to line in ld {src_line: (negated, positive, combined)}
        line = 1  # current line (line_id)
        node_to_indicator = {}  # {node : indicator_node}
        root_nodes = []
        for line_id, clause, c_type in source:
            if c_type == "atom":
                result = ld.add_atom(identifier)
                identifier += 1
                line_map[line_id] = (-result, result, result)
                line += 1
            elif c_type == "conj":
                and_nodes = [
                    line_map[abs(src_line)][src_line > 0]
                    for src_line in clause.children
                ]
                negated_and_nodes = [
                    line_map[abs(src_line)][src_line < 0]
                    for src_line in clause.children
                ]

                if clause.name is None:
                    result = ld.add_and(and_nodes, source.get_name(line_id))
                    result_neg = ld.add_or(
                        negated_and_nodes, source.get_name(-line_id)
                    )
                    line_map[line_id] = (result_neg, result, result)
                    line += 2
                else:
                    # head
                    head = ld.add_atom(
                        identifier=identifier,
                        probability=True,
                        group=None,
                        name=clause.name,
                    )
                    identifier += 1
                    # body
                    body = ld.add_and(and_nodes)  # source.get_name(i))
                    negated_body = ld.add_or(negated_and_nodes)
                    # combined
                    combined_false = ld.add_and([-head, negated_body])
                    combined_true = ld.add_and([head, body])
                    combined = ld.add_or([combined_false, combined_true])

                    node_to_indicator[combined] = head
                    line_map[line_id] = (combined_false, combined_true, combined)
                    line += 6
                    root_nodes.append(combined)

            elif c_type == "disj":
                or_nodes = [
                    line_map[abs(src_line)][src_line > 0]
                    for src_line in clause.children
                ]
                negated_or_nodes = [
                    line_map[abs(src_line)][src_line < 0]
                    for src_line in clause.children
                ]

                if clause.name is None:
                    result = ld.add_or(or_nodes, source.get_name(line_id))
                    result_neg = ld.add_and(
                        negated_or_nodes, source.get_name(-line_id)
                    )
                    line_map[line_id] = (result_neg, result, result)
                    line += 2
                else:
                    # head
                    head = ld.add_atom(
                        identifier=identifier,
                        probability=True,
                        group=None,
                        name=clause.name,
                    )
                    identifier += 1
                    # body
                    body = ld.add_or(or_nodes)  # source.get_name(i))
                    negated_body = ld.add_and(negated_or_nodes)
                    # combined
                    combined_false = ld.add_and([-head, negated_body])
                    combined_true = ld.add_and([head, body])
                    combined = ld.add_or([combined_false, combined_true])

                    node_to_indicator[combined] = head
                    line_map[line_id] = (combined_false, combined_true, combined)
                    line += 6
                    root_nodes.append(combined)

            else:
                raise TypeError("Unknown node type")

        for name, node, label in source.get_names_with_label():
            if (
                label == ld.LABEL_QUERY
                or label == ld.LABEL_EVIDENCE_MAYBE
                or label == ld.LABEL_EVIDENCE_NEG
                or label == ld.LABEL_EVIDENCE_POS
            ):  # TODO required?
                if node is None or node == 0:
                    ld.add_name(name, node, label)
                else:
                    mapped_line = line_map[abs(node)][2]
                    sign = -1 if node < 0 else 1
                    if (
                        node_to_indicator.get(mapped_line) is not None
                    ):  # Change internal node indicator
                        ld.add_name(
                            name, sign * node_to_indicator[mapped_line], label
                        )
                    else:
                        ld.add_name(name, sign * mapped_line, label)

        if len(root_nodes) > 0:
            root_key = ld.add_and(root_nodes, name=None)
        else:
            root_key = None

        rename = {n:node_to_indicator[line_map[n][2]] for n in line_map if line_map[n][2] in node_to_indicator}
        # Copy constraints
        for c in source.constraints():
            ld.add_constraint(c.copy(rename))
        for c in source.cycle_constraints():
            ld.add_cycle_constraint(c.copy(rename))
    
    return ld


@transform(LogicDAG, SDDExplicitDefined)
def build_explicit_from_logicdag(source, destination, **kwdargs):
    """Build an SDD2 from a LogicDAG.

    :param source: source formula
    :type source: LogicDAG
    :param destination: destination formula
    :type destination: SDDExplicit
    :param kwdargs: extra arguments
    :return: destination
    :rtype: SDDExplicit
    """

    print(source)

    # print(line_map)
    # v = mktempfile(".vtreez")
    # ld.get_manager().get_manager().vtree().save(v.encode())
    # print(ld.get_manager().get_manager().vtree().dot())
    # print(ld.get_manager().get_manager().vtree().var_count())

    # cnf = special_cnf(source)
    print("________")
    cnf_aspmc = dag_to_aspmc_cnf(source)
    print(cnf_aspmc)
    sep, aspmc_vtree = tree_from_cnf(cnf_aspmc, tree_type=aspmc_Vtree)
    pysdd_vtree = aspmc_vtree_to_pysdd_vtree(aspmc_vtree)
    print(pysdd_vtree.dot())
    destination._create_manager(vtree=pysdd_vtree, separator=sep)

    with Timer("Compiling %s" % destination.__class__.__name__):
        identifier = 0
        line_map = (
            dict()
        )  # line in source mapped to line in destination {src_line: (negated, positive, combined)}
        line = 1  # current line (line_id)
        node_to_indicator = {}  # {node : indicator_node}
        root_nodes = []
        for line_id, clause, c_type in source:
            if c_type == "atom":
                result = destination.add_atom(
                    identifier,
                    clause.probability,
                    clause.group,
                    source.get_name(line_id),
                    cr_extra=False,
                )
                identifier += 1

                line_map[line_id] = (-result, result, result)
                line += 1
            elif c_type == "conj":
                and_nodes = [
                    line_map[abs(src_line)][src_line > 0]
                    for src_line in clause.children
                ]
                negated_and_nodes = [
                    line_map[abs(src_line)][src_line < 0]
                    for src_line in clause.children
                ]

                if clause.name is None:
                    result = destination.add_and(and_nodes, source.get_name(line_id))
                    result_neg = destination.add_or(
                        negated_and_nodes, source.get_name(-line_id)
                    )
                    line_map[line_id] = (result_neg, result, result)
                    line += 2
                else:
                    # head
                    head = destination.add_atom(
                        identifier=identifier,
                        probability=True,
                        group=None,
                        name=clause.name,
                    )
                    identifier += 1
                    # body
                    body = destination.add_and(and_nodes)  # source.get_name(i))
                    negated_body = destination.add_or(negated_and_nodes)
                    # combined
                    combined_false = destination.add_and([-head, negated_body])
                    combined_true = destination.add_and([head, body])
                    combined = destination.add_or([combined_false, combined_true])

                    node_to_indicator[combined] = head
                    line_map[line_id] = (combined_false, combined_true, combined)
                    line += 6
                    root_nodes.append(combined)

            elif c_type == "disj":
                or_nodes = [
                    line_map[abs(src_line)][src_line > 0]
                    for src_line in clause.children
                ]
                negated_or_nodes = [
                    line_map[abs(src_line)][src_line < 0]
                    for src_line in clause.children
                ]

                if clause.name is None:
                    result = destination.add_or(or_nodes, source.get_name(line_id))
                    result_neg = destination.add_and(
                        negated_or_nodes, source.get_name(-line_id)
                    )
                    line_map[line_id] = (result_neg, result, result)
                    line += 2
                else:
                    # head
                    head = destination.add_atom(
                        identifier=identifier,
                        probability=True,
                        group=None,
                        name=clause.name,
                    )
                    identifier += 1
                    # body
                    body = destination.add_or(or_nodes)  # source.get_name(i))
                    negated_body = destination.add_and(negated_or_nodes)
                    # combined
                    combined_false = destination.add_and([-head, negated_body])
                    combined_true = destination.add_and([head, body])
                    combined = destination.add_or([combined_false, combined_true])

                    node_to_indicator[combined] = head
                    line_map[line_id] = (combined_false, combined_true, combined)
                    line += 6
                    root_nodes.append(combined)

            else:
                raise TypeError("Unknown node type")

        for name, node, label in source.get_names_with_label():
            if (
                label == destination.LABEL_QUERY
                or label == destination.LABEL_EVIDENCE_MAYBE
                or label == destination.LABEL_EVIDENCE_NEG
                or label == destination.LABEL_EVIDENCE_POS
            ):  # TODO required?
                if node is None or node == 0:
                    destination.add_name(name, node, label)
                else:
                    mapped_line = line_map[abs(node)][2]
                    sign = -1 if node < 0 else 1
                    if (
                        node_to_indicator.get(mapped_line) is not None
                    ):  # Change internal node indicator
                        destination.add_name(
                            name, sign * node_to_indicator[mapped_line], label
                        )
                    else:
                        destination.add_name(name, sign * mapped_line, label)

        if len(root_nodes) > 0:
            root_key = destination.add_and(root_nodes, name=None)
        else:
            root_key = None

        rename = {n:node_to_indicator[line_map[n][2]] for n in line_map if line_map[n][2] in node_to_indicator}
        # Copy constraints
        for c in source.constraints():
            destination.add_constraint(c.copy(rename))
        for c in source.cycle_constraints():
            destination.add_cycle_constraint(c.copy(rename))

        destination.build_dd(root_key)

    # print(ld)
    # print("--------")
    # print(destination)
    # print(destination.atomcount)
    # print(ld.get_weights())
    # print(destination.get_weights())
    # print(line_map)
    # print(destination.get_manager().get_manager().var_count())
    # print(destination.sdd_to_dot(None, show_id=True))

    return destination
