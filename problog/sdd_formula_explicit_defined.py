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
# from .aspmc.aspmc.compile.vtree import vtree as vtree
from .aspmc.aspmc.compile.cnf import CNF as ASPMC_CNF
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


    def update_vtree(self):
        cnf = self.sdd_to_aspmc_cnf()
        aspmc_vtree = tree_from_cnf(cnf, tree_type=Vtree)
        print(aspmc_vtree)
        pysdd_vtree = self.aspmc_vtree_to_pysdd_vtree(aspmc_vtree)
        self.inode_manager = self.get_manager().from_vtree()

    def sdd_to_aspmc_cnf(self):
        """
        Encode this SDD into an aspmc cnf formula
        """
        problog_cnf = CNF.create_from(self)
        quantify = []
        for i, node, t in self:
            if t == "atom":
                if isinstance(node.probability, Term) and  node.probability.is_float():
                    p = node.probability
                    n = 1-node.probability.value # works only with nologspace
                    problog_cnf.add_comment(f"p weight {i} {p};{p};{p};{p};{p};{p};")
                    problog_cnf.add_comment(f"p weight {-i} {n};{n};{n};{n};{n};{n};")
                    quantify.append(str(i))
        q_comment = "p quantify " + " ".join(quantify)
        s_comment = "p semirings aspmc.semirings.probabilistic aspmc.semirings.two_nat"
        problog_cnf.add_comment(q_comment)
        problog_cnf.add_comment(s_comment)
        problog_cnf.add_comment("p quantify")
        problog_cnf.add_comment('p transform lambda w : w[0]/w[1]')
        cnf_file = mktempfile(".cnf")
        print(problog_cnf.to_dimacs())
        with open(cnf_file, "w") as f:
            f.write(problog_cnf.to_dimacs())
        aspmc_cnf = ASPMC_CNF(path=cnf_file)
        # print(aspmc_cnf)
        return aspmc_cnf

    def aspmc_vtree_to_pysdd_vtree(self, aspmc_vtree):
        """
        Convert vtree from aspmc class to a vtree for pysdd (sdd manager)
        """
        pass

    def build_dd(self, root_key=None):
        """
        Build the SDD structure from the current self.nodes starting at index root_key.
        :param root_key: The key of the root node in self.nodes. None if the root must be True.
        """
        # Set root
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

    def __init__(self, varcount=0, auto_gc=False, vtree=None):
        """Create a new SDDExplicitManager.

        :param varcount: number of initial variables
        :type varcount: int
        :param auto_gc: use automatic garbage collection and minimization
        :type auto_gc: bool
        :param var_constraint: A variable ordering constraint. Currently only x_constrained namedtuple are allowed.
        :type var_constraint: x_constrained
        """
        SDDExplicitManager.__init__(self, varcount=varcount, auto_gc=auto_gc)


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

    # same as SDDExplicit but without the x-constrained part

    # Get init varcount
    init_varcount = kwdargs.get("init_varcount", -1)
    
    if init_varcount == -1:
        init_varcount = source.atomcount
        for _, clause, c_type in source:
            if c_type != "atom" and clause.name is not None:
                init_varcount += 1
    destination.init_varcount = init_varcount

    # build
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
        destination.update_vtree()

    return destination
