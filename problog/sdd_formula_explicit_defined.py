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
# sys.path.append(os.path.join(os.path.dirname(__file__), 'aspmc'))
# from .aspmc.aspmc.compile.vtree import Vtree as aspmc_Vtree
# from .aspmc.aspmc.compile.cnf import CNF as aspmc_CNF
# from .aspmc.aspmc.compile.constrained_compile import tree_from_cnf 
from aspmc.compile.vtree import Vtree as aspmc_Vtree
from aspmc.compile.cnf import CNF as aspmc_CNF
from aspmc.compile.constrained_compile import tree_from_cnf 

from .formula import LogicDAG, LogicFormula
from .core import transform
from .errors import InconsistentEvidenceError
from .util import Timer
from .evaluator import SemiringProbability, SemiringLogProbability
from .sdd_formula import SDDEvaluator, SDD, x_constrained
from .forward import _ForwardSDD, ForwardInference
from .sdd_formula import SDDManager

from .sdd_formula_explicit import SDDExplicit, SDDExplicitManager, x_constrained_named
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

    def _create_manager(self, varcount = 0, var_constraint=None, vtree=None, separator=None):
        mgr = SDDExplicitDefinedManager(
            auto_gc=self.auto_gc,
            varcount=varcount,
            var_constraint=var_constraint,
            vtree = vtree,
            separator = separator
        )
        self.inode_manager = mgr
        return mgr

class SDDExplicitDefinedManager(SDDExplicitManager):
    """
    Manager for SDDs with one root which use the explicit encoding, for example where c :- a,b is represented as
    {c,a,b} v {-c,(-a v -b)}). !Beware!, the self.nodes() of this class might be inaccessible (empty) when calling
    clean_nodes(self, root_inode).
    """

    def __init__(self, varcount=0, var_constraint=None,  auto_gc=False, vtree=None, separator=None):
        """Create a new SDDExplicitManager.

        :param varcount: number of initial variables
        :type varcount: int
        :param auto_gc: use automatic garbage collection and minimization
        :type auto_gc: bool
        :param var_constraint: A variable ordering constraint. Currently only x_constrained namedtuple are allowed.
        :type var_constraint: x_constrained
        """
        SDDExplicitManager.__init__(self, varcount=varcount, auto_gc=auto_gc, var_constraint=var_constraint, vtree=vtree)
        self.separator = separator


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

    # print(source)

    cnf_aspmc = dag_to_aspmc_cnf(source)
    # print(cnf_aspmc)
    sep, aspmc_vtree = tree_from_cnf(cnf_aspmc, tree_type=aspmc_Vtree)
    pysdd_vtree = aspmc_vtree_to_pysdd_vtree(aspmc_vtree)
    # print(pysdd_vtree.dot())
    var_ids = []
    for id, clause, c_type in source:
        if c_type == "atom":
            if type(clause.probability) != bool:
                var_ids.append(id)

    destination._create_manager(
        varcount=pysdd_vtree.var_count(),
        var_constraint= x_constrained(X=var_ids), 
        vtree=pysdd_vtree, 
        separator=sep
    )

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


    # print("--------")
    # print(destination)
    # print(destination.sdd_to_dot(None, show_id=True))

    return destination
