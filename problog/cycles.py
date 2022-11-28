"""
problog.cycles - Cycle-breaking
-------------------------------

Cycle breaking in propositional formulae.

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

from .logic import Term
from .core import transform
from .util import Timer
from .formula import LogicFormula, LogicDAG, LogicGraph
from .constraint import TrueConstraint

from collections import defaultdict
import logging

from problog import constraint

cycle_var_prefix = "problog_cv_"

# noinspection PyUnusedLocal
# @transform(LogicFormula, LogicDAG)
def break_cycles(source, target, translation=None, **kwdargs):
    """Break cycles in the source logic formula.

    :param source: logic formula with cycles
    :param target: target logic formula without cycles
    :param kwdargs: additional arguments (ignored)
    :return: target
    """

    logger = logging.getLogger("problog")
    with Timer("Cycle breaking"):
        cycles_broken = set()
        content = set()
        if translation is None:
            translation = defaultdict(list)

        for q, n, l in source.labeled():
            if source.is_probabilistic(n):
                newnode = _break_cycles(
                    source, target, n, [], cycles_broken, content, translation
                )
            else:
                newnode = n
            target.add_name(q, newnode, l)

        # TODO copy constraints

        translation = defaultdict(list)
        for q, n, v in source.evidence_all():
            if source.is_probabilistic(n):
                newnode = _break_cycles(
                    source,
                    target,
                    abs(n),
                    [],
                    cycles_broken,
                    content,
                    translation,
                    is_evidence=True,
                )
            else:
                newnode = n
            if n is not None and n < 0:
                newnode = target.negate(newnode)
            if v > 0:
                target.add_name(q, newnode, target.LABEL_EVIDENCE_POS)
            elif v < 0:
                target.add_name(q, newnode, target.LABEL_EVIDENCE_NEG)
            else:
                target.add_name(q, newnode, target.LABEL_EVIDENCE_MAYBE)

        logger.debug("Ground program size: %s", len(target))
        return target


def _break_cycles(
    source,
    target,
    nodeid,
    ancestors,
    cycles_broken,
    content,
    translation,
    is_evidence=False,
):
    negative_node = nodeid < 0
    nodeid = abs(nodeid)

    if not is_evidence and not source.is_probabilistic(
        source.get_evidence_value(nodeid)
    ):
        ev = source.get_evidence_value(nodeid)
        if negative_node:
            return target.negate(ev)
        else:
            return ev
    elif nodeid in ancestors:
        cycles_broken.add(nodeid)
        return None  # cyclic node: node is False
    elif nodeid in translation:
        ancset = frozenset(ancestors + [nodeid])
        for newnode, cb, cn in translation[nodeid]:
            # We can reuse this previous node iff
            #   - no more cycles have been broken that should not be broken now
            #       (cycles broken is a subset of ancestors)
            #   - no more cycles should be broken than those that have been broken in the previous
            #       (the previous node does not contain ancestors)

            if cb <= ancset and not ancset & cn:
                cycles_broken |= cb
                content |= cn
                if negative_node:
                    return target.negate(newnode)
                else:
                    return newnode

    child_cycles_broken = set()
    child_content = set()

    node = source.get_node(nodeid)
    nodetype = type(node).__name__
    if nodetype == "atom":
        newnode = target.add_atom(
            node.identifier, node.probability, node.group, node.name
        )
    else:
        children = [
            _break_cycles(
                source,
                target,
                child,
                ancestors + [nodeid],
                child_cycles_broken,
                child_content,
                translation,
                is_evidence,
            )
            for child in node.children
        ]
        newname = node.name
        if newname is not None and child_cycles_broken:
            newfunc = (
                cycle_var_prefix
                + newname.functor
                + "_cb_"
                + str(len(translation[nodeid]))
            )
            newname = Term(newfunc, *newname.args)
        if nodetype == "conj":
            newnode = target.add_and(children, name=newname)
        else:
            newnode = target.add_or(children, name=newname)

        if target.is_probabilistic(newnode):
            # Don't add the node if it is None
            # Also: don't add atoms (they can't be involved in cycles)
            content.add(nodeid)

    translation[nodeid].append(
        (newnode, child_cycles_broken, child_content - child_cycles_broken)
    )
    content |= child_content
    cycles_broken |= child_cycles_broken

    if negative_node:
        return target.negate(newnode)
    else:
        return newnode


@transform(LogicGraph, LogicDAG)
def break_neg_cycles(source, target, translation=None, **kwdargs):
    """Break cycles in the source logic formula.

    :param source: logic formula with negative cycles
    :param target: target logic formula without cycles
    :param kwdargs: additional arguments (ignored)
    :return: target
    """
    logger = logging.getLogger("problog")
    # print(source)
    # print(source.to_prolog())
    # print("*********")
    target.vtree = source.vtree
    with Timer("Cycle breaking"):
        # cycles_broken = {k:[] for k in range(1,len(source))}
        content = set()
        if translation is None:
            translation = defaultdict(list)
        for q, n, l in source.labeled():
            newnode, todo_broken, visited = _break_neg_cycles(
                source, target, n, [], None, content, translation
            )
            content |= visited
            target.add_name(q, newnode, l)
        for l, id in source.evidence():  #
            new_id, _, _ = translation[abs(id)]
            target.add_evidence(name=l, key=new_id, value=(id > 0))
        # print(target.to_prolog())
        # print(target)
        return target


def _break_neg_cycles(
    source,
    target,
    nodeid,
    ancestors,
    cycles_broken,
    content,
    translation,
    is_evidence=False,
    lvl=0,
):
    negative_node = nodeid < 0
    nodeid = abs(nodeid)
    node = source.get_node(nodeid)
    nodetype = type(node).__name__

    # print(nodeid, nodetype, ancestors)

    if nodeid in ancestors:
        """
        Cycle found, since positive cycles should already be broken, it involves negation:
        replace current node with a new var newnode2 and the parent with a new variable newnode1.
        Then recursively replace them with the new variables and add the constraints that bind the
        new variables to the original ones
        """
        cycle_nodes = ancestors[ancestors.index(nodeid) :]
        cycle = "_".join([str(i) for i in cycle_nodes])
        # print("cycle", nodeid, ancestors, cycle)
        newname1 = source.get_node(ancestors[-1]).name
        newname2 = node.name
        if newname1 is None:
            newname1 = Term(str(ancestors[-1]))
        if newname2 is None:
            newname1 = Term(str(nodeid))
        newfunc1 = cycle_var_prefix + newname1.functor + "_" + cycle
        newfunc2 = cycle_var_prefix + newname2.functor + "_" + cycle
        newname1 = Term(newfunc1, *newname1.args)
        newname2 = Term(newfunc2, *newname2.args)
        newnode1 = target.add_atom(
            newfunc1, probability=target.WEIGHT_NEUTRAL, name=newname1
        )
        newnode2 = target.add_atom(
            newfunc2, probability=target.WEIGHT_NEUTRAL, name=newname2
        )

        # ancestors[-1] : add newnode1 to the children for first cycle breaking var
        #               : add consistency constraints between ancestors[-1] and newnode1
        # ancestors[-2] : add newnode2 to the children for second cycle breaking var
        #               : but second cycle breaking var is tied to current node (ancestors[0])
        #                 so carry nodeid to the origin of the cycle to get the corresponding new id
        #                 for the cycle breaking var constraints
        # newnode1/newnode2: the two new vars breaking the cycle between current node and parent
        # nodeid: depending how long the cycle is, we remember that when we get back to nodeid in the recursion
        #         we have to add the constraint between its cb variable (newnode2) and the new id in target
        cycle_info = [(ancestors[-1], ancestors[-2], newnode1, newnode2, nodeid)]
        # print(cycle_info)
        return newnode2, cycle_info, content
    else:
        broken_cycles = []
        constraints = []
        if nodeid in content:  # already visited and done
            # print(nodeid, " visited")
            id, cb, cont = translation[nodeid]
            content |= cont
            return id, cb, content
        content.add(nodeid)
        if nodetype == "atom":
            newnode = target.add_atom(
                node.identifier, node.probability, node.group, node.name
            )
        else:
            # ind = "  " * lvl
            # print(ind, nodeid, node.children)
            children = []
            for child in node.children:
                new_c, child_broken_cycles, content_c = _break_neg_cycles(
                    source,
                    target,
                    child,
                    ancestors + [nodeid],
                    cycles_broken,
                    content,
                    translation,
                    is_evidence,
                    lvl=lvl + 1,
                )
                skip = False
                for cycle in child_broken_cycles:
                    if cycle[0] == nodeid:
                        """
                        we just closed a cycle: replace child with the new var
                        """
                        if child < 0:
                            children.append(target.negate(cycle[3]))
                        else:
                            children.append(cycle[3])
                        constraints.append(cycle[2])
                        skip = skip or True
                        broken_cycles.append(
                            cycle
                        )  # and propagate the info for the second var
                    elif cycle[1] == nodeid:
                        """
                        a cycle was closed by the child: add the second cb var
                        """
                        if child < 0:
                            children.append(target.negate(cycle[2]))
                        else:
                            children.append(cycle[2])
                        skip = skip or True
                        if (
                            cycle[4] == nodeid
                        ):  # if newnode2 is for this node add constraints and forget
                            constraints.append(cycle[3])
                        else:  # move upwards until we find the origin of the cycle
                            broken_cycles.append(cycle)
                    elif (
                        cycle[4] == nodeid
                    ):  # if newnode2 is for this node add constraints and forget
                        constraints.append(cycle[3])
                    else:  # move upwards until we find the origin of the cycle
                        broken_cycles.append(cycle)
                # print(ind, nodeid, child, skip, children, broken_cycles)
                if not skip:
                    children.append(new_c)
                content |= content_c
            # print(ind, "children of ", nodeid, ":", children)
            newname = node.name
            if nodetype == "conj":
                newnode = target.add_and(children, name=newname)
            else:
                newnode = target.add_or(children, name=newname)
            # print(ind, "newid for ", nodeid, ":", newnode)

        translation[nodeid] = (newnode, broken_cycles, content)

        # print("\t",nodeid, constraints, broken_cycles)
        for k in constraints:  # add all the iff constraints involving newnode
            c1 = target.add_and([newnode, k])
            c2 = target.add_and([-newnode, -k])
            c = target.add_or([c1, c2], name=Term(f"iff_{newnode}_{k}"))
            target.add_cycle_constraint(TrueConstraint(c))

        if negative_node:
            return target.negate(newnode), broken_cycles, content
        else:
            return newnode, broken_cycles, content
