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
from collections import namedtuple

from .formula import LogicDAG, LogicFormula
from .core import transform
from .errors import InconsistentEvidenceError
from .util import Timer
from .evaluator import SemiringProbability, SemiringLogProbability
from .sdd_formula import SDDEvaluator, SDD, x_constrained
from .forward import _ForwardSDD, ForwardInference
from .sdd_formula import SDDManager


from .util import mktempfile


# noinspection PyBroadException
try:
    from pysdd import sdd
    from pysdd.iterator import SddIterator
except Exception as err:
    sdd = None


class SDDExplicit(SDD):
    """
        This formula is using the cnf-encoding (c :- a,b = {c,a,b} v {-c,(-a v -b)}). This implies there is an
        indicator variable for each derived literal and the circuit consists of a single root node on which we do WMC.
        Evidence and querying is done by modifying the weights.
    """

    transform_preference = 11

    def __init__(self, sdd_auto_gc=False, **kwdargs):
        SDD.__init__(self, sdd_auto_gc=sdd_auto_gc, **kwdargs)
        self._root = None

    def _create_manager(self):
        return SDDExplicitManager(
            auto_gc=self.auto_gc,
            var_constraint=self.var_constraint,
            varcount=self.init_varcount,
        )

    def _create_evaluator(self, semiring, weights, **kwargs):
        return SDDExplicitEvaluator(self, semiring, weights, **kwargs)

    def get_root_inode(self):
        """ Get the current root inode. This includes the constraints. """
        if self._root is None:
            self._root = self.get_manager().true()
        return self._root

    def to_formula(self):
        """
        Extracts a LogicFormula from the SDD.
        """
        formula = LogicFormula(keep_order=True)
        root_node = self.get_root_inode()
        i = self._to_formula(formula, root_node, {})
        for n, q, l in self.labeled():
            formula.add_name(n, i, l)

        return formula

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
        # print("SDD built")

    # def cleanup_inodes(self):
    #    """
    #    Dereferences all but the root node and empties self.nodes. ! Do not use self.nodes afterwards !
    #    This is usually called right after the build process, build_dd(root_key).
    #    """
    #    self.get_manager().clean_nodes(self.get_root_inode())

    def to_dot(self, *args, **kwargs):
        if kwargs.get("use_internal"):
            return self.to_internal_dot(node=self.get_root_inode())
        else:
            return self.to_formula().to_dot(*args, **kwargs)

    def sdd_to_dot(self, node, litnamemap=None, show_id=False, merge_leafs=False):
        """
        SDD for the given node, formatted for use with Graphviz dot. This method provides more control over the used
        symbols than to_internal_dot (see litnamemap). Primes are given by a dotted line, subs by a full line.

        :param node: The node to get the dot from.
        :param litnamemap: A dictionary providing the symbols to use. The following options are available:
            1. literals, e.g. {1:'A', -1:'-A', ...},
            2. True/False, e.g. {true':'1', 'false':'0'}
            3. And/Or e.g. {'mult':'x', 'add':'+'}
        :param show_id: Whether to display the ids of each sdd node.
        :param merge_leafs: Whether to merge the same leaf nodes. True results in less nodes but makes it harder to
        render without having crossing lines.
        :return: The dot format of the given node. When node is None, the root node is used.
        :rtype: str
        """
        used_node = node if node is not None else self.get_root_inode()
        return super().sdd_to_dot(
            node=used_node,
            litnamemap=litnamemap,
            show_id=show_id,
            merge_leafs=merge_leafs,
        )

class SDDExplicitManager(SDDManager):
    """
    Manager for SDDs with one root which use the explicit encoding, for example where c :- a,b is represented as
    {c,a,b} v {-c,(-a v -b)}). !Beware!, the self.nodes() of this class might be inaccessible (empty) when calling
    clean_nodes(self, root_inode).
    """

    def __init__(self, varcount=0, auto_gc=False, var_constraint=None, vtree=None):
        """Create a new SDDExplicitManager.

        :param varcount: number of initial variables
        :type varcount: int
        :param auto_gc: use automatic garbage collection and minimization
        :type auto_gc: bool
        :param var_constraint: A variable ordering constraint. Currently only x_constrained namedtuple are allowed.
        :type var_constraint: x_constrained
        """
        SDDManager.__init__(
            self, varcount=varcount, auto_gc=auto_gc, var_constraint=var_constraint, vtree=vtree
        )
        self._x_node = None

    @staticmethod
    def create_from_SDDManager(sddmgr):
        new_mgr = SDDExplicitManager()
        vars(new_mgr).update(vars(sddmgr))
        return new_mgr

    def get_vtree_vars(self, vtree):
        # optimize by calling once
        if vtree.is_leaf():
            return [vtree.var()]
        else:
            vars = []
            if vtree.left() is not None:
                vars += self.get_vtree_vars(vtree.left())
            if vtree.right() is not None:
                vars += self.get_vtree_vars(vtree.right())
            return vars

    def get_x_node(self):
        if self._x_node is None:
            vtree = self.get_manager().vtree()
            return self._get_x_node(vtree)
        else:
            return self._x_node
    
    def _get_x_node(self, vtree):
        if vtree is None:
            return None
        else:
            if vtree.left() is not None:
                vleft = set(self.get_vtree_vars(vtree.left()))
            else:
                vleft = set()
            if vtree.right() is not None:
                vright = set(self.get_vtree_vars(vtree.right()))
            else:
                vright = set()
            x_constrained = {i for i in range(0,len(self.x_constraint)) if self.x_constraint[i]}
            vars_in_x = x_constrained & vleft.union(vright)
            if len(vars_in_x) == 0:
                self._x_node = vtree.position()
            else:
                self._get_x_node(vtree.right())
            return self._x_node

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

        def func_weightedmodelcounting(
            node, rvalues, expected_prime_vars, expected_sub_vars
        ):
            """ Method to pass on to SddIterator's ``depth_first`` to perform weighted model counting."""
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
                if node.vtree().position() == self.get_x_node():
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
    # Error: wrong refcounts. parent.deref() = all descendents.deref()
    # def clean_nodes(self, root_inode):
    #    """
    #    Cleans up all references except root_inode's. This means all other nodes will be dereferenced.
    #    ! Beware, after calling this self.nodes will be empty, do not use it afterwards !
    #    :param root_inode: The node not to dereference.
    #    :type root_inode: SddNode
    #    """
    #    for node in self.nodes:
    #        if node is not None and node.id != root_inode.id:
    #            while node.ref_count() > 0:
    #                node.deref()
    #    self.nodes = []


class SDDExplicitEvaluator(SDDEvaluator):
    def __init__(self, formula, semiring, weights=None, **kwargs):
        SDDEvaluator.__init__(self, formula, semiring, weights, **kwargs)
        # print(self.formula.sdd_to_dot(self.formula.get_manager().cycle_constraint_dd, litnamemap=self.formula.lnm, show_id=True))

    def propagate(self):
        self._initialize()
        self.evaluate_evidence(recompute=True)
        self.normalization = self._evidence_weight

    def _evaluate_evidence(self, recompute=False):
        if self._evidence_weight is None or recompute:
            self._evidence_weight = self._evaluate_root()
            if self.semiring.is_zero(self._evidence_weight):
                raise InconsistentEvidenceError(context=" during compilation")
        # print("Ev weight", self._evidence_weight)
        return self._evidence_weight

    def evaluate_fact(self, node):
        return self.evaluate(node)

    def evaluate_custom(self, node):
        return self.evaluate(self, node)

    def evaluate(self, node, normalize=True):
        # print("evaluating ", node, self._evidence_weight)
        # Trivial case: node is deterministically True or False
        if node == self.formula.TRUE:
            if not self.semiring.is_nsp():
                result = self.semiring.one()
            else:
                # calculates weight of root node.
                result = self._evaluate_root()
                if normalize:
                    result = self.semiring.normalize(result, self._get_z())

        elif node is self.formula.FALSE:
            result = self.semiring.zero()
        else:
            # Set query weight
            index = self.formula.atom2var[abs(node)]
            p_orig, n_orig = self.weights.get(index)
            self._set_value(index, (node > 0))
            # for c in self.formula.cycle_constraints():
            #     self._set_value(c.node, (c.node > 0))

            # Calculate result
            result = self._evaluate_root()

            # Restore query weight
            self.set_weight(index, p_orig, n_orig)

            # print("result: %s" % result)
            # print("normalization: %s" % self._get_z())
            # print("normalizing %s with %s and result before %s" % (normalize, self._get_z(), result))
            if normalize:
                result = self.semiring.normalize(result, self._get_z())
            # print("normalized result: %s" % result)
            # print("----------------------------------------")

        return self.semiring.result(result, self.formula)

    def _evaluate_root(self):
        """
        Evaluate the circuit (root node) with the current weights (self.weights)
        :return: The WMC of the circuit with the current weights
        """
        pr_semiring = isinstance(
            self.semiring, (SemiringLogProbability, SemiringProbability)
        )
        query_def_inode = self.formula.get_root_inode()
        return self._get_manager().wmc(
            query_def_inode,
            self.weights,
            self.semiring,
            pr_semiring=pr_semiring,
            perform_smoothing=True,
            smooth_to_root=True,
        )

    def _reset_value(self, index, pos, neg):
        self.set_weight(index, pos, neg)

    def _set_value(self, index, value):
        """Set value for given node.

        :param index: index of node (for which atom2var[atom] = index)
        :param value: True if we only want a positive weight and set the negative weight to semiring.zero().
                      Else, we only want the negative weight and set the positive weight to semiring.zero().
        """
        if value:
            pos = self.weights[index][0]
            self.set_weight(index, pos, self.semiring.zero())
        else:
            neg = self.weights[index][1]
            self.set_weight(index, self.semiring.zero(), neg)


x_constrained_named = namedtuple(
    "x_constrained", "X_named"
)  # X_named = list of literal names that have to appear before the rest


@transform(LogicDAG, SDDExplicit)
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
    # print("---")

    # Get init varcount
    init_varcount = kwdargs.get("init_varcount", -1)
    var_constraint_named = kwdargs.get("var_constraint", None)
    
    vcn = []
    for _, clause, c_type in source:
        if c_type == "atom":
            if type(clause.probability) != bool:
                vcn.append(clause.name)
    var_constraint_named =  x_constrained_named(vcn)

    if var_constraint_named is not None and isinstance(
        var_constraint_named, x_constrained_named
    ):
        var_names = {var: True for var in var_constraint_named.X_named}
        var_ids = []
        atomcount = 1
        for _, clause, c_type in source:
            if clause.name is not None:
                if var_names.get(clause.name, False):
                    var_ids.append(atomcount)
                atomcount += 1
        destination.var_constraint = x_constrained(X=var_ids)

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

    return destination


@transform(_ForwardSDD, SDDExplicit)
def build_explicit_from_forwardsdd(source, destination, **kwdargs):
    """Build an SDDExplicit from a _ForwardSDD.

    :param source: source formula
    :type source: _ForwardSDD
    :param destination: destination formula
    :type destination: SDDExplicit
    :param kwdargs: extra arguments
    :return: destination
    :rtype: SDDExplicit
    """
    with Timer("Compiling %s" % destination.__class__.__name__):
        ForwardInference.build_dd(source)

        # Make sure all atoms exist in atom2var.
        for name, node, label in source.labeled():
            if source.is_probabilistic(node):
                source.get_inode(node)

        source.copy_to_noref(destination)
        destination.inode_manager = SDDExplicitManager.create_from_SDDManager(
            destination.inode_manager
        )

        root_nodes = []
        var_to_indicator = dict()
        for line_id, clause, c_type in destination:
            if (c_type == "conj" or c_type == "disj") and clause.name is not None:
                # head
                head = destination.add_atom(
                    identifier=destination.get_next_atom_identifier(),
                    probability=True,
                    group=None,
                    name=destination.get_name(line_id),
                )
                var_to_indicator[line_id] = head

                body = line_id

                combined_true = destination.add_and([head, body])
                combined_false = destination.add_and([-head, -body])
                combined = destination.add_or([combined_false, combined_true])

                root_nodes.append(combined)

        if len(root_nodes) > 0:
            root_key = destination.add_and(root_nodes, name=None)
        else:
            root_key = None

        for name, node, label in source.get_names_with_label():
            new_node = None
            if node is not None:
                new_node = var_to_indicator.get(abs(node), abs(node))
                if node < 0:
                    new_node = -new_node
            destination.add_name(name, new_node, label)

        # query_nodes = destination.queries()
        # evidence_nodes_all = destination.evidence_all()
        # evidence_nodes = destination.evidence()

        destination.build_dd(root_key)
    return destination
