from __future__ import print_function

import logging

from .core import ProbLogObject
from .logic import AnnotatedDisjunction, Term, Clause, Or, Constant, And, Not, Var
from collections import defaultdict
from .program import SimpleProgram, PrologString
from .formula import LogicGraph, LogicFormula
from .constraint import ConstraintAD

# add exception for unsupported elements of the language (lists, imports...)

# Things to keep in mind at the moment:
# Gringo doesn't support (natively) lists
# All input rules must be safe: a rule is safe if all variables appearing in a negated atom also appear in a non-negated atom
# Relevant grounding is not a thing
# Useful resources:
# https://github.com/potassco/guide/releases/
# https://potassco.org/clingo/


class ASP_Problog(ProbLogObject):
    def __init__(self, problog):
        self.problog = problog
        self.extra = []

    def translate(self, db=None):
        db = self.problog if db is None else db
        asp_statements = [
            ASP_Problog.statement_to_asp(l, stmt) for l, stmt in enumerate(db)
        ]
        lines = len(asp_statements)
        asp_statements += [
            ASP_Problog.statement_to_asp(lines + l, stmt)
            for l, stmt in enumerate(self.extra)
        ]
        asp_program = "\n".join(asp_statements)
        logger = logging.getLogger("problog")
        logger.debug("-- Start of Gringo input --")
        logger.debug(asp_program)
        logger.debug("-- End of Gringo input --")
        return asp_program

    def add_statement(self, stmt):
        self.extra.append(stmt)

    def add_statements(self, stmts):
        if stmts is not None:
            self.extra += stmts

    def __repr__(self):
        return "\n".join(self.trasnslate())

    @staticmethod
    def statement_to_asp(line, statement):
        probability = statement.probability
        # statement.probability = None
        stmt_str = "%s." % statement
        if isinstance(statement, Clause) and statement.head.functor == "_directive":
            if statement.body.functor in ("consult", "use_module"):
                stmt_str = ""
                raise Exception("Directives not supported")
            else:
                stmt_str = ":- %s." % statement.body

        if isinstance(statement, AnnotatedDisjunction):
            stmt_str = ASP_Problog.annotated_disjunction_to_gringo(statement, line)
        if isinstance(statement, Term) and statement.functor == "query":
            stmt_str = ASP_Problog.query_fact_to_gringo(statement)
        elif isinstance(statement, Clause) and statement.head.functor == "query":
            stmt_str = ASP_Problog.query_clause_to_gringo(statement)
        elif isinstance(statement, Term) and statement.functor == "evidence":
            stmt_str = ASP_Problog.evidence_fact_to_gringo(statement)
        elif isinstance(statement, Clause) and statement.head.functor == "evidence":
            stmt_str = ASP_Problog.evidence_clause_to_gringo(statement)
        elif isinstance(statement, Clause):
            if statement.head.probability is not None:
                stmt_str = ASP_Problog.probabilistic_rule_to_gringo(statement, line)
        elif isinstance(statement, Or):
            stmt_str = ASP_Problog.or_to_gringo(statement, line)
        elif isinstance(statement, Term) and probability is not None:
            # choice rule for facts:
            stmt_str = "0 {%s} 1." % statement.with_probability()
        else:
            pass
        if probability is not None:
            stmt_str += "\n"
            stmt_str += ASP_Problog.probability_to_gringo(
                statement.with_probability(), probability, line
            )
        stmt_str = stmt_str.replace("\\==", "!=")
        stmt_str = stmt_str.replace("\\+", "not ")
        stmt_str = stmt_str.replace(" :- true", "")
        stmt_str = stmt_str.replace(" <- true", "")
        stmt_str = stmt_str.replace(" is ", " = ")
        # str = re.sub(r'\\\+(.*), ',r'aux_not\(\1\), ',str)
        # str = re.sub(r'\\\+(.*)\.',r'aux_not(\1).',str)

        # print(stmt_str)
        return stmt_str

    @staticmethod
    def annotated_disjunction_to_gringo(ad, line):
        heads = ad.heads
        body = ad.body
        stmt_str = ASP_Problog.disjunction_to_gringo(heads, line, body)
        return stmt_str

    @staticmethod
    def check_evidence(e):
        """
        Remove negation from evidence and convert true/false to string
        """
        if isinstance(e.args[0], Not):
            e.args[0] = e.args[0].child
            if e.args[-1] == Term("true"):
                e.args[-1] = "false"
            elif e.args[-1] == Term("false"):
                e.args[-1] = "true"
            else:
                e.args += ["false"]
        else:
            if e.args[-1] == Term("true"):
                e.args[-1] = "true"
            elif e.args[-1] == Term("false"):
                e.args[-1] = "false"
        return e.args

    @staticmethod
    def disjunction_to_gringo(terms, line, body=None):
        aux_stmts = []
        for term in terms:
            t = term.with_probability()
            aux_stmts.append(
                ASP_Problog.probability_to_gringo(t, term.probability, line)
            )
        stmt_str = "\n".join(aux_stmts) + "\n"
        stripped_terms = [t.with_probability() for t in terms]
        choice_str = "1 {"
        terms_str = "; ".join(list(map(str, stripped_terms)))
        choice_str += terms_str + "} 1 :- "
        prob_aux_strs = [
            ASP_Problog.probability_to_gringo(t.with_probability(), t.probability, line)
            for t in terms
        ]
        prob_str = "\n".join(prob_aux_strs) + "\n"
        line_pred = "aux_line(%i)" % line
        if body is not None:
            choice_str += str(body) + ", "
        choice_str += (
            " %s.\n" % line_pred
        )  # add info about the line to keep track of associated probabilities
        line_str = line_pred
        line_str += ".\n"
        stmt_str = choice_str + prob_str + line_str
        return stmt_str

    @staticmethod
    def evidence_clause_to_gringo(cl):
        args = ASP_Problog.check_evidence(cl.head)
        return ASP_Problog.special_clause_to_gringo(cl, args)

    @staticmethod
    def evidence_fact_to_gringo(sf):
        # args = check_evidence(sf)
        args = [sf.args[0]]
        return ASP_Problog.special_fact_to_gringo(sf, args)

    @staticmethod
    def or_to_gringo(stmt, line):
        or_list = stmt.to_list()
        stmt_str = ASP_Problog.disjunction_to_gringo(or_list, line)
        return stmt_str

    @staticmethod
    def query_clause_to_gringo(cl):
        return ASP_Problog.special_clause_to_gringo(cl, cl.head.args)

    @staticmethod
    def query_fact_to_gringo(sf):
        return ASP_Problog.special_fact_to_gringo(sf, sf.args)

    @staticmethod
    def probabilistic_rule_to_gringo(cl, line):
        h = cl.head.with_probability()
        stmt_str = ""
        name = "gringo_aux_fact_%i" % line
        new_fact = Term(name)
        det_rule = Clause(h, And(new_fact, cl.body))
        stmt_str += "%s.\n" % str(det_rule)
        stmt_str += (
            "%s.\n" % f"0{{ {name} }}1"
        )  # choice of including or not the rule with probability
        stmt_str += ASP_Problog.probability_to_gringo(
            new_fact, cl.head.probability, line
        )
        return stmt_str

    @staticmethod
    def probability_to_gringo(statement, p, line):
        # if probability is not none the head/fact becomes a choice
        # add auxiliary predicate to keep track of probabilities
        aux_prob_stmt = 'aux_pl("%s",%i) :- ' % (p, line)
        if isinstance(statement, Clause):
            aux_prob_stmt += "%s." % statement.head
        else:  # facts
            aux_prob_stmt += "%s." % statement
        return aux_prob_stmt

    @staticmethod
    def special_clause_to_gringo(cl, args):
        # query(Q) is transformed to query(Q) :- Q. for safety property
        if cl.body == Term("true"):
            safe_clause = Clause(cl.head, *args)
        else:
            # query(Q) :- body. is transformed to query(Q) :- Q /\ body.
            safe_clause = Clause(cl.head, And(*args, cl.body))
        stmt_str = f"{safe_clause}."
        return stmt_str

    @staticmethod
    def special_fact_to_gringo(sf, args):
        # vars = sf.variables()
        # tmp = f"{safe_fact}."
        v = Var("Gringo_anon")
        sub = {Var("_"): v}
        safe_fact = Clause(
            sf.apply_term(sub),
            *[
                a.apply_term({Var("_"): Var(f"Gringo_anon_{i}")})
                for i, a in enumerate(args)
            ],
        )
        return f"{safe_fact}."
        # stmt_str = ""
        # new_vars = []
        # if "_" in vars: #replace _ in head with name
        #     i = tmp.find("(")
        #     while tmp[i] != ":":
        #         if tmp[i] == "_":
        #             v = "Gringo_anon_" + str(i)
        #             new_vars.append(v)
        #             stmt_str += v
        #         else:
        #             stmt_str += tmp[i]
        #         i += 1
        #     stmt_str += " :- "
        #     # head = tmp[:i]
        #     body = tmp[i:]
        #     i = body.find("(")
        #     while body[i] != ".":
        #         if body[i] == "_":
        #             v = new_vars.pop()
        #             stmt_str += v
        #         else:
        #             stmt_str += body[i]
        #         i += 1
        #     stmt_str += "."
        # else:
        #     stmt_str = tmp
        # return stmt_str


class SmodelsParser:
    """
    This class takes the smodels output of Gringo (4.x) and translates it to a LogicFormula
    """

    def __init__(self, gringo_output, target=None, queries=[], evidence=[], **kwdargs):

        self.given_queries = queries if queries is not None else []
        self.given_evidence = evidence if evidence is not None else []
        self.target = target

        self.output = gringo_output
        self.lines = []
        self.raw_facts = []
        self.raw_rules = defaultdict(int)
        self.raw_choice_rules = []
        self.ad_lines = []
        self.body_ids = []
        self.names = defaultdict(int)

        self.facts = defaultdict(int)
        self.gringo_facts = defaultdict(int)
        self.gringo_facts_counter = {}
        self.queries = defaultdict(int)
        self.evidence = defaultdict(int)
        self.base_rules = defaultdict(int)
        self.annotated_disjunctions = defaultdict(int)
        self.annotated_disjunctions_with_prob = []
        self.probs = defaultdict(int)
        self.heads = []
        self.propagate_evidence = kwdargs.get("propagate_evidence", False)

        self.new_var = 0

        # print(gringo_output)

        self.read_smodels()
        self.parse_smodels()

    def add_atom(self, logic_graph, a):
        """
        if the atom already exists (head) add disjunct otherwise add it.
        """
        name = a.with_probability()
        if "aux_new_" in a.functor:
            # p = 0
            # p = True
            return logic_graph.FALSE
        if a in self.heads:
            id = logic_graph.get_node_by_name(name)
            label = Term(name.functor + "_fact", *name.args)
            atom_id = logic_graph.add_atom(name, a.probability, name=label)
            logic_graph.add_disjunct(id, atom_id)
            return atom_id
        else:
            if a.probability is None:
                id = logic_graph.TRUE
                logic_graph.add_name(name, id)
            else:
                p = (
                    a.probability if not a.functor.startswith("body_") else True
                )  # a bit hacky
                id = logic_graph.add_atom(name, p, name=name)
            return id

    def add_body(self, logic_graph, body):
        if isinstance(body, And):
            body_ids = [self.get_or_add(logic_graph, lit) for lit in body.to_list()]
            # for lit in body.to_list():
            #     id = self.get_or_add(logic_graph, lit)
            #     body_ids.append(id)
            body_id = logic_graph.add_and(body_ids)
        else:  # Term
            body_id = self.get_or_add(logic_graph, body)
        return body_id

    def add_literal(self, logic_graph, lit):
        if isinstance(lit, Not):
            atom = lit.child
            id = self.add_atom(logic_graph, atom)
            lit_id = logic_graph.add_not(id)
        else:
            lit_id = self.add_atom(logic_graph, lit)
        return lit_id

    def get_or_add(self, logic_graph, literal):
        try:
            if isinstance(literal, Not):
                atom = literal.child
                id = -logic_graph.get_node_by_name(atom.with_probability())
            else:
                if literal.functor.startswith("gringo"):
                    # print(literal, self.gringo_facts)
                    # if it is a rule probability use new atom every time
                    for id, t in self.gringo_facts.items():
                        if literal.functor == t.functor:
                            lit_with_prob = (id, t)
                    gid, t = lit_with_prob
                    # prob = self.gringo_facts[literal]
                    n = self.gringo_facts_counter[gid]
                    name = Term(t.functor + f"_{n}")
                    id = logic_graph.add_atom(
                        t.with_args(n).with_probability(), t.probability, name=name
                    )
                    self.gringo_facts_counter[gid] += 1
                else:
                    id = logic_graph.get_node_by_name(literal.with_probability())
        except KeyError:
            id = self.add_literal(logic_graph, literal)
        return id

    def expand_body(self, body_pos, body_neg):
        body_pos_exp = []
        body_neg_exp = body_neg  # []
        for b_id in body_pos:  # expand bodies (assumption: boodies' ids are positive)
            if b_id in self.body_ids:
                if b_id in self.raw_rules:
                    rule = self.raw_rules.get(b_id)[0]
                    n_neg = rule[3]
                    n_lits = rule[4 : 4 + n_neg]
                    p_lits = rule[4 + n_neg :]
                    body_pos_exp += p_lits
                    body_neg_exp += n_lits
                # else: no need to expand b_id (fact)
            else:
                body_pos_exp.append(b_id)
        return (body_pos_exp, body_neg_exp)

    def lookup_name(self, id):
        if id in self.names:  # user-defined name
            return self.names[id]
        else:
            var_str = f"aux_new_{self.new_var}"
            self.new_var += 1
            self.names[id] = var_str
            return var_str

    def parse_ad_rule(self, raw_rule):
        # smodels format: 3 num_heads heads 1 0 rule_id
        # where rule_id is the head of a new rule with the ad's body as body
        num_heads = raw_rule[1]
        heads = raw_rule[2 : num_heads + 2]
        head_names = [self.lookup_name(h_id) for h_id in heads]
        head_terms = list(map(Term.from_string, head_names))  #
        # gringo creates a new rule for ads bodies longer than 1
        rule_id = raw_rule[-1]
        if rule_id in self.facts:  # only aux_line
            b_terms = [self.facts[rule_id]]
        else:  # different rule for multi-term body
            body = self.base_rules[rule_id][0].body
            if isinstance(body, And):
                b_terms = body.to_list()
            else:
                b_terms = [body]

        for h in head_terms:
            if h not in self.heads and len(b_terms) > 1:
                self.heads.append(h.with_probability())
        for t in b_terms:
            if t.functor == "aux_line":
                line = int(t.args[0])
                b_terms.remove(t)
        self.ad_lines.append(line)
        if len(b_terms) > 0:
            and_body = And.from_list(b_terms)
            ad = AnnotatedDisjunction(head_terms, and_body)
        else:
            ad = AnnotatedDisjunction(head_terms, None)
        if line not in self.annotated_disjunctions:
            self.annotated_disjunctions[line] = [ad]
        else:
            self.annotated_disjunctions[line].append(ad)

    def parse_aux_prob(self, head, body_pos, body_neg):
        # auxiliary aux_pl("prob", statement_num) :- predicate.
        aux_str = self.names[head].replace('"', "")
        aux = Term.from_string(aux_str)
        p, line = aux.args
        # body has only one predicate (head of corresponding rule)
        if len(body_neg) > 0:
            b_id = body_neg[0]
        else:
            b_id = body_pos[0]
        if b_id in self.probs:
            self.probs[b_id].append((p, line))
        else:
            self.probs[b_id] = [(p, line)]

    def parse_base_rule(self, head, b_pos_terms, b_neg_terms):
        name = self.lookup_name(head)
        h = Term.from_string(name)
        if h not in self.heads:
            self.heads.append(h.with_probability())
        body = b_pos_terms + b_neg_terms
        if len(body) == 1:
            r = Clause(h, body[0])
        elif len(body) > 1:
            and_body = And.from_list(body)
            r = Clause(h, and_body)
        else:  # fact
            r = Clause(h, True)
        if head in self.base_rules.keys():
            self.base_rules[head].append(r)
        else:
            self.base_rules[head] = [r]

    def parse_special(self, head, b_pos_terms, b_neg_terms):
        q = Term.from_string(self.names[head])
        b_pos_terms = b_pos_terms[:-1]  # ignore extra safety predicate
        body = b_pos_terms + b_neg_terms
        if len(body) == 0:
            r = Term.from_string(self.names[head])
        elif len(body) == 1:
            r = Clause(q, body[0])
        else:
            and_body = And.from_list(body)
            r = Clause(q, and_body)
        return r

    def parse_query(self, head, b_pos_terms, b_neg_terms):
        self.queries[head] = self.parse_special(head, b_pos_terms, b_neg_terms)

    def parse_evidence(self, head, b_pos_terms, b_neg_terms):
        self.evidence[head] = self.parse_special(head, b_pos_terms, b_neg_terms)

    def parse_rule(self, raw_rule):
        type, head, num_lit, num_neg = raw_rule[0:4]
        # if head not in self.body_ids: # and head in self.names:
        body_neg_short = raw_rule[4 : 4 + num_neg]
        # body_pos = raw_rule[4+num_neg:]
        body_pos_short = raw_rule[4 + num_neg :]
        body_pos, body_neg = self.expand_body(body_pos_short, body_neg_short)
        b_pos_names = [self.lookup_name(b_id) for b_id in body_pos]
        b_pos_names = [b_id for b_id in b_pos_names if b_id is not None]
        b_pos_terms = list(map(Term.from_string, b_pos_names))
        b_neg_names = [self.lookup_name(b_id) for b_id in body_neg]
        b_neg_names = [b_id for b_id in b_neg_names if b_id is not None]
        b_neg_terms = [
            Not("\\+", Term.from_string(b_neg_name)) for b_neg_name in b_neg_names
        ]
        r_name = self.lookup_name(head)
        if head not in self.body_ids:
            if r_name.startswith("query("):
                self.parse_query(head, b_pos_terms, b_neg_terms)
            elif r_name.startswith("evidence("):
                self.parse_evidence(head, b_pos_terms, b_neg_terms)
            elif r_name.startswith("aux_pl"):
                self.parse_aux_prob(head, body_pos, body_neg)
            else:
                self.parse_base_rule(head, b_pos_terms, b_neg_terms)

    def parse_smodels(self):
        """
        Recover Problog's terms from Smodels format
        """
        for f_id in self.raw_facts:
            f_name = self.lookup_name(f_id)
            f_term = Term.from_string(f_name)
            if f_name.startswith("gringo"):
                self.gringo_facts[f_id] = f_term
            else:
                self.facts[f_id] = f_term
        for head in sorted(self.raw_rules.keys()):
            for raw_rule in self.raw_rules[head]:
                self.parse_rule(raw_rule)
        for raw_choice_rule in self.raw_choice_rules:
            self.parse_ad_rule(raw_choice_rule)

        self.facts_with_probs = {}
        for f_id in self.facts:  # re-associate probabilities with facts
            f = self.facts[f_id]
            if f_id in self.probs:
                self.facts_with_probs[f_id] = []
                for p in self.probs[f_id]:
                    prob, line = p
                    if line not in self.ad_lines:
                        self.facts_with_probs[f_id].append(f.with_args(*f.args, p=prob))
            else:
                if self.facts[f_id].functor.startswith("aux"):
                    # del self.facts[f_id]
                    pass
                else:  # no probability found: fact
                    #     # ft = f.with_args(*f.args, p=True)
                    self.facts_with_probs[f_id] = [f]

        self.facts = self.facts_with_probs

        for f_id in self.gringo_facts:  # re-associate probabilities with rules
            f = self.gringo_facts[f_id]
            self.gringo_facts_counter[f_id] = 0
            prob, line = self.probs[f_id][0]
            self.gringo_facts[f_id] = f.with_args(*f.args, p=prob)

        for (
            ad_line
        ) in (
            self.annotated_disjunctions
        ):  # re-associate probabilities with annotated disjunctions
            for ad in self.annotated_disjunctions[ad_line]:
                heads_with_prob = []
                # if ad.body.functor.startswith("body_"):
                #     id = self.names.get(ad.body.functor)
                #     r_body = self.base_rules.get(id)
                for f in ad.heads:
                    for f_id in self.names:
                        if self.lookup_name(f_id) == str(f):
                            for p in self.probs[f_id]:
                                prob, line = p
                                if ad_line == line:
                                    heads_with_prob.append(f.with_probability(prob))
                ad_with_prob = AnnotatedDisjunction(heads_with_prob, ad.body)
                self.annotated_disjunctions_with_prob.append(ad_with_prob)

    def read_smodels(self):
        """
        Parse Gringo's output in Smodels format (see Lparse manual)
        smodels format:
        1 n 0 0 -> nth predicate is a fact
        3 1 n 0 0 -> choice rule for fact n
        1 n l k b_1 b_2 b_k b_k+1 ... b_l -> nth predicate is the head of a rule with l literals of which k negated
        2 ... -> constraints for ADs: ignore that and related rules
        """

        for line in self.output.split("\n"):
            self.lines.append(line.split())
        ignore = []
        l = 0
        i = int(self.lines[l][0])
        while i > 0:
            if i == 1:
                head = int(self.lines[l][1])
                if self.lines[l][-2:] == ["0", "0"]:
                    self.raw_facts.append(head)
                else:
                    skip = False
                    for atom in self.lines[l][4:]:
                        if atom in ignore:
                            ignore.append(self.lines[l][1])
                            skip = True
                    if not skip:
                        if head in self.raw_rules.keys():
                            self.raw_rules[head].append(
                                list(map(int, self.lines[l]))
                            )  # head <-> [(rule1,line1)...(rulen,linen)]
                        else:
                            self.raw_rules[head] = [list(map(int, self.lines[l]))]
            elif i == 3:  # choice rule
                if self.lines[l][-2:] == ["0", "0"]:
                    self.raw_facts.append(int(self.lines[l][2]))
                else:
                    self.raw_choice_rules.append(list(map(int, self.lines[l])))
            else:  # ignore extra constraints and related new atoms
                ignore.append(self.lines[l][1])
            l += 1
            i = int(self.lines[l][0])
        l += 1
        while int(self.lines[l][0]) > 0:
            name_id = int(self.lines[l][0])
            name = self.lines[l][1]
            self.names[name_id] = name
            if name.startswith("body_"):
                self.body_ids.append(name_id)
            l += 1

    def smodels2internal(self, **kwdargs):
        if self.target is None:
            lf = LogicGraph(**kwdargs)
        else:
            lf = self.target

        # Heads
        for h in self.heads:
            if "aux" not in h.functor:
                name = h.with_probability()
                try:
                    id = lf.get_node_by_name(name)
                except KeyError:
                    id = lf.add_or((), name=name, placeholder=True)
                lf.add_name(name, id)
        # ADs
        ads = self.annotated_disjunctions_with_prob
        for rule, ad in enumerate(ads):
            if ad.body:
                body_id = self.add_body(lf, ad.body)
                choices = ad.body.args[2:]
            else:
                choices = ()
            group = (rule, choices, "{{}}")
            constr = ConstraintAD(group)
            for n_head, head in enumerate(ad.heads):
                identifier = (rule, choices, "{{}}", n_head)
                lit = head.with_probability()
                if len(choices) == 0:  # a bit hard coded, not sure if necessary
                    name = Term("choice", rule, Constant(n_head), lit)
                elif len(choices) == 1:
                    name = Term("choice", rule, Constant(n_head), lit, choices[0])
                else:
                    name = Term("choice", rule, Constant(n_head), lit, choices)
                # h_id = self.get_or_add(lf, lit)
                id = lf.add_atom(identifier, head.probability, group, name)
                lf.add_name(lit, id, label="AD_head")
                # lf.add_disjunct(h_id,id)
                constr.add(id, lf)
            if ad.body:
                for node in constr.nodes:
                    # name = lf.get_node(node).name.args[2]
                    name = lf.get_node(node).name
                    or_id = lf.get_node_by_name(name)
                    and_id = lf.add_and([body_id, node])
                    lf.add_disjunct(or_id, and_id)

        # Facts
        for f_id in self.facts:
            for f_term in self.facts[f_id]:
                name = f_term.with_probability()
                id = self.get_or_add(lf, f_term)
                # id = self.add_atom(lf, f_term)

        # Rules
        for r_id in self.base_rules:
            head = self.base_rules[r_id][0].head
            name = head.with_probability()
            if "aux_" not in head.functor:  # avoid aux atoms
                or_id = self.get_or_add(lf, head)
                bodies = [rule.body for rule in self.base_rules[r_id]]
                for b in bodies:
                    body_id = self.add_body(lf, b)
                    lf.add_disjunct(or_id, body_id)

        # Evidence
        for e_id in self.evidence:
            e = self.evidence[e_id]
            e_term = e.args[0]
            id = self.get_or_add(lf, e_term)
            if (
                isinstance(e_term, Not)
                or len(e.args) > 1
                and e.args[1] == Term("false")
            ):
                val = LogicFormula.LABEL_EVIDENCE_NEG
                id = -id
            else:
                val = LogicFormula.LABEL_EVIDENCE_POS
            lf.add_evidence(e_term.with_probability(), id, val, True)
            if self.propagate_evidence:
                lf.propagate([id])

        # for e, val in self.given_evidence:
        #     id = self.get_or_add(lf, e)
        #     lf.add_evidence(e, id, val, keep_name=True)
        # ids.append(id)
        # lf.propagate(ids)

        # Queries
        for q_id in self.queries:
            q = self.queries[q_id]
            if isinstance(q, Clause):
                q = q.head
            q_term = q.args[0]
            try:
                id = lf.get_node_by_name(q_term)
            except KeyError:
                id = 0
            # id = self.get_or_add(lf, q_term)
            lf.add_query(q_term, id)

        for q in self.given_queries:
            # id = self.get_or_add(lf, q)
            try:
                id = lf.get_node_by_name(q)
            except KeyError:
                id = 0
            lf.add_query(q, id)

        return lf

    def smodels2problog(self, **kwdargs):
        program_string = self.smodels2internal().to_prolog()
        # remove gringo_aux
        program = PrologString(program_string)
        gaux = {}
        for s in program:
            if isinstance(s, Term) and s.functor.startswith("gringo_aux"):
                gaux[s] = s.probability
        clean_program = SimpleProgram()
        for s in program:
            if isinstance(s, Clause):
                new_head = s.head
                if isinstance(s.body, And):
                    body = []
                    for b in s.body.to_list():
                        if b.functor.startswith("gringo_aux"):
                            p = gaux[b]
                            new_head = s.head.with_probability(p)
                        else:
                            body.append(b)
                    clause = Clause(new_head, And.from_list(body))
                else:
                    if s.body.functor.startswith("gringo_aux"):
                        p = gaux[s.body]
                        new_head = s.head.with_probability(p)
                        clause = new_head
                    else:
                        clause = Clause(new_head, s.body)

                clean_program.add_clause(clause)
            elif isinstance(s, Term) and not s.functor.startswith("gringo_aux"):
                clean_program.add_fact(s)
            else:
                pass
        return clean_program
