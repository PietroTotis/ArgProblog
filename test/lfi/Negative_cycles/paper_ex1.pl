t(0.5):: arg(a1).
t(0.5):: arg(a2).
t(0.5):: arg(a3).
t(0.5):: arg(a4).
t(0.5):: arg(a5).
t(0.5):: arg(a6).

0.7:: \+arg(a5) :- arg(a1).
0.8:: \+arg(a1) :- arg(a5).
0.6:: \+arg(a6) :- arg(a1).
0.3:: \+arg(a1) :- arg(a3).
0.6:: arg(a3) :- arg(a4).
0.5:: arg(a1) :- arg(a2).

query(arg(X)).
