t(0.5):: a.
t(0.5):: b :- a.

%t(_)::stress(X).
%t(_)::happy(X).
%
%t(_)::\+happy(X) :- stress(X).
%t(_)::\+stress(X) :- happy(X).

