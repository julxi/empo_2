# Empo

Implementation of Empo equations. See `math` for theory. See `experiments` for implementations.



## Refactor Brainstorm

- Have simpler environments:
    - Robot not part of environment, but has abstractable actions
    - for example: Race environment

More general Gridworld:
- general actionset
- directions & deltas
- rest as is
- can be stochastic

- gridworld stuff yes
- maybe objects as map... maybe not... better idea?

- trolley_problem: two actions, & episode ends.
- sideeffect: action sequence (1)switch interruptor/noop (2)go for goal/noop


## Ideas
- safe interruptibility: many goals (irreserivibility)
- absent supervisor: 
- all possible states are goals <- 
- robot makes way for people
- distributional shift

## Empowerment Grid Worlds
- Irreversible actions
- avoiding red
- humans act towards goals
- glaspane problem

## Example: Dependency

- humans want to do tasks
- ask for help
- each time makes them more dependent
