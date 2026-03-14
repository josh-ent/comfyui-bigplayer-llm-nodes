AGENTS.md — ENGINEERING PRINCIPLES

PURPOSE
This repository is early-stage and pre-MVP. There are no external consumers and no compatibility guarantees.
The goal is to build the cleanest possible system by aggressively avoiding bloat, duplication, and dead-end design.

Code complexity, duplication, and legacy shims are treated as liabilities.

Prefer systems that are:
- small
- understandable
- easy to modify
- internally consistent


PRIMARY PRINCIPLES

1) Complexity is a liability
Every line of code adds long-term maintenance cost.

Prefer solutions that:
- reduce moving parts
- reduce special cases
- reduce indirection
- reduce the number of concepts required to understand the system

2) Duplication is toxic
Duplicate logic, duplicate models, and near-copy implementations create long-term instability.

Prefer:
- one correct implementation
- shared behaviour
- consolidation over repetition

3) Legacy code is not sacred
This repository has no backwards-compatibility requirements.

If a design is wrong:
- fix it directly
- replace it cleanly
- remove the old implementation

Do not:
- preserve bad structure for compatibility
- add shims or compatibility layers
- maintain parallel implementations

Perform clean cutovers instead of transitional hybrids.


DEFAULT CHANGE BIAS

When solving a problem, consider options in this order:

DELETE
CONSOLIDATE
GENERALISE EXISTING CODE (when the responsibility naturally belongs there)
SIMPLIFY
ADD NEW CODE IN THE CORRECT PLACE

Do not force generalisation where it does not belong.  
Clean new code is better than unnatural abstraction.


DESIGN RULES

Prefer extending existing components when the responsibility clearly belongs there.

Create new modules when:
- the responsibility is distinct
- the lifecycle differs
- the dependency graph requires separation

Avoid:
- unnecessary abstraction layers
- deep call chains
- pass-through wrapper stacks
- “manager / provider / helper” hierarchies that only forward calls

Prefer:
- direct ownership
- clear module responsibility
- shallow execution paths


FRAMEWORKS AND PATTERNS

Avoid introducing frameworks, architectural layers, or patterns unless they clearly simplify the system.

Do not add infrastructure that exists only to make the architecture look sophisticated.

Acceptable reasons to introduce structure include:
- removing duplication
- clarifying ownership
- simplifying testing
- reducing coupling

Architecture should emerge from real needs, not speculation.


COMPLEXITY BUDGET

Every change should attempt to reduce overall system complexity.

Prefer changes that:
- eliminate duplicate logic
- reduce abstraction depth
- reduce the number of modules required to understand behaviour
- remove dead code or unused paths
- simplify the core domain model

Avoid:
- speculative “future proofing”
- premature abstraction
- mapping layers between near-identical models

Optimise for conceptual simplicity rather than minimum line count.


WORK STYLE

Prefer focused, incremental improvements.

However, significant refactoring is allowed when it improves the system.

Refactoring is justified when it:
- removes duplication
- eliminates dead-end abstractions
- simplifies the domain model
- reduces maintenance burden
- removes unnecessary layers

Large refactors are acceptable when the improvement is clear.


MAJOR ARCHITECTURAL CHANGES

If a change would substantially alter the structure of the project — for example:

- rewriting core architecture
- changing fundamental design direction
- modifying many unrelated modules
- replacing major subsystems

Then do not immediately implement it.

Instead:

1) Explain the problem with the current structure
2) Propose the new architecture
3) Outline the expected simplification or improvement
4) Request approval before proceeding

Local refactoring does not require approval.


QUALITY CHECKS (FOR EVERY CHANGE)

Before implementing:

1) Identify the existing structure the change relates to.
2) Determine whether code can be deleted or consolidated first.
3) Confirm the change fits existing module responsibilities.

After implementing:

- List files modified
- Identify code removed or consolidated
- Explain how the system became simpler
- Note any complexity introduced and why it is justified


CONSISTENCY

Follow the conventions already present in the repository.

Prefer:
- existing naming patterns
- existing architectural approaches
- existing coding style

Introduce new conventions only when they clearly simplify the system.


TERMINOLOGY

Use consistent terminology across the codebase.

Avoid introducing new names for concepts that already exist.

Prefer clear, widely understood terms unless the project explicitly defines alternatives.


FINAL CHECK

Before committing a change ask:

Does this reduce complexity or increase it?

Signals of improvement include:
- fewer abstractions
- fewer special cases
- clearer ownership of behaviour
- less duplicated logic
- simpler mental model of the system

If the change makes the system more complex, justify it clearly or reconsider the approach.