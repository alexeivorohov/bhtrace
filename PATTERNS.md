# Development and Documentation Patterns

This document outlines the architectural patterns, coding conventions, and documentation standards used in the `bhtrace` package.

## 1. Project Structure

**Source Code:** Located in the `bhtrace/` directory.\
**Modules:** The project is modularized by domain responsibility (e.g., `bhtrace.geometry`, `bhtrace.physics`, `bhtrace.grrt`).\
**Tests:** All test suites are located in the `test/` directory at the project root.\
**Examples:** Usage demonstrations and tutorials are provided in the `examples/` directory.
**Configuration:** Build requirements and metadata are managed via pyproject.toml.

## 2. Core Technologies & Dependencies

**PyTorch:** This ML framework implements efficient tensor operations and has wide support across various devices and accelerators. We leverage these features to make the computations under the hood of `bhtrace` package robust, versatile and cross-platform.\
**pytest:** Powerful framework for unit and integration testing. 

## 3. Key Design Patterns

### Abstract Base Classes (ABCs)

The `abc` module is used to define interfaces for core components such as `Spacetime` and `Particle`. This enforces a contract, ensuring that all concrete implementations provide the necessary physical methods.


### Inheritance & Composition

**Inheritance:** Used to implement specialized physics models (e.g., KerrBL inheriting from Spacetime).\
**Composition:** Objects are built via composition to maintain decoupling; for instance, a Particle object contains a reference to a Spacetime object.

### Registry Pattern

The `.utils.registry` module provides a centralized mechanism to access specific class implementations, instances, or callables. Registries, implemented in this module, have useful builtin features like signature check for callables on registration, registration decorators and many others.

### Facade Pattern

High-level interfaces such as `Trajectory`, `GRRTData`, and `LensingData` act as facades - they wrap simulation outputs and provide simplified methods for visualization and debugging, making the user-facing API more convenitent and straightforward.

### Lazy Data Transfer Objects

Subclasses of `PhysicsLocal` are used to minimize boilerplate and optimize computational overhead during local frame transformations. 

**Mechanism:** These classes use state inheritance and lazy initialization via `@cached_property`.

**Example**: `SpacetimeLocal` is used by `ParticleLocal` instance, created by `particle_instance.local(x)`. Most of `SpacetimeLocal` properties are wrapped with `@cached_property` decorator so they will be calculated only on need. `ParticleLocal` instance exposes it's `SpacetimeLocal` instance to the next enitities in a chain (e.g. `MediumLocal` or ray-tracing/grrt solvers), so there is no need in re-evaluation of any physical quantities of spacetime on this simulation step.

### Stratrgy  Pattern 

The Strategy pattern defines a family of interchangeable algorithms and encapsulates each one within a separate class. This allows to switch between different behaviors or computational methods at runtime without modifying the code that uses them.

We use this pattern to implement various graphic backens, numerical schemes and etc.

`PhysicsLocal` subclasses also implement this pattern. For example, there are implementations of `SpacetimeLocal`, specialized to certain, commonly used spacetimes. In these implementations intermediate quantities like $\Sigma$ in Kerr metric are defined in same "lazy" way as other properties and repeatedly used to calculate other quantities. Thus, when program operates with `PhysicsLocal` subclasses, the amount of calculations can be signinficantly reduced, especially for complex spacetimes like `EffectiveGeometry`.


### Single responsibility

Every class should encapsulate a single, well-defined responsibility (e.g., managing a physical entity), and every method should solve exactly one task. This modularity is essential for testability and prevents the creation of "God Objects" and overcomplicated functions.


## 4. Documentation Standards

### In-Code Documentation and Docstrings

All modules, classes, methods, and attributes must include docstrings. We follow the `numpydoc` standard to ensure compatibility with automated documentation generators like Sphinx.

### References and Implementation Details

**Scientific Rigor:** Classes and methods implementing physical laws or entities or numerical schemes must include a formal reference (e.g., DOI/ArXiv link, book link).\
**Preserving Context:** Original numerical schemes or physical expressions, proposed by authors or contributors should be documented format of `.md` files or `.ipynb` notebooks within the `docs/` folder to preserve derivation logic.

## 5. Naming Conventions

### Tensor Naming and Index Notation

To resolve ambiguity in high-rank or mixed-variance tensors, this package utilizes Signature Notation. Since Python lacks native support for subscripts and superscripts, we use a suffix string to represent the sequence of contravariant and covariant indices:
 - **u (up)**: Represents a contravariant index (upper index, e.g., $p^\mu \to $ `p_u`).
 - **d (down)**: Represents a covariant index (lower index, e.g., $p_\mu \to $ `p_d`)).

The order of the characters in the suffix corresponds exactly to the order of the indices in the tensor's mathematical definition.

### Exceptions and Special Cases
| Case | Rule | Example |
| :--- | :--- | :--- |
| **A. Essential (co)Vectors** | For fundamental (c0)vectors, indices may be dropped if context is clear. | $u^\mu \to$ `u`, $x^\mu \to$ `x` |
| **B. Metric Tensors** | The metric and its inverse are exempt from suffixes. | $g_{\mu\nu} \to$ `g`, $g^{\mu\nu} \to$ `ginv` |
| **C. Eliminating Repetitions** | If the rank of the tensor is evident from it's name and this representation has all-upper or all-down indexes, use `_up` or `_down` instead of `_uuuu` and `_ddd`. | $R^{\mu\nu} \to$ `Ricci_up` |
| **D. Derivatives** | Differentiation should be reflected in the variable/method name and derivative indexes come after tensor's own indexes | $F^{\mu\nu}_{,\xi} \to$ `dFdx_uud` |


### Einstein Sum as a Name

Not all quantities have common name or denotement. In this case their expression in einstein notation should be used with the following rules:
 - **Free indexes** should be noted by `signature` pattern described previously;
 - **Contracted indexes** should be noted by letters `w, q, v, s, p` for 4-dim axes and `i, j ,k, l` for 3-dim axes. 
 - **Comosition** In the name, notation for every next tensor in the product goes after underscore.
 - **Example**: $F^{\mu\nu} U_{\nu} \to$ `F_us_U_s`
  

### General Naming
*   **Classes:** `CamelCase` (e.g., `ThinDisk`).
*   **Variables & Methods:** `snake_case`. 
    * *Note:* Capital letters are permitted in variable names if they strictly follow standard physics literature notation.

