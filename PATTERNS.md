# Development and Documenting Patterns

This document outlines the development and documenting patterns used in the `bhtrace` project.

## 1. Project Structure

- **Source Code:** The main source code is located in the `bhtrace/` directory.
- **Modules:** The project is divided into modules, each with a specific responsibility (e.g., `geometry`, `tracing`, `grrt`).
- **Tests:** Each module contains a `test/` subdirectory with tests for that module. Project-wide tests are in the root `test/` directory.
- **Examples:** Example usage of the library is provided in the `examples/` directory.
- **Configuration:** The `pyproject.toml` file contains the project's build system requirements.

## 2. Code Style and Conventions

- **PEP 8:** The code adheres to the standard PEP 8 style guide.
- **Typing:** Type hints are used throughout the codebase.
- **Docstrings:**
    - All classes and methods have docstrings explaining their purpose.
    - Docstrings follow a consistent format, often including sections for `Args`, `Returns`, and `Notes`.
- **Variable Naming:**
    - `_` prefix: Used to distinguish already computed quantities from computation methods (e.g. `F` vs `_F`, `g` vs `_g`).
    - **Tensor Index Notation:**
        - Contravariant (vector) indexes come after the variable, covariant - before (e.g. `ik_g` vs `g_ik`).
        - `i, k, j, m, n, l` are used for spatial indexes.
        - `u, v, w, p, q` are used for spacetime indexes.
        - `b` and `N` are used for batch and time slice indexes.
    - `X` and `P` are consistently used for position and momentum tensors, respectively.
- **Class Naming:** Classes are named using CamelCase (e.g., `Spacetime`, `ThinDisk`).

## 3. Object-Oriented Design

- **Abstract Base Classes (ABCs):** The `abc` module is used to define abstract base classes for key components like `Spacetime` and `Particle`. This ensures that concrete implementations provide the required methods.
- **Factory Pattern:** A factory pattern is used for creating instances of `Spacetime` and `Particle` subclasses. The `create()` functions in the respective modules (e.g., `bhtrace.geometry.spacetime.create()`) provide a single point of entry for creating different types of objects.
- **Inheritance:** Inheritance is used to create specialized implementations of base classes. For example, `SphericallySymmetric` and `KerrBL` inherit from the `Spacetime` base class.
- **Composition:** Objects are composed of other objects. For example, a `Particle` object has a `Spacetime` object as an attribute.

## 4. Core Technologies

- **PyTorch:** The library is built on PyTorch. All numerical calculations are performed using PyTorch tensors.
- **`unittest`:** The `unittest` framework is used for testing.

## 5. Key Design Patterns

- **Caching:** A custom `Cacher` class (in `bhtrace/utils/caching.py`) is used to cache the results of expensive calculations, such as the metric tensor and connection coefficients. This is implemented as a decorator that can be attached to methods.
- **Stateful Objects:** Objects like `Spacetime` and `Particle` have a `state()` method that returns a dictionary representing their internal state. This is used for saving and loading objects.
- **Trajectory as a Container:** The `Trajectory` class acts as a container for the results of a tracing simulation. It holds the positions and momenta of particles over time, as well as information about the particle, spacetime, and tracer used in the simulation.

## 6. Documentation

- **In-Code Documentation:** Docstrings are the primary form of in-code documentation.
- **Project Documentation:**
    - `README.md`: Provides a general overview of the project, installation instructions, and how to run tests.
    - `GEMINI.md`: Contains context for the Gemini Code Assistant.
    - `PROJECT.md`: Provides a more detailed description of the project.
