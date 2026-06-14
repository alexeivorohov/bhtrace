# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.1] - 2026-06-14

### Added
- New module `.utils.units`, responsible for unit systems & unit conversion
- New module `.physics`, responsible for physical models & objects (except gravity)
- New `XLocal` classes for lazy-initialization of physical object properties
- New data structures for handling simulation outputs and the `.data` module
- Draft of new `.exact` module, responsible for exact solutions to particle motion in different spacetimes
- New `.scenarios.makers` module, providing methods to instantly obtain `Trajectory` for common cases
- Minimal CI pipeline configured

### Changed
- Completely rewrote `electrodynamics` module and moved it to the `.physics` module
- Overhaul of `.graphics` module
- Improved `Trajectory` integration with `.graphics` module and moved it to `.data` module
- Rewrote `EffectiveGeometry` class in `.spacetime` module
- Overhaul of `.globs` module
- Improvements in `.utils.registry` module
- Improved `.grrt.runner` module
- Changed plain python usage examples to `.ipynb` notebooks

### Fixed
- Fixed most of the physics in `medium` and `radiation` modules
- Tracers now actually do integrate backwards in time (affects particle velocities and dynamic spacetimes)
- Fixed some expressions in `spacetime` implementations

### Deprecated 
- `.utils.caching` module is being deprecated in favor of using `XLocal` classes to avoid re-evaluation of quantities

### Removed
- Outdated tests

## [0.1.0] - 2026-02-20

A minimally functional state.