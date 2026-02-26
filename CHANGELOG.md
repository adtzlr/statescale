# Changelog
All notable changes to this project will be documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-02-26

### Changed
- Change default behavior for math-functions to be applied on point-, cell- and field-data to `ModelResult.apply(np.mean, on_point_data=False, on_cell_data=False, on_field_data=False)`.

### Removed
- Remove `ModelResult.mean()`, use `ModelResult.apply(np.mean)` instead.

## [0.2.2] - 2026-01-27

### Fixed
- Minor fixes for PyPI package release.

## [0.2.1] - 2026-01-27

### Fixed
- Minor fixes for PyPI package release.

## [0.2.0] - 2026-01-26

### Changed
- Change name to `statescale` (was `snapsy`).

## [0.1.1] - 2026-01-23

### Fixed
- Fix tests.
- Fix 1d- and 2d-input shapes, including one-dimensional snapshots and signals.

## [0.1.0] - 2026-01-22

### Added
- Start using a Changelog.