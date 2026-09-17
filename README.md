# Physical AI — Evidence Room POC

A proof-of-concept for sim-to-real robotic manipulation applied to police evidence intake and storage.

## Overview

This project explores whether a robotic arm, trained entirely in simulation, can take over the manual work of an evidence room counter: receiving an item, logging it, and placing it into the correct storage slot. The goal is to demonstrate an end-to-end sim-to-real pipeline — not to build a production system.

The underlying approach:

- **Simulate first.** Robotic manipulation policies are trained in a MuJoCo simulation using domain randomisation (varying object size, mass, friction, and position) so the resulting policy generalises rather than overfitting to one scene.
- **Close the loop.** Sim-to-real transfer is treated as an iterative process — feedback from real-world deployment feeds back into simulation and training.
- **The pipeline is the product.** The value here is the training and transfer methodology, not any particular robot or piece of hardware.

## Use Case

Evidence rooms currently rely on an officer to manually log incoming evidence and store it in cabinets. This POC investigates automating that intake process for a constrained set of object types, with a full chain-of-custody event log generated automatically as items are handled.

## Object Classes (Phase 0)

- Rigid box (e.g. phone box)
- Sealed bag (evidence bag)
- Cylinder (e.g. bottle)
- Flat folder / document

## Status

Phase 0 proof-of-concept, in progress. Current milestone: a static MuJoCo scene with a robotic arm, intake counter, and a cabinet with addressable slots.

## Requirements

- Apple Silicon Mac (developed on M2, no GPU required)
- Python 3.13
- MuJoCo 3.x

## Setup

```bash
pip3 install -r requirements.txt
```

## Scope Note

This is an early-stage proof-of-concept covering four object classes in simulation only. It does not yet include a physical robot, learned policies, or real-world deployment.
