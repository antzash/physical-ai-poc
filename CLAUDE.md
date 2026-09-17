# Physical AI — Evidence Room POC

## Project Context
This is a Physical AI startup POC. The company builds sim-to-real robotic manipulation pipelines for high-stakes environments. The first use case is police evidence intake and storage — replacing the officer at the evidence room counter who manually logs incoming evidence and stores it in cabinets.

The core thesis:
- Train robotic policies in simulation using domain randomisation and reinforcement learning
- Close the sim-to-real gap through iterative feedback between simulation and real deployment
- The training pipeline itself is the IP, not the hardware

## Developer Setup
- MacBook Air M2 (Apple Silicon) — no NVIDIA GPU
- Python 3.13
- MuJoCo 3.x installed via pip3
- No physical robotic arm — simulation only for now

## What We Are Building (Phase 0 POC)
A MuJoCo simulation demonstrating:
1. Evidence room scene — intake counter + cabinet with addressable slots
2. Robotic arm picking objects from the counter
3. Domain randomisation — object size, mass, friction, position vary each episode
4. Objects placed into correct cabinet slots
5. Event logger simulating a chain-of-custody register

## Object Classes for POC
- Rigid box (phone box)
- Sealed bag (evidence bag)
- Cylinder (bottle)
- Flat folder/document

## Repo Structure
physical-ai-poc/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── models/
│   └── evidence_room.xml
└── scripts/
    ├── simulation.py
    ├── randomise.py
    └── logger.py

## Build Milestones
1. Static scene — arm + counter + cabinet loads in viewer ← START HERE
2. Scripted pick-and-place — hard-coded grasp, single object
3. Logger — custody events fire on pick and place
4. Domain randomisation — object properties vary per episode
5. Learned policy — RL training loop
6. Full integration — policy + logger + viewer end to end

## Key Constraints
- Must run on Apple Silicon M2, no GPU needed for Phase 0
- MuJoCo only, no Isaac Sim
- POC covers 4 object classes only — do not overclaim

## First Task
Build Milestone 1: working MuJoCo scene with arm, counter, cabinet and 4 slots. Launch interactive viewer. File: scripts/simulation.py + models/evidence_room.xml
