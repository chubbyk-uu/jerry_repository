# jerry_repository

This repository contains a Python script `three_body_sim.py` that animates a Newtonian three-body system. The program uses only the standard `tkinter` module and evolves a second, slightly perturbed copy to show how tiny differences grow over time, illustrating chaotic motion.

## Running the simulation

```bash
python3 three_body_sim.py
```

A window will open displaying both systems. Solid circles represent the original bodies and outlined circles show the perturbed ones. A label indicates the current step and the divergence between the two systems.
