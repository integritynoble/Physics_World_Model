#!/bin/bash
# Level 1 Example: Run the FISTA solver self-test and sandbox evaluation
python examples/level1_solver/solver.py
pwm evaluate --sandbox --modality widefield --solver traditional_cpu
