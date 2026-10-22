# MATH220 – Mathematical Modelling

Coursework for **MATH220 Mathematical Modelling** at the Norwegian University of Life Sciences (NMBU). The course page is [here](https://www.nmbu.no/emne/math220).

## About the course

MATH220 (10 ECTS) introduces different types of mathematical models and how they are derived, used and interpreted. The course covers three main themes:

- **Ordinary differential equations (ODEs):** modelling phenomena such as population dynamics and chemical reactions, equilibria and stability, linearization, and numerical solution methods.
- **Graph and network models:** using graphs to model relationships and interactions in e.g. transport networks, the internet and electrical circuits.
- **Stochastic models:** introducing uncertainty through random variables, with Markov chains, Poisson point processes and Kalman filters as examples.

Assessment is a modelling project (20 %) and a written exam (80 %). The project presentation must be approved to pass the course.

## Repository contents

```
MATH220/
├── assignments/   # Exercises and solutions from the course
└── project/       # Modelling project: Extended Kalman Filter for drone localization
```

## Project: Extended Kalman Filter for drone localization

The project applies the stochastic modelling part of the course to a practical problem: estimating the 3D position of a **DJI Tello** drone using an **Extended Kalman Filter (EKF)**.

- **State:** the drone's position `[x, y, z]`, with velocity commands entering as the control input.
- **Vision measurements:** ArUco markers with known world positions are detected with OpenCV. Range and bearing to each marker give a nonlinear measurement model, linearized with a Jacobian in the update step.
- **Height measurements:** the drone's built-in time-of-flight (ToF) sensor gives a direct height measurement.
- **Fusion:** the EKF combines the prediction from the control input with both sensor types, weighting each by its uncertainty.

The code is written in Python using NumPy and OpenCV, with the marker positions stored in a YAML map file.
