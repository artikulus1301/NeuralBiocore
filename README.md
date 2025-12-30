# Unified Consciousness System (Neurobiocore)

A comprehensive mathematical model of consciousness that integrates neurobiological dynamics, phenomenological experience (qualia), and social cognition (Theory of Mind). This system implements the **Grand Unified Model** (referencing Section 24 & 20b, Eq. 1199) to simulate the emergence of consciousness in artificial agents.

## 🧠 Key Components

The system is composed of three primary layers:

### 1. Neural Core (`Neuralbiocore_U` / `Neuralbiocore_U_for_GPU`)
Simulates the biological substrate of consciousness:
- **Neural Dynamics**: Spiking neural networks with realistic neurotransmitter modulation (GABA, Dopamine, Serotonin, etc.).
- **Topology**: Small-world network architecture for efficient information processing.
- **Plasticity**: Synaptic changes and learning mechanisms.

### 2. Phenomenological Layer (`u_qualia`)
Computational modeling of subjective experience:
- **Qualia Metrics**: Computes "Richness", "Coherence", and "Self-Recognition".
- **Integrated Information (Φ)**: Estimates the level of consciousness using GWT (Global Workspace Theory) and PCI-like metrics.

### 3. Social Cognition (`theory_of_mind`)
Implements high-level cognitive functions:
- **Theory of Mind (ToM)**: Agents maintain models of themselves (M0), others (M1), and meta-representations (M2).
- **Social Interaction**: Simulates social pain, empathy, and multi-agent dynamics.

## 🚀 Installation

Ensure you have Python installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

*Note: A GPU (CUDA) is recommended for optimal performance, as the system utilizes `torch` for tensor computations.*

## 🎮 Usage

The main entry point is `launcher_main.py`. You can run it directly for a quick demo or specify scenarios.

### Quick Start
Run the default demonstration (Consciousness Loss scenario):

```bash
python launcher_main.py
```

### Running Specific Scenarios
Use the `--scenario` flag to choose an experiment:

```bash
python launcher_main.py --scenario [SCENARIO_NAME]
```

**Available Scenarios:**

| Scenario Name | Description |
| :--- | :--- |
| `consciousness_loss` | **(Default)** Simulates gradual loss of consciousness under Propofol anesthesia. Tracks Φ, ToM, and emotions. |
| `social_interaction` | Simulates interaction between multiple agents (default 2). Includes "Social Rejection" event. |
| `dream_state` | Cycles through Wake, SWS (Slow Wave Sleep), and REM (Dreaming) stages. |
| `qualia_phenomenology` | Studies how emotional valence (Joy/Pain) modulates qualia richness and structure. |
| `pci_measurement` | Measures the Perturbational Complexity Index (PCI) under Wake, Light Anesthesia, and Deep Anesthesia. |

### Command Line Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--dt` | Time step in seconds | 0.05 |
| `--steps` | Total simulation steps | 600 |
| `--n_agents` | Number of agents (for social scenarios) | 2 |
| `--small_world` | Enable small-world network topology | False |
| `--disable_qualia` | Disable the Qualia computation layer | False |
| `--disable_social` | Disable the Social Cognition layer | False |

**Example:**
Run a social simulation with 3 agents for 1000 steps:
```bash
python launcher_main.py --scenario social_interaction --n_agents 3 --steps 1000
```

## 📊 Visualizations

The system automatically generates plots after each simulation run, visualizing:
- **Consciousness (Φ)** vs. Propofol levels.
- **Qualia Metrics**: Richness, Coherence, Self-Recognition.
- **Affective State**: Valence, Arousal, Social Pain.
- **Physiology**: ATP (Energy) levels.
- **Sense of Agency (SoA)**.
- **Social Intentions**: Self vs. Other models.
- **Sleep Stages**: Wake, SWS, REM.

## 📂 Project Structure

- `launcher_main.py`: Main entry point and scenario coordinator.
- `Neuralbiocore_U_for_GPU.py`: Core neural simulation logic (GPU-accelerated).
- `u_qualia.py`: computation of qualitative states.
- `theory_of_mind.py`: Implementation of social cognition and agent interaction.
- `requirements.txt`: Python package dependencies.

---
*Based on the Mathematical Model of Consciousness (Sections 20b, 24, 25a).*
