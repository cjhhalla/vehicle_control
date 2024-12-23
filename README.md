# Project README

## 📂 File Structure
The main files and directories in this project and their roles are as follows:

- **`run_setting.py`**:  
  Initializes the local frame and configures ROS topic processing.

- **`run_control.py`**:  
  Generates input control signals.

- **`sim_data.bag`**:  
  Simulation data file for running test codes.

- **`visualize/vis.py`**:  
  Visualizes text and point data.

- **`rviz/local_frame.rviz`**:  
  RViz configuration file for the local vehicle frame.

- **`rviz/visual.rviz`**:  
  RViz configuration file for the global vehicle frame.

---

## 🚀 Getting Started

### 1. **Initial Setup**
Run the following command to initialize the required settings:
```bash
./can.sh
This command sets up the connection between NovAtel GNSS and CAN.

### 2. **Running the Code**
Follow these steps to execute the project
