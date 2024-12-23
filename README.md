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
Run the following command to initialize the required settings: `./can.sh`


This command sets up the connection between NovAtel GNSS and CAN.

---

### 2. **Running the Code**
Follow these steps to execute the project:

#### **Connect CAN data to Morai Simulation**
Run the `can2morai.py` script to connect CAN data with Morai and the vehicle: `/start_code/can2morai.py`


#### **Switch Modes**
- **77**: Activate Autonomous Mode  
- **1001**: Reset Processing  

#### **Start the Simulation**
Use the following command to launch the settings: `roslaunch setting.launch`


---

## 🛠 Key Features

### **Simulation Data Processing**
- Use the `sim_data.bag` file to test and validate vehicle data.

### **Visualization**
- **Local Frame Visualization**:  
  Use `rviz/local_frame.rviz` to visualize the vehicle's local frame in RViz.  
- **Global Frame Visualization**:  
  Use `rviz/visual.rviz` to check the global frame data of the vehicle.

### **Input Control**
- Use `run_control.py` to generate control signals and experiment with different scenarios.

---

## 📋 Notes

### **Environment Compatibility**
- Ensure GNSS and CAN connections are correctly configured.

### **Mode Settings**
- Enter **77** to activate autonomous mode.  
- Use **1001** for initialization if needed.

---

## 🛡 Troubleshooting

### **CAN Connection Issues**
- Re-run the `./can.sh` script to reset the connection.

### **ROS Topic Processing Errors**
- Check the ROS topic configuration in `run_setting.py`.

### **Visualization Issues**
- Verify that the correct `.rviz` file is loaded in RViz.

---

## 📞 Contact
If you have any questions about the project, feel free to reach out:

- **Email**: cjh200001@gmail.com  
- **GitHub**: [cjhhalla](#)
