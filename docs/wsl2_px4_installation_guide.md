# Step 1: Aerospace Simulation Environment Setup
**Target:** Windows 10/11 -> Ubuntu 22.04 (WSL2) -> PX4 SITL -> Gazebo

Because PX4, ROS 2, and Gazebo are built exclusively for Linux, we must install a native Linux kernel inside your Windows machine using Windows Subsystem for Linux (WSL2).

---

## Phase A: Install WSL2 and Ubuntu
*You must run this command in a Windows PowerShell opened as **Administrator**.*

1. Open your Start Menu, type `PowerShell`, right-click it, and select **Run as Administrator**.
2. Run the following command:
   ```powershell
   wsl --install -d Ubuntu-22.04
   ```
3. **Restart your computer.**
4. When you log back into Windows, a Ubuntu terminal will automatically pop up asking you to create a UNIX username and password. Do so.

---

## Phase B: Download the PX4 Flight Stack
*Run these commands inside your newly created **Ubuntu** terminal, NOT PowerShell.*

1. Update your brand-new Linux system:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```
2. Clone the official PX4 Autopilot source code:
   ```bash
   git clone https://github.com/PX4/PX4-Autopilot.git --recursive
   ```
3. Navigate into the directory:
   ```bash
   cd PX4-Autopilot
   ```

---

## Phase C: Install the Physics Engine (Gazebo & Dependencies)
*PX4 provides a massive setup script that automatically installs Gazebo, compilers, and Python libraries.*

1. Run the official Ubuntu setup script:
   ```bash
   bash ./Tools/setup/ubuntu.sh
   ```
   *(Note: This will take 15-30 minutes and download several gigabytes of data. It will prompt you for your Linux password).*
2. **Close the Ubuntu terminal and open a new one** (this applies the new environment variables).

---

## Phase D: Launch Your First 3D Simulation!
*If everything is installed correctly, you can now launch a 3D quadcopter.*

1. Navigate back to the PX4 folder:
   ```bash
   cd PX4-Autopilot
   ```
2. Build and launch the Gazebo simulation:
   ```bash
   make px4_sitl gz_x500
   ```

If successful, a 3D window (Gazebo) will open showing a quadcopter sitting on a virtual runway, waiting for our ROS 2 OS to command it to take off!
