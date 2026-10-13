## Environment Setup

This project uses Python 3.11 and Qiskit for quantum computing experiments.

### Prerequisites

- Python 3.11
- pip

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/SoumyabrataBose52/QEM_files.git
cd QEM_files
```

**2. Create and activate a virtual environment**

```bash
python3.11 -m venv venv2
source venv2/bin/activate
```

**3. Install dependencies**

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**4. Verify the installation**

```bash
python -c "import qiskit; print('Qiskit version:', qiskit.__version__)"
```
