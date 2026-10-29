## Environment Setup

This project uses Python 3.11 and Qiskit.

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/SoumyabrataBose52/QEM_files.git
cd QEM_files
```

**2. Create and activate a virtual environment**

```bash
conda env create -f environment.yml
conda activate qem
```

**3. Install dependencies**

```bash
python -m pip install -r requirements.txt
```

**4. Verify the installation**

```bash
python -c "import qiskit; print('Qiskit version:', qiskit.__version__)"
```
