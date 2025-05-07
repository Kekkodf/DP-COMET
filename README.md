<h1 align="center">
  <img src="./img/repo/COMET-DP.png" alt="EQO" width="200">
  <br>
  DP-COMET: A Differential Privacy Context Obfuscation MEchanism for Textual Data
  <br>
</h1>

<h4 align="center">Anonymous DP-COMET repository for double-blind revisions of Short-Paper submitted at <a href="https://www.cikm2025.org/" target="_blank">CIKM'25</a>. ID Paper: </h4>

<p align="center">
  <a href="https://img.shields.io/badge/license-GNU_GPL_3.0-blue.svg">
    <img src="https://img.shields.io/badge/license-GNU_GPL_3.0-blue.svg" alt="License">
  </a>
  <a href="https://img.shields.io/badge/python-3.10%2B-blue.svg">
    <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python Version">
  </a>
  <a href="https://img.shields.io/badge/conda-24%2B-blue.svg">
    <img src="https://img.shields.io/badge/conda-24%2B-blue.svg" alt="Conda Version">
  </a>
  <a href="https://img.shields.io/badge/OS-Linux%20%7C%20Windows%20%7C%20MacOS-lightgrey.svg">
    <img src="https://img.shields.io/badge/OS-Linux%20%7C%20Windows%20%7C%20MacOS-lightgrey.svg" alt="OS">
  </a>
  <a href="https://img.shields.io/badge/status-release-green.svg">
    <img src="https://img.shields.io/badge/status-release_1.0.0-green.svg" alt="Release Version">
  </a>
</p>

## 🌍 Setup Environment
1. Clone the repository

2. Generate the environment using the `environment.yml` file in `./config/` folder. You can use the following command to create a conda environment:
```bash
conda env create -f config/environment.yml
```
Then verify the environment:
```bash
conda env list
```

3. Activate the environment:
```bash
conda activate dp-comet
```

Now you can run the code in the repository with all the dependencies installed. Once you are done, you can deactivate the environment:
```bash
conda deactivate
```

In the `./config/requirements.txt` file, you can find the list of packages used in the project. 

## 📂 Directory Structure
```bash
├── config
│   ├── environment.yml
│   ├── requirements.txt
├── data
├── img
│   ├── repo
├── logs
├── src
│   ├── __init__.py
│   ├── dp-comet.py
│   ├── utils
│      ├── __init__.py
│      ├── mylogger.py
├── test
├── README.md
├── LICENSE
├── .gitignore
```
The `main.py` script will generate a sample test folder, apply the DP-COMET mechanism, perform the experiments and store the results in it. The results logs will be saved in the `logs/` folder.

## 🧪 Test DP-COMET

To test the contextual obfuscation mechanism, you can use the `main.py` script and run the following command:
```bash
python3 main.py
```

An example is shown below:


## 🆘 Support
More Information about Author(s) and Support of the Mechanism released upon acceptance of the paper due to double-anonymous review process.

## 📜 License
This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

---

> [Anonymous Author(s)](https://github.com/) &nbsp;&middot;&nbsp;
> GitHub [@no_author(s)](https://github.com/)