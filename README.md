# Personalization-proof Sniper Backdoor
Code for my research internship, based on the paper "Sniper Backdoor: Single Client Targeted Backdoor Attack in Federated Learning". SaTML'23. During the internship I will investigate the persistence of the targeted backdoor through personalization; how does local model personalization affect the backdoor performance? 

## Workflow

The project can be run sequentially using the bash scripts for i.i.d. data and non i.i.d. data. The bash scripts are set up to use a local conda venv, which can be prepared using requirements.txt. 

### IID
```bash
run.sh
```

### Non-IID
```bash
run_noniid.sh
```

It is also possible to run the steps of the project as individual parts, as results are stored locally. For example, to run the FL training:
```bash
python 1-main.py --dataname mnist --n_clients 10 --n_epochs 5 --lr 0.1 --iid --dir './' --momentum 0.9
```

