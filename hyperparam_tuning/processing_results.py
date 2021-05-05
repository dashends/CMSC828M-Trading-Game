from os import listdir
from os.path import isfile, join

dir1 = listdir("optuna_params/")
dir2 = listdir("optuna_params2/")


results = []
for f in dir1:
    with open(join("optuna_params/", f), "r") as file:
        l = file.read().split("\n")
        mean = float(l[0].split("\t")[0].split()[-1])
        std = float(l[0].split("\t")[1].split()[-1])
        results.append(("1_" + f, mean, std, l[1]))
for f in dir2:
    with open(join("optuna_params2/", f), "r") as file:
        l = file.read().split("\n")
        mean = float(l[0].split("\t")[0].split()[-1])
        std = float(l[0].split("\t")[1].split()[-1])
        results.append(("2_" + f, mean, std, l[1]))
		
results = sorted(results, key=lambda x: -x[1])
print(results)