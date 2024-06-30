## Steps to run project

* Clone repository
```
git clone https://github.com/Ozu-bezariusu/hillel_ml.git
```

* Navigate to homework_9 folder
```
cd hillel_ml/homework_9
```
* Install conda env
```
conda env create -f environment.yml
```
* Create jupyter kernel
```
python -m ipykernel install --user --name=myenv
```

* Open `fcnn.ipynb`
* Go to `Device Set up` section and choose your GPU device.
* Go to VS Code, choose kernel you've created and run `fcnn.ipynb`