# dl_car_segmentation

## Cluster setup

https://docs.google.com/document/d/1jzy5hv6RheZ_8ec-4QrVJz3P-zotoyLpagMSiYCJZy0/edit

### Login

Login via:

```
ssh s23xxxx@login.hpc.dtu.dk
```

You must need the DTU VPN or be at DTU

VPN: https://itswiki.compute.dtu.dk/index.php/Cisco_VPN

### Clone Git

```
git clone https://github.com/darioschmid/dl_car_segmentation.git
```

### Upload Data

Upload Data via the following command on your local machine:

```
scp arrays-20231106T124132Z-001.zip s23xxxx@login.hpc.dtu.dk:~/data
```

Login…
and unzip

```
unzip data/arrays-20231106T124132Z-001.zip -d dl_car_segmentation/data
```

### Prepare Python

```
cd dl_car_segmentation

module load python3/3.10.12

python3 -m venv .venv

source .venv/bin/activate

pip3 install --upgrade pip

pip3 install -r requirements.txt
```

Setup done

## Train on Cluster

Login via:

```
ssh s23xxxx@login.hpc.dtu.dk
```

You must need a VPN or be at DTU

VPN: https://itswiki.compute.dtu.dk/index.php/Cisco_VPN

prepare config.json and upload the data

Add a job to the GPU Queue

```
bsub < cluster-train.sh
```

# Augment data

Firstly zip the landscapes folder and upload it to the cluster by executing the following command on your local machine:

```
scp landscapes.zip s23xxxx@login.hpc.dtu.dk:~/data
```

SSH into the cluster, create folders and unzip the landscapes:

```
ssh s23xxxx@login.hpc.dtu.dk
cd dl_car_segmentation/carseg_data
mkdir images
mkdir arrays_augmented
unzip ~/data/landscapes.zip ~/dl_car_segmentation/carseg_data/images
```

In order to augment the data by placing cars onto landscape images, run the following command:

```
cd dl_car_segmentation/helper
python3 augment_data.py
```
# Augment Photo and Create Train and Test photos
Im postTrainSplitter.py muss der pfad geändet werden wo man die Train und Test gespeichter werden.
Ausserdem noch den Pfad zu den photo_arrays geben (alle von 0-169)
Die Anzahl von augmentierten photos angeben (weiter unten) bei 0 kopiert er einfach nur die echten bilder

```
source .venv/bin/activate   ### on cluster
cd dl_car_segmentation/helper
python3 postTrainSplitter.py
```

# Histogram erstellen
Im config des trainierten models den img_dir auf den ***/postTrain/test ordener setzen...
```
python3 test.py -r pfad_zu_model_pth
```
Nun sollte er alle 34 test bilder (photo_1-photo_34) nehmen und das model laufen lassen, Histogram wird automatisch erstellt.

# training a model (assuming ssh connection already open, python module loaded and data already augmented)
configure config.json

Pix2Pix:
```
"name": "Pix2Pix",
...
"type": "Pix2PixModel",
    "args": {
      "gf_dim": 10,
      "df_dim": 64,
      "c_dim": 3
    }
...
"trainer": {
    "type": "GAN",
    ...
}
```

UNet:
```
"name": "UNet",
...
"type": "Unet",
    "args": {
      "n_channels": 3,
      "n_classes": 10,
      "bilinear": false
    }
...
"trainer": {
    "type": "Normal",
    ...
}
```

specify paths for the data loader, batch size and epochs
```
"n_gpu": 1
...
"data_loader": {
  "data_dir": "TRAIN_PATH_TO_NPY_ARRAYS",
  "batch_size": X,
}
...
"trainer": {
  "epochs": 40,
}
```

adjust cluster-train.sh

configure memory on gpu: line 14
```
#BSUB -R "rusage[mem=30GB]"
```

check runtime: line 12
```
#BSUB -W 1:00
```

check: line 37
```
python3 train.py -c config.json
```

check if pipeline runs on CPU before putting it on the GPU
set batch size to 1
```
"batch_size": 1
```

run train.py on cpu
```
cd dl_car_segmentation
python3 train.py -c config.json
```

if losses are getting printed, continue with next step

create output folder in /dl_car_segmentation

train on GPU
```
bsub < cluster-train.sh
```
Output prints found in new folder 

create histogram

check name of last saved checkpoint in gpu_xxxxxxxx.out file
example for pix2pix:
```
Saving checkpoint: saved/models/Pix2Pix/1129_214518/checkpoint-epoch20.pth ...
```

adjust corresponding config.json
```
"n_gpu": 1
"data_loader": {
  "data_dir": "TEST_PATH_TO_NPY_ARRAYS",
}
```

add following code to test.py in line 92 (after creating the histogram)
```
plt.savefig("hist.pdf")
```

run test.py on checkpoint
example for pix2pix:
```
python test.py --resume ./saved/models/Pix2Pix/1129_214518/checkpoint-epoch20.pth
```

download gpu_xxxxxxxx.out, hist.pdf, checkpoint.pth and corresponding config.json as well as the events.out file from logs to save all relevant information of the model  

