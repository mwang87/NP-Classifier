## NP Classifier

![production-integration](https://github.com/mwang87/NP-Classifier/workflows/production-integration/badge.svg)

We typically will deploy this locally. To bring everything up,
you need docker and docker-compose.

## Montai Fork Changes

**Note:** The instructions remain mostly the same, we have re-written them for clarity. Read the original instructions [here](#original-instructions)

- The fork has a new branch `dev` with a small amount of modifications. Please checkout to the `dev` branch or clone the dev branch to use the fork
- `Dockerfile` was modified to use `rdkit` the latest version, and a few more `g++` dependencies for `rdkit` were added
- `Classifier/models_folder/models/convert_keras_to_tf.py` was modified to disable `tf.keras.backend.set_learning_phase(0)`
- `Makefile` was modified to use `linux/amd64` platform for building, and running docker images

### TLDR Instructions

- In Docker Desktop for Mac, Add the path `/Users/<>/NP-Classifier/output` to `Docker -> Preferences... -> Resources -> File Sharing.` Replace `<>/` with your username/path to the `NP-Classifier` folder
- Run `cd Classifier/models_folder/models` and `sh ./get_models.sh` to download the models
- This will download the models to `Classifier/models_folder/models/models` and convert them to `HDF5` format
- Run `docker network create nginx-net` to create a network
- Run `make server-compose` to build and run the server
- Visit `http://localhost:6541/` to view the dashboard

## Original Instructions

### Local Server

#### Downloading NP Classifier Models

```
cd Classifier/models_folder/models
sh ./get_models.sh
```

NOTE: Make sure you have python installed and tensorflow version 2.3.0 installed to convert the keras models into HDF5 TF2 models.  

#### Building Dockerized Server

If you didn't do it already, you will need a network.

```shell
docker network create nginx-net
```

```shell
make server-compose
```

### Checking Model Metadata

We pass through tensorflow serving at this url:

```/model/metadata```

If the model input names change, then we need to change it in the code

### Checking input/output layer names

Input layers' names should be "input_2048" and "input_4096"

Output layer's name should be "output"

### APIs

Classify programmatically

```/classify?smiles=<>```

You can also provide cached flag to the params to get the cached version so make it faster

## License

The license as included for the software is MIT. Additionally, all data, models, and ontology are licensed as [CC0](https://creativecommons.org/share-your-work/public-domain/cc0/).

## Privacy

We try our best to balance privacy and understand how users are using our tool. As such, we keep in our logs which structures were classified but not which users queried the structure.
