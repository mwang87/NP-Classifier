## NP Classifier

We typitcally will deploy this locally. To bring everything up

```server-compose```

### Checking Model Metadata

We pass through tensorflow serving at this url:

```/model/metadata```

### APIs

Classify programmatically 

```/classify?smiles=<>```

You can also provide cached flag to the params to get the cached version so make it faster