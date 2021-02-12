## NP Classifier

![production-integration](https://github.com/mwang87/NP-Classifier/workflows/production-integration/badge.svg)

We typitcally will deploy this locally. To bring everything up

```server-compose```

### Checking Model Metadata

We pass through tensorflow serving at this url:

```/model/metadata```

If the model input names change, then we need to change it in the code

### APIs

Classify programmatically 

```/classify?smiles=<>```

You can also provide cached flag to the params to get the cached version so make it faster

## License

The license as included for the software is MIT. Additionally, all data, models, and ontology are licensed as [CC0](https://creativecommons.org/share-your-work/public-domain/cc0/).

## Privacy

We try our best to balance privacy and understand how users are using our tool. As such, we keep in our logs which structures were classified but not which users queried the structure. 
