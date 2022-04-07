# Coronavirus tweets NLP - Text Classification with LSTMs
<hr/>


Dataset from [kaggle](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification/code) 

## Install requirements
```shell
pip install -r requirments.txt
```

## Usage

<hr/>

### `corona_nlp_api.py` - Flask API

### Arguments
- `--port`  Server port

### Example
```shell
python corona_nlp_api.py --port 80
```
<br/>

<hr/>


### API Inference 

`<api_url>/sentiment/<text>`

### Example
`http://127.0.0.1:80/sentiment/I hate COVID`

