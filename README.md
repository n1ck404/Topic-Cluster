# Topic Cluster

Using NLP and machine learning technology to implement a BI tool that can find the themes / topics within a text dataset.

## Simple usage

Download the code, and run:

```shell
pip install -r requirements.txt
```

to download the dependency library.

```shell
python -m spacy download en
```

to download the model of Spacy

```shell
python UIFunction.py
```

which is an entrance for the UI.

Also can use:

```shell
python TCmodel.py
```

or import this model in code to use special function, such as

```python
import Tcmodel.py
model = Tcmodel.TcModel()
model.load('../lda')
model._doc_topic
```

## File structure

```shell
.
├── code
│   ├── Json_to_csv.py
│   ├── TCmodel.py
│   ├── UI.py
│   ├── UIFunction.py
│   └── icon.png
├── README.md
└── requirements.txt
```

## Denpendency

Python >= 3.6

```shell
gensim>=3.4.0,
matplotlib>=2.2.2,
nltk>=3.3,
numpy>=1.14.3,
pandas>=0.23.0,
Pillow>=5.1.0,
pyLDAvis>=2.1.2,
seaborn>=0.8.1,
spacy>=2.0.12,
wordcloud>=1.5.0,
PyQt5>=5.11.3,
```

