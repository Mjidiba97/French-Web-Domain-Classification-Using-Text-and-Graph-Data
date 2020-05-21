# French-Web-Domain-Classification-Using-Text-and-Graph-Data

Web Domain classification is a major field of research for information retrieval. Most of the studies claim that domain classification is strongly correlated with its content but requires very appro- priate descriptors. In this paper, we are aiming to explore both text and graph data for the creation of a classification model for french web domains.


## Input Data
Our input dataset consist of a directed graph, a list of texts and labels.

<b>graph_data</b> (.txt file): the data is in the form of a directed graph. This graph contains 28002 vertices and 319498 directed weighted edges. Nodes correspond to domains and edges correspond to the total number of hyperlinks connecting two domains.

<b>text_data</b> (folder): each file in the folder represents the total text of all the web pages of the corresponding domain. We have a total of 2554 French domains with text data.

<b>train_labels</b> and <b>test_labels</b>: they contain the indexes of the doamins along with their labels. In total we have 8 different labels: business/finance, entertainment, tech/science, education/research, politics/government/law, health/medical, news/press and sports.

## Code

The notebook file is a numerical implementation that reproduces all the graphs and results in the paper. We also included two python files: 'text_baseline.py' and 'graph_baseline.py', which are basically blueprints that can serve for the reader to implement their own models and do more experimentation and validation in ease.
