# EVA Pahse 2 : Word Embedding#

Word embedding is a process of vectorizing the tokens from a text dataset. Embeding layer takes tokens converted into integers as input & outputs an associated vector.

Learning word embedding while training th enetwork is not effective when the data set is small. Hence in this assignment we are using pretrained word embedding which was trained in a different problem. We are using GloVe as precomputed database of word embeddings.

Glove precomputed data base with 6B tokens & dimension 100 are used in the code.  We are training our model for 1st 8000 samples in the dataset. Trained the model for 50 epochs with final validation accuracy 68.39%.





![](https://i.imgur.com/hvyqyNy.png)





​                                                                        Fig 1: Final Validation accuracy













![](https://i.imgur.com/gumyuU4.png)



​                   Fig 2: Validation accuracy vs training accuracy & validation loss vs training loss





