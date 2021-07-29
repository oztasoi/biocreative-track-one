## Short Term Improvements:
- [ ] Retrieve an output with the current model from the given test data set as Run 1.
- [ ] Change all evaluations based on the micro-average system.

## Long Term Improvements:
- [ ] Add another optimizer, such as AdamW.
- [ ] Integrate graph based approach with the BERT.
- [ ] Improve current hyperparameter tuning by expanding the lr and wd instance sets. (7, 7) - (9, 9)
    - [ ] Change the weight decay. Emphasize it more.
- [ ] Create an ensemble voting based on majority voting.
    - [ ] Ensemble Models: dmis-lab/biobert-v1.1, monologg/biobert_v1.1_pubmed, allenai/scibert_scivocab_uncased