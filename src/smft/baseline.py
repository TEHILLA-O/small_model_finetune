"""TF-IDF plus logistic regression baseline trained on exactly the same SST-2 rows."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from smft.metrics import classification_report


def tfidf_baseline(train, dev, eval_splits):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vec.fit_transform(train.texts)
    best = None
    for c in [0.25, 1.0, 4.0, 16.0]:
        m = LogisticRegression(C=c, max_iter=2000).fit(X, train.labels)
        acc = float((m.predict(vec.transform(dev.texts)) == np.array(dev.labels)).mean())
        if best is None or acc > best[0]:
            best = (acc, c, m)
    _, c, model = best
    probs, metrics = {}, {"C_chosen_on_dev": c}
    for s in eval_splits:
        p = model.predict_proba(vec.transform(s.texts))[:, 1]
        probs[s.name] = p
        metrics[s.name] = classification_report(p, s.labels)
    return {"probs": probs, "metrics": metrics}
