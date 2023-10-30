from metaflow import (
    FlowSpec,
    step,
    Flow,
    current,
    Parameter,
    IncludeFile,
    card,
    current,
)
from metaflow.cards import Table, Markdown, Artifact, Image
import numpy as np
from dataclasses import dataclass

def labeling_function(row):
    if row["rating"] >= 4 and row["recommended_ind"] == 1:
        return 1
    elif row["rating"] < 4 and row["recommended_ind"] == 0:
        return 0
    else:
        return np.nan

@dataclass
class ModelResult:
    "A custom struct for storing model evaluation results."
    name: None
    params: None
    pathspec: None
    acc: None
    rocauc: None
    bal_acc: None


class BaselineChallenge(FlowSpec):
    split_size = Parameter("split-sz", default=0.2)
    data = IncludeFile("data", default="data/womens_clothing_ecommerce_reviews.csv")
    n_models = Parameter("n_models", default=15)
    scoring = Parameter("scoring", default="bal_acc")

    @step
    def start(self):
        import pandas as pd
        import io
        from sklearn.model_selection import train_test_split

        # load dataset packaged with the flow.
        # this technique is convenient when working with small datasets that need to move to remove tasks.
        df = pd.read_csv(io.StringIO(self.data), index_col=0)

        # filter down to reviews and labels
        df.columns = ["_".join(name.lower().strip().split()) for name in df.columns]
        df = df[~df.review_text.isna()]
        df["review"] = df["review_text"].astype("str")
        _has_review_df = df[df["review_text"] != "nan"]
        reviews = _has_review_df["review_text"]
        labels = _has_review_df.apply(labeling_function, axis=1)
        self.df = pd.DataFrame({"label": labels, **_has_review_df}).dropna()

        # split the data 80/20, or by using the flow's split-sz CLI argument
        _df = pd.DataFrame({"review": reviews, "label": labels}).dropna()
        self.traindf, self.valdf = train_test_split(_df, test_size=self.split_size)
        print(f"num of rows in train set: {self.traindf.shape[0]}")
        print(f"num of rows in validation set: {self.valdf.shape[0]}")

        self.next(self.baseline, self.model_fanout)

    @step
    def baseline(self):
        "Compute the baseline"
        from sklearn.dummy import DummyClassifier
        from sklearn.metrics import balanced_accuracy_score, roc_auc_score, accuracy_score

        baseline_clf = DummyClassifier()

        self._name = "baseline"
        params = baseline_clf.get_params()
        pathspec = f"{current.flow_name}/{current.run_id}/{current.step_name}/{current.task_id}"

        baseline_clf.fit(self.traindf, y=self.traindf.label)

        predictions = baseline_clf.predict_proba(self.valdf)[:,1]
        acc = accuracy_score(self.valdf.label, predictions > 0.5)
        bal_acc = balanced_accuracy_score(self.valdf.label, predictions > 0.5, adjusted=True)
        rocauc = roc_auc_score(self.valdf.label, predictions)

        self.results = [ModelResult(repr(baseline_clf), params, pathspec, acc, rocauc, bal_acc)]
        self.next(self.aggregate)

    @step
    def model_fanout(self):
        from sklearn.model_selection import ParameterSampler
        param_sets = {
            'batch_size': [8, 16, 32, 32],
            'dense_sz': [10, 15, 20],
            'dropout': [0.05, 0.1, 0.15],
            'epochs': [10, 20, 30],
            'lr': [0.001, 0.002, 0.005],
            'max_df': [0.7, 0.75, 0.8],
            'min_df': [0.001, 0.005, 0.01],
            'vocab_sz': [250, 300, 350, 400],
        }
        ps = ParameterSampler(param_sets, n_iter=self.n_models)
        self.hyperparam_set = list(ps)
        print(f"Traing model for {len(self.hyperparam_set)} different hyperparam sets")
        self.next(self.model, foreach="hyperparam_set")

    @step
    def model(self):
        from model import NbowModel

        model = NbowModel(**self.input)

        self._name = "model"
        params = model.get_params()
        pathspec = f"{current.flow_name}/{current.run_id}/{current.step_name}/{current.task_id}"

        print(f"Training model: {model}")
        print(f" Params: {params}")
        model.fit(X=self.traindf["review"], y=self.traindf["label"])

        valy = self.valdf["label"]
        predictions = model.predict(self.valdf["review"])
        acc = model.eval_acc(valy, predictions)
        bal_acc = model.eval_bal_acc(valy, predictions)
        rocauc = model.eval_rocauc(valy, predictions)
        self.result = ModelResult(
            repr(model),
            params,
            pathspec,
            acc,
            rocauc,
            bal_acc,
        )

        self.next(self.join)

    @step
    def join(self, inputs):
        self.results = [i.result for i in inputs]
        self.next(self.aggregate)

    def add_one(self, rows, result, df):
        "A helper function to load results."
        rows.append(
            [
                Markdown(result.name),
                Artifact(result.params),
                Artifact(result.pathspec),
                Artifact(result.bal_acc),
                Artifact(result.acc),
                Artifact(result.rocauc),
            ]
        )
        df["name"].append(result.name)
        df["accuracy"].append(result.acc)
        df["bal_acc"].append(result.bal_acc)
        df["roc_auc"].append(result.rocauc)
        return rows, df

    @card(type="corise")
    @step
    def aggregate(self, inputs):
        from collections import defaultdict
        from itertools import chain
        import seaborn as sns
        import matplotlib.pyplot as plt
        from matplotlib import rcParams

        def cmp(mr: ModelResult):
            return getattr(mr, self.scoring)

        results = list(chain(*[i.results for i in inputs]))
        self.results = list(sorted(results, key=cmp, reverse=True))

        rows = []
        violin_plot_df = defaultdict(list)
        for result in self.results:
            print(result)
            rows, violin_plot_df = self.add_one(rows, result, violin_plot_df)
        self.plot_df = violin_plot_df

        current.card.append(Markdown("# All models from this flow run"))
        current.card.append(
            Table(
                rows,
                headers=["Model name", "Params", "Task pathspec", "Bal Accuracy", "Accuracy", "ROCAUC"],
            )
        )

        rcParams.update({"figure.autolayout": True})
        
        fig1, ax1 = plt.subplots(1, 1)
        plt.xticks(rotation=90)
        sns.violinplot(data=violin_plot_df, x="name", y="accuracy", ax=ax1)
        current.card.append(Image.from_matplotlib(fig1))

        fig2, ax2 = plt.subplots(1, 1, figsize=(10,10))
        sns.scatterplot(data=violin_plot_df, x="bal_acc", y="roc_auc", size="accuracy", hue="name", alpha=0.5, ax=ax2)
        plt.tight_layout()
        current.card.append(Image.from_matplotlib(fig2))

        self.next(self.end)

    @step
    def end(self):
        self.best_result = self.results[0]
        print(f"Best Model: {self.best_result.name}")
        print(f"  pathspec - {self.best_result.pathspec}")
        print(f"  params   - {self.best_result.params}")
        print(f"  acc      - {self.best_result.acc}")
        print(f"  bal-acc  - {self.best_result.bal_acc}")
        print(f"  roc-auc  - {self.best_result.rocauc}")


if __name__ == "__main__":
    BaselineChallenge()
