
import tensorflow as tf
from tensorflow.keras import layers, optimizers, regularizers
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer


class NbowModel(BaseEstimator):
    def __init__(
        self, 
        vocab_sz: int = 100,
        min_df: float = 0.005,
        max_df: float = 0.75,
        dropout: float = 0.10,
        dense_sz: int = 15,
        lr: float = 0.002,
        batch_size: int = 32,
        epochs: int = 10,
    ):
        self.vocab_sz = vocab_sz
        self.min_df = min_df
        self.max_df = max_df
        self.dropout = dropout
        self.dense_sz = dense_sz
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self._init_model()

    def _init_model(self):
        # Instantiate the CountVectorizer
        self.cv = CountVectorizer(
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words="english",
            strip_accents="ascii",
            max_features=self.vocab_sz,
        )

        # Define the keras model
        inputs = tf.keras.Input(shape=(self.vocab_sz,), name="input")
        x = layers.Dropout(self.dropout)(inputs)
        x = layers.Dense(
            self.dense_sz,
            activation="relu",
            kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
        )(x)
        predictions = layers.Dense(
            1,
            activation="sigmoid",
        )(x)
        self.model = tf.keras.Model(inputs, predictions)
        opt = optimizers.Adam(learning_rate=self.lr)
        self.model.compile(
            loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"]
        )

    def fit(self, X, y):
        print(X.shape)
        res = self.cv.fit_transform(X).toarray()
        self.model.fit(x=res, y=y, batch_size=self.batch_size, epochs=self.epochs, validation_split=0.2)

    def predict(self, X):
        print(X.shape)
        res = self.cv.transform(X).toarray()
        return self.model.predict(res)

    def eval_acc(self, labels, pred, threshold=0.5):
        return balanced_accuracy_score(labels, pred > threshold, adjusted=True)

    def eval_rocauc(self, labels, pred):
        return roc_auc_score(labels, pred)

    @property
    def model_dict(self):
        return {"vectorizer": self.cv, "model": self.model}

    @classmethod
    def from_dict(cls, model_dict):
        "Get Model from dictionary"
        nbow_model = cls(len(model_dict["vectorizer"].vocabulary_))
        nbow_model.model = model_dict["model"]
        nbow_model.cv = model_dict["vectorizer"]
        return nbow_model

    def __repr__(self):
        return f"{self.__class__.__name__} <{hash(frozenset(self.get_params()))}>"
