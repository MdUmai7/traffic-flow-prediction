from django.urls import path

from . import views


urlpatterns = [
    path("", views.index, name="home"),
    path("index.html", views.index, name="index"),
    path("train/rnn/", views.train_rnn, name="train_rnn"),
    path("train/gru/", views.train_gru, name="train_gru"),
    path("train/lstm/", views.train_lstm, name="train_lstm"),
    path("predict.html", views.predict, name="predict"),
    path("TrainRNN", views.TrainRNN, name="TrainRNN"),
    path("TrainGRU", views.TrainGRU, name="TrainGRU"),
    path("TrainLSTM", views.TrainLSTM, name="TrainLSTM"),
    path("Predict.html", views.Predict, name="Predict"),
    path("PredictAction", views.PredictAction, name="PredictAction"),
]
