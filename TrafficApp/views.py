import base64
from io import BytesIO
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from django.shortcuts import render
from keras.layers import Dense, Dropout, GRU, LSTM, SimpleRNN
from keras.models import Sequential, model_from_json
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
DATASET_PATH = BASE_DIR / "traffic.csv"
WINDOW_SIZE = 10

MODEL_DIR.mkdir(exist_ok=True)

input_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler = MinMaxScaler(feature_range=(0, 1))


def _load_training_data():
    dataframe = pd.read_csv(DATASET_PATH, nrows=300)
    dataframe["DateTime"] = pd.to_datetime(dataframe["DateTime"])
    dataframe = dataframe.drop(columns=["ID"])
    dataframe["Year"] = dataframe["DateTime"].dt.year
    dataframe["Month"] = dataframe["DateTime"].dt.month
    dataframe["Date_no"] = dataframe["DateTime"].dt.day
    dataframe["Hour"] = dataframe["DateTime"].dt.hour

    feature_frame = dataframe[["Junction", "Year", "Month", "Date_no", "Hour"]].astype("float32")
    target_frame = dataframe[["Vehicles"]].astype("float32")

    scaled_features = input_scaler.fit_transform(feature_frame.values)
    scaled_targets = target_scaler.fit_transform(target_frame.values)

    x_train = []
    y_train = []
    for index in range(WINDOW_SIZE, len(scaled_features)):
        x_train.append(scaled_features[index - WINDOW_SIZE:index, :])
        y_train.append(scaled_targets[index, 0])

    return (
        dataframe,
        feature_frame,
        np.asarray(x_train, dtype="float32"),
        np.asarray(y_train, dtype="float32"),
    )


TRAFFIC_DATA, FEATURE_FRAME, X_TRAIN, Y_TRAIN = _load_training_data()


def _load_saved_model(json_path, weights_path):
    if not json_path.exists() or not weights_path.exists():
        return None

    with json_path.open("r", encoding="utf-8") as json_file:
        loaded_model = model_from_json(json_file.read())

    loaded_model.load_weights(str(weights_path))
    return loaded_model


def _save_model(model, json_path, weights_path):
    model.save_weights(str(weights_path))
    with json_path.open("w", encoding="utf-8") as json_file:
        json_file.write(model.to_json())


def _plot_training_graph(actual_values, predicted_values, title):
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(actual_values, color="#ef476f", linewidth=2, label="Actual")
    axis.plot(predicted_values, color="#06d6a0", linewidth=2, label="Predicted")
    axis.set_title(title)
    axis.set_xlabel("Training samples")
    axis.set_ylabel("Vehicles")
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right")
    figure.tight_layout()

    buffer = BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _plot_forecast_graph(hour_labels, predicted_values, area_name):
    figure, axis = plt.subplots(figsize=(10, 4))
    x_positions = np.arange(len(predicted_values))
    axis.plot(x_positions, predicted_values, color="#118ab2", linewidth=2.5, marker="o", markersize=4)
    axis.fill_between(x_positions, predicted_values, color="#118ab2", alpha=0.12)
    axis.set_title(f"24-Hour Forecast for {area_name}")
    axis.set_xlabel("Hour")
    axis.set_ylabel("Predicted vehicles")
    tick_positions = list(range(0, len(hour_labels), 3))
    if tick_positions[-1] != len(hour_labels) - 1:
        tick_positions.append(len(hour_labels) - 1)
    axis.set_xticks(tick_positions)
    axis.set_xticklabels([hour_labels[position] for position in tick_positions])
    axis.grid(alpha=0.25)
    figure.tight_layout()

    buffer = BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _calculate_rmse(actual_values, predicted_values):
    return float(np.sqrt(mean_squared_error(actual_values, predicted_values)))


def _build_rnn_model():
    model = Sequential()
    model.add(SimpleRNN(units=32, input_shape=(X_TRAIN.shape[1], X_TRAIN.shape[2]), activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


def _build_gru_model():
    model = Sequential()
    model.add(GRU(units=32, input_shape=(X_TRAIN.shape[1], X_TRAIN.shape[2]), activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


def _build_lstm_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_TRAIN.shape[1], X_TRAIN.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def _load_or_train_model(model_builder, json_name, weights_name, epochs):
    json_path = MODEL_DIR / json_name
    weights_path = MODEL_DIR / weights_name

    saved_model = _load_saved_model(json_path, weights_path)
    if saved_model is not None:
        return saved_model

    trained_model = model_builder()
    trained_model.fit(X_TRAIN, Y_TRAIN, epochs=epochs, batch_size=16, verbose=0)
    _save_model(trained_model, json_path, weights_path)
    return trained_model


def _render_training_page(request, template_name, model_builder, json_name, weights_name, epochs, title):
    context = {}
    if request.method == "POST":
        model = _load_or_train_model(model_builder, json_name, weights_name, epochs)
        predicted_values = model.predict(X_TRAIN, verbose=0)
        predicted_values = target_scaler.inverse_transform(predicted_values).ravel()
        actual_values = target_scaler.inverse_transform(Y_TRAIN.reshape(-1, 1)).ravel()

        context["graph"] = _plot_training_graph(actual_values, predicted_values, title)
        context["rmse"] = f"{_calculate_rmse(actual_values, predicted_values):.2f}"

    return render(request, template_name, context)


def _resolve_junction(area_value):
    area_text = str(area_value or "").strip()
    if not area_text:
        return 1, "Junction 1"

    match = re.search(r"\d+", area_text)
    if match:
        junction = int(match.group())
        if 1 <= junction <= 4:
            return junction, area_text if not area_text.isdigit() else f"Junction {junction}"

    junction = (sum(ord(character) for character in area_text) % 4) + 1
    return junction, area_text


# def _traffic_status(predicted_values):
#     peak_traffic = max(predicted_values)
#     if peak_traffic < 20:
#         return "Low"
#     if peak_traffic < 35:
#         return "Medium"
#     return "High"
# def _traffic_status(predicted_values):
#     avg_traffic = sum(predicted_values) / len(predicted_values)

#     if avg_traffic < 18:
#         return "Low"
#     elif avg_traffic < 22:
#         return "Medium"
#     else:
#         return "High"
# def _traffic_status(predicted_values):
#     avg = sum(predicted_values) / len(predicted_values)

#     if avg < 0.4 * max(predicted_values):
#         return "Low"
#     elif avg < 0.7 * max(predicted_values):
#         return "Medium"
#     else:
#         return "High"
# def _traffic_status(predicted_values):
#     avg = sum(predicted_values) / len(predicted_values)

#     if avg < 18:
#         return "Low"
#     elif avg < 21:
#         return "Medium"
#     else:
#         return "High"

def _seed_sequence_for_junction(junction):
    junction_rows = TRAFFIC_DATA[TRAFFIC_DATA["Junction"] == junction]
    if len(junction_rows) >= WINDOW_SIZE:
        feature_rows = FEATURE_FRAME.loc[junction_rows.index].tail(WINDOW_SIZE)
        last_timestamp = junction_rows["DateTime"].iloc[-1]
    else:
        feature_rows = FEATURE_FRAME.tail(WINDOW_SIZE)
        last_timestamp = TRAFFIC_DATA["DateTime"].iloc[-1]

    scaled_sequence = input_scaler.transform(feature_rows.values.astype("float32"))
    return scaled_sequence, pd.Timestamp(last_timestamp)


def _generate_forecast(area_value):
    junction, area_name = _resolve_junction(area_value)
    model = _load_or_train_model(_build_lstm_model, "model.json", "model_weights.h5", epochs=100)
    sequence, current_time = _seed_sequence_for_junction(junction)

    predicted_values = []
    hour_labels = []

    for _ in range(24):
        scaled_prediction = model.predict(sequence[np.newaxis, :, :], verbose=0)
        predicted_value = float(target_scaler.inverse_transform(scaled_prediction)[0][0])

        current_time = current_time + pd.Timedelta(hours=1)
        predicted_values.append(max(predicted_value, 0.0))
        hour_labels.append(current_time.strftime("%H:%M"))

        next_features = np.array(
            [[junction, current_time.year, current_time.month, current_time.day, current_time.hour]],
            dtype="float32",
        )
        scaled_next_features = input_scaler.transform(next_features)
        sequence = np.vstack([sequence[1:], scaled_next_features])

    # return {
    #     "area": area_name,
    #     "graph": _plot_forecast_graph(hour_labels, predicted_values, area_name),
    #     "status": _traffic_status(predicted_values),
    # }
    peak_value = max(predicted_values)
    peak_index = predicted_values.index(peak_value)
    peak_hour = hour_labels[peak_index]

    avg_value = sum(predicted_values) / len(predicted_values)

    return {
    "area": area_name,
    "graph": _plot_forecast_graph(hour_labels, predicted_values, area_name),

    # NEW DATA
    "peak_value": round(peak_value, 2),
    "peak_hour": peak_hour,
    "avg_value": round(avg_value, 2),
}


def index(request):
    return render(request, "index.html")


def train_rnn(request):
    return _render_training_page(
        request,
        "train_rnn.html",
        _build_rnn_model,
        "rnn_model.json",
        "rnn_model_weights.h5",
        epochs=100,
        title="Simple RNN Training Results",
    )


def train_gru(request):
    return _render_training_page(
        request,
        "train_gru.html",
        _build_gru_model,
        "gru_model.json",
        "gru_model_weights.h5",
        epochs=700,
        title="GRU Training Results",
    )


def train_lstm(request):
    return _render_training_page(
        request,
        "train_lstm.html",
        _build_lstm_model,
        "model.json",
        "model_weights.h5",
        epochs=100,
        title="LSTM Training Results",
    )


def predict(request):
    context = {}
    if request.method == "POST":
        area_value = request.POST.get("area") or request.POST.get("t1")
        context.update(_generate_forecast(area_value))
    return render(request, "predict.html", context)


def TrainRNN(request):
    return train_rnn(request)


def TrainGRU(request):
    return train_gru(request)


def TrainLSTM(request):
    return train_lstm(request)


def Predict(request):
    return predict(request)


def PredictAction(request):
    return predict(request)
