from importlib.resources import path

import numpy
import IPython
import keras
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
import math
import re
import os
from plot_keras_history import show_history
from Windows import MultiStepLastBaseline, RepeatBaseline, WindowGenerator
from sklearn.metrics import r2_score

dir_workspace = "./BBDD/"

#mpl.rcParams['figure.figsize'] = (8, 6)
#mpl.rcParams['axes.grid'] = False

def listar_Zonas_All():
    dirlist = os.listdir(dir_workspace)
    for f in dirlist:
        print(f)
        for data in os.listdir(dir_workspace+f):
            print("\t-->"+data)


def pedirModeloyDatos(climateZone):
    path = dir_workspace + climateZone + "/"
    if not os.path.exists(path):
        print("La zona climatologica no existe")
    else:
        try:
            model = keras.models.load_model(path + 'model.h5')
            test_df = pd.read_csv(path + 'solarData.csv')
        except Exception as e:
            print("Dicha zona no tiene data set")
            print(e)
    print("Ha terminado la lectura")
    return model, test_df, path


def listar_Zonas_Prediccion_P():
    try:
        for f in os.listdir(dir_workspace):
            for file in os.listdir(dir_workspace + f):
                if "predicciones_electricidad.csv" == file:
                    print(f)

    except:
        print("Dicha zona no tiene data set")


def listar_Zonas_Prediccion_S():
    try:
        for f in os.listdir(dir_workspace):
            for file in os.listdir(dir_workspace + f):
                if "predicciones_solares.csv" == file:
                    print(f)

    except:
        print("Dicha zona no tiene data set")



def listar_Zonas_Modelos():
    try:
        for f in os.listdir(dir_workspace):
            for file in os.listdir(dir_workspace+f):
                    if "model.h5" == file:
                        print(f)

    except:
        print("Dicha zona no tiene data set")

def add_Data_to_DataSet(climateZone, csv_path, new_data):
    path = dir_workspace + climateZone + "/"
    if not os.path.exists(path):
        print("La zona climatologica no existe")
    else:
        try:
            df = pd.read_csv(path + csv_path)
            df_new = pd.read_csv(new_data)
            vertical_concat = pd.concat([df, df_new], axis=0)
            vertical_concat.to_csv(path + csv_path, index=False)
        except:
            print("Dicha zona no tiene data set")

def pedirDataSets(climateZone):
    path = dir_workspace + climateZone + "/"
    if not os.path.exists(path):
        print("La zona climatologica no existe")
    else:
        try:
            train_df = pd.read_csv(path + "train.csv")
            val_df = pd.read_csv(path + "val.csv")
            test_df = pd.read_csv(path + "test.csv")
        except:
            print("Dicha zona no tiene data set")
    return train_df, val_df, test_df, path

def pedirDirectorioYData(climateZone, csv_path):
    path = dir_workspace + climateZone + "/"
    if not os.path.exists(path):
        print("La zona climatologica no existe")
    else:
        try:
            df = pd.read_csv(path + csv_path)
        except:
            print("Dicha zona no tiene data set")
    return path, df

def listar_Resultados_Zona(climateZone):
    path = dir_workspace + climateZone + "/"
    if not os.path.exists(path):
        print("La zona climatologica no existe")
    else:
        try:
            for f in os.listdir(path):
                if "training_results.pdf" == f:
                    print(f)

        except:
            print("Dicha zona no tiene data set")

def listar_Predicciones_Zona(climateZone):
    path = dir_workspace + climateZone + "/"
    if not os.path.exists(path):
        print("La zona climatologica no existe")
    else:
        try:
            for f in os.listdir(path):
                if "predicciones_solares.csv" == f:
                    df = pd.read_csv(path + f)
                    print(df)
                elif "predicciones_electricidad.csv" == f:
                    df = pd.read_csv(path + f)
                    print(df)

        except:
            print("Dicha zona no tiene data set")


def get_general_dir():
    return dir_workspace

def set_general_dir(dir):
    global dir_workspace
    dir_workspace = dir

def regex_filter(val, regex):
    val = str(val)
    if val:
        mo = re.search(regex, val)
        if mo:
            return True
        else:
            return False
    else:
        return False


def plot_hist_data_attribute(attribute, df, vSolar, path):
    list_data = []
    if attribute in vSolar:
        for value in df[attribute]:
            if value < 0 or value >= 10:
                list_data.append(value)

        _, bins = pd.cut(list_data, bins=200, retbins=True)
        plt.title(attribute)
        plt.hist(list_data, bins, color='blue')
        plt.savefig(path + "images/" + attribute+"Hist.png", bbox_inches='tight')
        plt.close()
    else:
        print("no es solar")
        _, bins = pd.cut(df[attribute], bins=200, retbins=True)
        plt.title(attribute)
        plt.hist(df[attribute], bins, color='blue')
        plt.savefig(path + "images/" + attribute+"Hist.png", bbox_inches='tight')
        plt.close()



def cargarDataSet(climateZone, name_data_set_raw, csv_path):
    df = pd.read_csv(csv_path)
    path = dir_workspace + climateZone + "/"
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path+"images")
        path = path + name_data_set_raw
        df.to_csv(path, index=False)
        return True
    return False


def forzarCargarDataSet(climateZone, name_data_set_raw, csv_path):
    path = dir_workspace + climateZone + "/"
    filelist = os.listdir(path)
    for f in filelist:
        os.remove(os.path.join(path, f))
    os.rmdir(path)
    os.makedirs(path)
    path = path + name_data_set_raw
    df = pd.read_csv(csv_path)
    df.to_csv(path, index=False)


def rellenarValoresNoTomados(df, var2):
    for i in df.columns:
        if i != var2:
            x = 0
            for j in df.index:
                if j == df.index[0]:
                    if math.isnan(df[i][j]):
                        if not math.isnan(df[i][df.index[x + 1]]):
                            df[i][j] = df[i][df.index[x + 1]]
                    elif df[i][j] == -999:
                        if not df[i][df.index[x + 1]] != -999:
                            df[i][j] = df[i][df.index[x + 1]]
                elif j == df.index[len(df.index)-1]:
                    if math.isnan(df[i][j]):
                        if not math.isnan(df[i][df.index[x - 1]]):
                            df[i][j] = df[i][df.index[x - 1]]
                    elif df[i][j] == -999:
                        if not df[i][df.index[x - 1]] != -999:
                            df[i][j] = df[i][df.index[x - 1]]
                elif math.isnan(df[i][j]):
                    if not math.isnan(df[i][df.index[x - 1]]):
                        df[i][j] = df[i][df.index[x - 1]]
                    if not math.isnan(df[i][df.index[x + 1]]):
                        df[i][j] = df[i][df.index[x + 1]]
                elif df[i][j] == -999:
                    if not df[i][df.index[x - 1]] != -999:
                        df[i][j] = df[i][df.index[x - 1]]
                    if not df[i][df.index[x + 1]] != -999:
                        df[i][j] = df[i][df.index[x + 1]]
                x = x + 1
    return df

def rellenarValoresNoTomadosV(df, var):
    for i in df.columns:
        if i == var:
            x = 0
            for j in df.index:
                if j == df.index[0]:
                    if math.isnan(df[i][j]):
                        if not math.isnan(df[i][df.index[x + 1]]):
                            df[i][j] = df[i][df.index[x + 1]]
                    elif df[i][j] == -999:
                        if not df[i][df.index[x + 1]] != -999:
                            df[i][j] = df[i][df.index[x + 1]]
                elif j == df.index[len(df.index)-1]:
                    if math.isnan(df[i][j]):
                        if not math.isnan(df[i][df.index[x - 1]]):
                            df[i][j] = df[i][df.index[x - 1]]
                    elif df[i][j] == -999:
                        if not df[i][df.index[x - 1]] != -999:
                            df[i][j] = df[i][df.index[x - 1]]
                elif math.isnan(df[i][j]):
                    if not math.isnan(df[i][df.index[x - 1]]):
                        df[i][j] = df[i][df.index[x - 1]]
                    if not math.isnan(df[i][df.index[x + 1]]):
                        df[i][j] = df[i][df.index[x + 1]]
                elif df[i][j] == -999:
                    if not df[i][df.index[x - 1]] != -999:
                        df[i][j] = df[i][df.index[x - 1]]
                    if not df[i][df.index[x + 1]] != -999:
                        df[i][j] = df[i][df.index[x + 1]]
                x = x + 1
    return df


def procesoSustituirPorlaMedia(df, var2, date_time):
    for i in df.columns:
        if i != var2:
            for j in df[i].index:
                if numpy.isnan(df[i][j]):
                    mean = calculate_mean(df, i, j, date_time)
                    df[i][j] = mean
                if df[i][j] == -999:
                    mean = calculate_mean(df, i, j, date_time)
                    df[i][j] = mean
    return df


def misma_fecha_menos_anyo(actual_date, date_time):
    value = actual_date.day == date_time.day
    value1 = actual_date.month == date_time.month
    value2 = actual_date.hour == date_time.hour
    value3 = actual_date.minute == date_time.minute
    return value and value1 and value2 and value3


def calculate_mean(df, i, x, date_time):
    actual_date = date_time[x]
    contador = 0
    mean = 0
    for j in df[i].index:
        if x != j:
            d = date_time[j]
            value = misma_fecha_menos_anyo(actual_date, d)
            if value:
                if not numpy.isnan(df[i][j]):
                    if not df[i][j] == -999:
                        mean += df[i][j]
                        contador += 1
    if contador == 0:
        return mean
    return mean/contador


def procesoDelaDiferencia(df, var2):
    for i in df.columns:
        if i != var2:
            x = 0
            for j in df.index:
                if j == df.index[0]:
                    if math.isnan(df[i][j]):
                        if not math.isnan(df[i][df.index[x + 1]]):
                            df[i][j] = df[i][df.index[x + 1]]
                    elif df[i][j] == -999:
                        if not df[i][df.index[x + 1]] != -999:
                            df[i][j] = df[i][df.index[x + 1]]
                elif j == df.index[len(df.index)-1]:
                    if math.isnan(df[i][j]):
                        if not math.isnan(df[i][df.index[x - 1]]):
                            df[i][j] = df[i][df.index[x - 1]]
                    elif df[i][j] == -999:
                        if not df[i][df.index[x - 1]] != -999:
                            df[i][j] = df[i][df.index[x - 1]]
                elif math.isnan(df[i][j]):
                    if not math.isnan(df[i][df.index[x - 1]]) and not math.isnan(df[i][df.index[x + 1]]):
                        df[i][j] = (df[i][df.index[x + 1]]+df[i][df.index[x - 1]])/2
                elif df[i][j] == -999:
                    if df[i][df.index[x + 1]] != -999 and df[i][df.index[x - 1]] != -999:
                        df[i][j] = (df[i][df.index[x + 1]]+df[i][df.index[x - 1]])/2
                x = x + 1
    return df

def procesoDelaDiferenciaV(df, var):
    for i in df.columns:
        if i == var:
            x = 0
            for j in df.index:
                if j == df.index[0]:
                    if math.isnan(df[i][j]):
                        if not math.isnan(df[i][df.index[x + 1]]):
                            df[i][j] = df[i][df.index[x + 1]]
                    elif df[i][j] == -999:
                        if not df[i][df.index[x + 1]] != -999:
                            df[i][j] = df[i][df.index[x + 1]]
                elif j == df.index[len(df.index)-1]:
                    if math.isnan(df[i][j]):
                        if not math.isnan(df[i][df.index[x - 1]]):
                            df[i][j] = df[i][df.index[x - 1]]
                    elif df[i][j] == -999:
                        if not df[i][df.index[x - 1]] != -999:
                            df[i][j] = df[i][df.index[x - 1]]
                elif math.isnan(df[i][j]):
                    if not math.isnan(df[i][df.index[x - 1]]) and not math.isnan(df[i][df.index[x + 1]]):
                        df[i][j] = (df[i][df.index[x + 1]]+df[i][df.index[x - 1]])/2
                elif df[i][j] == -999:
                    if df[i][df.index[x + 1]] != -999 and df[i][df.index[x - 1]] != -999:
                        df[i][j] = (df[i][df.index[x + 1]]+df[i][df.index[x - 1]])/2
                x = x + 1
    return df


def procesoDeSustitucionPorMinimo(df, var2):
    for i in df.columns:
        if i != var2:
            for j in df.index:
                if numpy.isnan(df[i][j]):
                    df[i][j] = 0
                if df[i][j] == -999:
                    df[i][j] = 0
    return df

def procesoDeSustitucionPorMinimoV(df, var):
    for i in df.columns:
        if i == var:
            for j in df.index:
                if numpy.isnan(df[i][j]):
                    df[i][j] = 0
                if df[i][j] == -999:
                    df[i][j] = 0
    return df


def depurar(var2, df, tmp1, tmp2):
    date_time = getTimeSteps(df, var2)
    df = procesoSustituirPorlaMedia(df, var2, date_time)
    df = procesoDelaDiferencia(df, var2)
    df = rellenarValoresNoTomados(df, var2)
    df = procesoDeSustitucionPorMinimo(df, var2)

    df, date_time = getOnlyPerHour(df, date_time, tmp1, tmp2)
    return date_time, df


def show_histograms(df, vSolar, path):
    list_attributes = df.columns
    for attribute in list_attributes:
        print(attribute)
        plot_hist_data_attribute(attribute, df, vSolar, path)

def showTimeEvolutionVar(df, date_time, value, path, typeofEv):
    plot_cols = [value]
    plot_features = df[plot_cols]
    plot_features.index = date_time
    _ = plot_features.plot(subplots=True)
    plt.savefig(path + "images/" + value +"Ev_"+typeofEv+".png", bbox_inches='tight')
    plt.close()

def showTimeEvolution(df, date_time, path, typeofEv, var=-1):
    if var==-1:
        for value in df.columns:
            showTimeEvolutionVar(df, date_time, value, path, typeofEv)
    else:
        showTimeEvolutionVar(df, date_time, var, path, typeofEv)


def showStation(dfAux, time_variable, estacion, path, typeofEv, var=-1):
    dfEstacion = dfAux[dfAux[time_variable].apply(regex_filter, regex=(estacion))]
    date_time = getTimeSteps(dfEstacion, time_variable)
    dfEstacion = dfEstacion.drop(time_variable, axis=1)
    showTimeEvolution(dfEstacion, date_time, path, typeofEv, var)


def sacarEstadisticas(df, climateZone):
    estadisticas = df.describe().transpose()
    estadisticas = pd.DataFrame(estadisticas)
    estadisticas.to_csv(climateZone+"estadisticas.csv")
    return df.describe().transpose()


def calcularMatrizCorrelacion(df, path):
    mat = df.corr()
    sns.heatmap(mat, annot=False)
    plt.savefig(path + "images/" + 'matriz.png', bbox_inches='tight')
    plt.close()
    return mat


def eliminarSegunMatrizCorrelacion(df, y_label, mat, umbralInf, umbralSup, path):
    list_vars = []
    for attr in df.columns:
        if attr != 'JulianTime':
            data = mat[y_label][attr]
            if data > umbralInf and data < umbralSup:
                df = df.drop(attr, axis=1)
                list_vars.append(attr)
    file = open(path + 'varsEliminadas.txt', 'w')
    for item in list_vars:
        file.write(item + "\n")
    file.close()
    return df


def addTimeData(df, date_time):
    timestamp_s = date_time.map(pd.Timestamp.timestamp)
    day = 24 * 60 * 60
    year = (365.2425) * day

    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    return df

def normalizeDataYDivideIntoSets(df, path):
    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    df_std = (df - train_mean) / train_std
    df_std = df_std.melt(var_name='Column', value_name='Normalized')
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
    _ = ax.set_xticklabels(df.keys(), rotation=90)

    plt.savefig(path + "images/" + 'normalized.png', bbox_inches='tight')
    plt.close()
    return train_df, val_df, test_df

def saveData(df, path, name):
    df.to_csv(path + name, index=False)

def saveSets(train_df, val_df, test_df, path):
    train_df.to_csv(path + "train.csv", index=False)
    val_df.to_csv(path + "val.csv", index=False)
    test_df.to_csv(path + "test.csv", index=False)


def preprocessData(path, df, time_variable, y_label, tmp1, tmp2, vSolar, umbralInf, umbralSup):
    date_time, df = depurar(time_variable, df, tmp1, tmp2)

    dfAux = df
    df = df.drop(time_variable, axis=1)

    show_histograms(df, vSolar, path)
    showTimeEvolution(df, date_time, path, "year")
    print(dfAux)
    showStation(dfAux, time_variable, '-09-|-10-|-11-', path, "autumn")
    showStation(dfAux, time_variable, '-01-|-02-|-12-', path, "winter")
    showStation(dfAux, time_variable, '-03-|-04-|-05-', path, "spring")
    showStation(dfAux, time_variable, '-06-|-07-|-08-', path, "summer")

    print(sacarEstadisticas(df, path))

    mat = calcularMatrizCorrelacion(df, path)

    df = eliminarSegunMatrizCorrelacion(df, y_label, mat, umbralInf, umbralSup, path)

    df = addTimeData(df, date_time)

    train_df, val_df, test_df = normalizeDataYDivideIntoSets(df, path)

    saveSets(train_df, val_df, test_df, path)

def LSTM_model(train_X, train_y, val_X, val_y, path, epochs, batch_size, patience, neurons):
    print(epochs)
    print(batch_size)
    print(neurons)
    print(patience)
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
    history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping],
                        validation_data=(val_X, val_y), verbose=2,
                        shuffle=False)
    model.save(path + "modelU.h5")
    return history

def splitIntoSetsXY(df, y_label):
    i = df.columns.get_loc(y_label)
    # split into input and outputs
    df = df.values
    df_X, df_y = df[:, [z for z in range(df[0].size) if not z in [i]]], df[:, i]
    # reshape input to be 3D [samples, timesteps, features]
    df_X = df_X.reshape((df_X.shape[0], 1, df_X.shape[1]))
    return df_X, df_y


def supervised(train_df, val_df, path, ylabel, epochs, batch_size, patience, neurons):
    train_X, train_y, = splitIntoSetsXY(train_df, ylabel)
    val_X, val_y = splitIntoSetsXY(val_df, ylabel)
    history = LSTM_model(train_X, train_y, val_X, val_y, path, epochs, batch_size, patience, neurons)
    # plot history
    if history:
        try:
            show_history(history)
        except:
            print("Error")
    else:
        print("Error fallo a la hora de realizar el entrenamiento")
    plt.savefig(path + "images/" + 'history_supervised.png', bbox_inches='tight')
    plt.close()

def createLSTM_model(OUT_STEPS, num_features, neurons):
    multi_lstm_model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, lstm_units].
        # Adding more `lstm_units` just overfits more quickly.
        tf.keras.layers.LSTM(neurons, return_sequences=False),
        # Shape => [batch, out_steps*features].
        tf.keras.layers.Dense(OUT_STEPS * num_features,
                              kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features].
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])
    return multi_lstm_model

def compareMethods(path, multi_performance, multi_lstm_model, multi_val_performance):
    x = np.arange(len(multi_performance))
    width = 0.3

    metric_index = multi_lstm_model.metrics_names.index('mean_absolute_error')
    val_mae = [v[metric_index] for v in multi_val_performance.values()]
    test_mae = [v[metric_index] for v in multi_performance.values()]

    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(ticks=x, labels=multi_performance.keys(),
               rotation=45)
    plt.ylabel(f'MAE (average over all times and outputs)')
    _ = plt.legend()
    plt.savefig(path + "images/" + 'mae_comparison.png', bbox_inches='tight')
    plt.close()


def createMultiStepModel(path, multi_window, multi_val_performance, multi_performance, y_label):
    last_baseline = MultiStepLastBaseline()
    last_baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                          metrics=[tf.keras.metrics.MeanAbsoluteError()])

    multi_val_performance['Last'] = last_baseline.evaluate(multi_window.val)
    multi_performance['Last'] = last_baseline.evaluate(multi_window.test, verbose=0)
    multi_window.plot(last_baseline, plot_col=y_label)
    plt.savefig(path + "images/" + 'MultiStepModel.png', bbox_inches='tight')
    plt.close()

    return multi_performance, multi_val_performance


def createRepeatModel(path, multi_window, multi_val_performance, multi_performance, y_label):
    repeat_baseline = RepeatBaseline()
    repeat_baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                            metrics=[tf.keras.metrics.MeanAbsoluteError()])
    multi_val_performance['Repeat'] = repeat_baseline.evaluate(multi_window.val)
    multi_performance['Repeat'] = repeat_baseline.evaluate(multi_window.test, verbose=0)
    multi_window.plot(repeat_baseline, plot_col=y_label)
    plt.savefig(path + "images/" + 'RepeatModel.png', bbox_inches='tight')
    plt.close()

    return multi_performance, multi_val_performance


def unsupervised(train_df, val_df, test_df, path, neurons, y_label):
    num_features = train_df.shape[1]
    OUT_STEPS = 24*7*2
    multi_window = WindowGenerator(input_width=24*7*2,
                                   label_width=OUT_STEPS,
                                   shift=OUT_STEPS, train_df=train_df, val_df=val_df, test_df=test_df)

    multi_val_performance = {}
    multi_performance = {}

    multi_performance, multi_val_performance = createMultiStepModel(path, multi_window, multi_val_performance, multi_performance, y_label)

    multi_performance, multi_val_performance = createRepeatModel(path, multi_window, multi_val_performance,
                                                                    multi_performance, y_label)
    multi_lstm_model = createLSTM_model(OUT_STEPS, num_features, neurons)

    history = multi_window.compile_and_fit(multi_lstm_model, multi_window, path)

    IPython.display.clear_output()

    multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
    multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
    multi_window.plot(multi_lstm_model, plot_col=y_label)

    plt.savefig(path + "images/" + 'LSTM.png', bbox_inches='tight')
    plt.close()

    compareMethods(path, multi_performance, multi_lstm_model, multi_val_performance)


def data_train(train_df, val_df, test_df, path, y_label, epochs=100, batch_size=2000, patience=1, neurons=128, neurons1=32):
    unsupervised(train_df, val_df, test_df, path, neurons1, y_label)
    supervised(train_df, val_df, path, y_label, epochs, batch_size, patience, neurons)

def prepareForPrediction(test_df, y_label):
    print(test_df.columns)
    i = test_df.columns.get_loc(y_label)
    test_mean, test_std = getMeanAndStd(test_df)
    test_df = (test_df - test_mean) / test_std
    test_X, test_y = splitIntoSetsXY(test_df, y_label)

    return test_X, test_y, test_mean, test_std


def predictAndSave(model, path, test_X, test_y, test_mean, test_std, y_label):
    pred_y = []
    result = []
    for x in range(0, len(test_X), 2000):
        batch = test_X[x:x + 2000]
        result.append(model.predict(batch))
    for batch in result:
        for z in batch:
            pred_y.append(z[0])
    y_true, predictions = np.array(test_y), np.array(pred_y)
    mae = np.mean(np.abs(y_true - predictions))
    mse = np.square(np.subtract(y_true, predictions)).mean()
    rmse = np.sqrt(np.square(np.mean((y_true - predictions))))
    try:
        rmlse = np.sqrt(np.mean(np.square(np.log(y_true + 1) - np.log(predictions + 1))))
    except Exception as e:
        print(e)
        rmlse = 0
    r2 = r2_score(y_true, predictions)
    file = open(path + 'metricasEvaluacion.txt', 'w')
    file.write("MAE: " + str(mae) + "\n")
    file.write("MSE: " + str(mse) + "\n")
    file.write("RMSE: " + str(rmse) + "\n")
    file.write("RMLSE: " + str(rmlse) + "\n")
    file.write("R2: " + str(r2) + "\n")
    file.close()
    dfpred_y = pd.DataFrame(pred_y)
    dfpred_y = (dfpred_y * test_std[y_label]) + test_mean[y_label]
    dfpred_y[dfpred_y < 0] = 0
    test_y = (test_y * test_std[y_label]) + test_mean[y_label]
    predicciones_reales = pd.DataFrame(test_y)
    dfpred_y.to_csv(path + "predicciones_solares.csv", index=False)
    predicciones_reales.to_csv(path + "reales_solares.csv", index=False)
    return dfpred_y


def calcularProduccion(dfpred_y, area, coef1, coef2, path):
    result_generacion = []
    list_predictions = dfpred_y[0][:]
    for value in list_predictions:
        result_generacion.append(area * coef1 * value * coef2)
    resultados = pd.DataFrame(result_generacion)
    resultados.to_csv(path + "predicciones_electricidad"+str(area)+"_"+str(coef1)+"_"+str(coef2)+".csv", index=False)


def predict_supervised(test_df, model, path, time_variable, y_label, tmp1, tmp2, area, ratio, coef, umbralInf, umbralSup):
    date_time, test_df = depurar(time_variable, test_df, tmp1, tmp2)
    test_df = test_df.drop(time_variable, axis=1)
    test_df = eliminarVariablesSegunVarianza(path, test_df)
    test_df = addTimeData(test_df, date_time)
    test_X, test_y, test_mean, test_std = prepareForPrediction(test_df, y_label)
    dfpred_y = predictAndSave(model, path, test_X, test_y, test_mean, test_std, y_label)
    calcularProduccion(dfpred_y, area, ratio, coef, path)


def getResultsAnual(test_df, model, path, test_mean, test_std, y_label):
    i = test_df.columns.get_loc(y_label)
    data = test_df[len(test_df) - (24 * 7 * 2) - 1:len(test_df) - 1]
    results = []
    resultsSolar = []
    example_window = tf.stack([np.array(data)])
    predictions = model(example_window)
    for pred in predictions[0][:]:
        results.append(pred[i].numpy())# * test_std[y_label] + test_mean[y_label])
        resultsSolar.append(pred[i].numpy() * test_std[y_label] + test_mean[y_label])
    for pred in range(0, 26):
        predictions = model(predictions)
        for pred in predictions[0][:]:
            results.append(pred[i].numpy())# * test_std[y_label] + test_mean[y_label])
            resultsSolar.append(pred[i].numpy() * test_std[y_label] + test_mean[y_label])
    predictions, y_true = np.array(results), np.array((test_df[y_label][:9072]-test_mean[y_label])/test_std[y_label])
    mae = np.mean(np.abs(y_true - predictions))
    mse = np.square(np.subtract(y_true, predictions)).mean()
    rmse = np.sqrt(np.square(np.mean((y_true - predictions))))
    try:
        rmlse = np.sqrt(np.mean(np.square(np.log(y_true+1) - np.log(predictions+1))))
    except Exception as e:
        print(e)
        rmlse = 0
    r2 = r2_score(y_true, predictions)
    file = open(path + 'metricasEvaluacion.txt', 'w')
    file.write("MAE: " + str(mae) + "\n")
    file.write("MSE: " + str(mse) + "\n")
    file.write("RMSE: " + str(rmse) + "\n")
    file.write("RMLSE: " + str(rmlse) + "\n")
    file.write("R2: " + str(r2) + "\n")
    file.close()
    resultsCSV = pd.DataFrame(resultsSolar)
    resultsCSV.to_csv(path + "predicciones_solares.csv", index=False)
    return resultsCSV


def predict_unsupervised(test_df, model, path, time_variable, y_label, tmp1, tmp2, area, ratio, coef, umbralInf, umbralSup):
    date_time, test_df = depurar(time_variable, test_df, tmp1, tmp2)
    test_df = test_df.drop(time_variable, axis=1)
    test_df = eliminarVariablesSegunVarianza(path, test_df)
    test_df = addTimeData(test_df, date_time)
    test_mean, test_std = getMeanAndStd(test_df)
    test_df = (test_df - test_mean) / test_std
    resultsCSV = getResultsAnual(test_df, model, path, test_mean, test_std, y_label)
    calcularProduccion(resultsCSV, area, ratio, coef, path)


def predict(test_df, model, path, time_variable, y_label, my_training, tmp1, tmp2, area, ratio, coef, umbralInf, umbralSup):
    if my_training:
        predict_supervised(test_df, model, path, time_variable, y_label, tmp1, tmp2, area, ratio, coef, umbralInf, umbralSup)
    else:
        predict_unsupervised(test_df, model, path, time_variable, y_label, tmp1, tmp2, area, ratio, coef, umbralInf, umbralSup)

def getMeanAndStd(df):
    return df.mean(), df.std()

def getMeanAndStdVar(path, var):
    df = pd.read_csv(path)
    print("No esta bien")
    print(df)
    contador = 0
    first_row = df.columns[0]
    for i in df.index:
        if df[first_row][i] == var:
            break
        contador +=1
    return df['mean'][df.index[contador]], df['std'][df.index[contador]]

def eliminarVariablesSegunVarianza(path, test_df):
    with open(path + 'varsEliminadas.txt', 'r') as file:
        lines = file.readlines()
        for l in lines:
            l = l[:len(l)-1]
            test_df = eliminarVariable(l, test_df)
    return test_df

def detect_real_time_errors(path, test_df, model, time_variable, y_label, area, ratio, coef):
    dfOriginal = test_df
    test_df = eliminarVariablesSegunVarianza(path, test_df)
    date_time = getTimeSteps(test_df, time_variable)
    test_df = test_df.drop(time_variable, axis=1)
    test_df = addTimeData(test_df, date_time)
    test_mean, test_std = getMeanAndStdVar(path+"estadisticas.csv", y_label)
    test_df = (test_df - test_mean) / test_std

    print(test_df.columns)
    # split into input and outputs
    test_X, test_y = splitIntoSetsXY(test_df, y_label)

    pred_y = model.predict(test_X)
    dfpred_y = pd.DataFrame(pred_y)

    production = area * coef * ((dfpred_y[0][0] * test_std) + test_mean) * ratio
    production_actual = area * coef * dfOriginal[y_label].values[0] * ratio
    print(production)
    print(production_actual)
    print(dfOriginal[y_label].values[0])
    print(dfpred_y[0][0]* test_std + test_mean)
    deteccion = abs(production_actual/production)
    if (deteccion==1):
        print("El sistema funciona con optimamente")
    else:
        if (deteccion < 1 and deteccion >= 0.8):
            print("Se necesita mantenimiento en el panel. Falla detectada del 20%")
        elif (deteccion >= 0.7 and deteccion < 0.8):
            print("Valor de producción detectado como anomalo. Revise el estado del cielo y reporte incidencia")
        elif (deteccion >= 0.5 and deteccion < 0.7):
            print("Posible averia del panel revise su estado.")
        elif (deteccion < 0.5):
            print("Revise la instalacion")

def getTimeSteps(df, time):
    try:
        date_time = pd.to_datetime(df[time], format='%d-%m-%Y %H:%M:%S')
    except:
        date_time = pd.to_datetime(df[time], format='%Y-%m-%d %H:%M:%S')
    return date_time

def getOnlyPerHour(df, date_time, tmp1, tmp2):
    df = df[tmp1::tmp2]
    date_time = date_time[tmp1::tmp2]
    return df, date_time

def estudio(path, var1, var2, df, tmp1, tmp2, vSolar):
    dfOriginal = df
    sacarEstadisticas(df[var1], path)
    plot_hist_data_attribute(var1, df, vSolar, path)
    date_time = getTimeSteps(df, var2)
    df, date_time = getOnlyPerHour(df, date_time, tmp1, tmp2)
    showTimeEvolutionVar(df, date_time, var1, path, "year")
    showStation(dfOriginal, var2, '-09-|-10-|-11-', path, "autumn", var1)
    showStation(dfOriginal, var2, '-01-|-02-|-12-', path, "winter", var1)
    showStation(dfOriginal, var2, '-03-|-04-|-05-', path, "spring", var1)
    showStation(dfOriginal, var2, '-06-|-07-|-08-', path, "summer", var1)


def calculate_meanV(df, i, x, date_time, inf, sup):
    actual_date = date_time[x]
    contador = 0
    mean = 0
    for j in df[i].index:
        if x != j:
            if misma_fecha_menos_anyo(actual_date, date_time[j]):
                if not numpy.isnan(df[i][j]):
                    if inf <= df[i][j] <= sup:
                        mean += df[i][j]
                        contador+=1
    if contador == 0:
        return mean
    return mean/contador

def depurarV(var, df, inf, sup, date_time):
    for j in df[var].index:
        if df[var][j] < inf:
            mean = calculate_meanV(df, var, j, date_time, inf, sup)
            df[var][j] = mean
        elif df[var][j] > sup:
            mean = calculate_meanV(df, var, j, date_time, inf, sup)
            df[var][j] = mean
    return df

def estudiarVariable(path, var1, var2, df, tmp1, tmp2, vSolar):
    estudio(path, var1, var2, df, tmp1, tmp2, vSolar)

def depurarVariable(var, inf, sup, df, time_variable):
    date_time = getTimeSteps(df, time_variable)
    df = depurarV(var, df, inf, sup, date_time)
    df = procesoDelaDiferenciaV(df, var)
    df = rellenarValoresNoTomadosV(df, var)
    df = procesoDeSustitucionPorMinimoV(df, var)
    return df

def eliminarVariable(var, df):
    df = df.drop(var, axis=1)
    return df
