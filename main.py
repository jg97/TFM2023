import demostrador
import pandas as pd
import datetime

def eliminarVariables(df):
    for colum in df.columns:
        if "_flag" in colum:
            df = demostrador.eliminarVariable(colum, df)
    return df

def eliminarVariablesDeUnaZona(df):
    for colum in df.columns:
        if "_flag" in colum or "Cell2_" in colum or "Cell3_" in colum:
            df = demostrador.eliminarVariable(colum, df)
    return df


def pedirDirectorios():
    csv_path = input("path to csv to load: ")
    climate_zone = input("Name of the Climate Zone: ")
    name_data_set_raw = input("Name of the data set loaded: ")
    return name_data_set_raw, csv_path, climate_zone

def listar_Zonas_All():
    demostrador.listar_Zonas_All()

def listar_Predicciones_Zona():
    climate_zone = input("path to csv to load: ")
    demostrador.listar_Predicciones_Zona(climate_zone)


def listar_Resultados_Zona():
    climate_zone = input("path to csv to load: ")
    demostrador.listar_Resultados_Zona(climate_zone)



def printOpciones():
    print("Acciones disponibles: ")
    print("1. Eliminar una variable")
    print("2. Realizar estudio estadistico de una variable")
    print("3. Eliminar valores espureos o no validos de una variable")


def estudiar_DataSet():
    path, df = demostrador.pedirDirectorioYData("Banglades", "solarData.csv")
    sigue = True
    while sigue:
        printOpciones()
        print(df.columns)
        instruccion = int(input("Select one option: "))
        if instruccion > 0 and instruccion < 4:
            if instruccion == 1:
                var1 = input("Select a variable: ")
                df = demostrador.eliminarVariable(var1, df)
            elif instruccion == 2:
                var1 = input("Select a variable: ")
                var2 = input("Select Time variable: ")
                print("Realizar estudio estadistico")
                demostrador.estudiarVariable(path, var1, var2, df, 5, 6,
                                             ['DHI_ThPyra2_Wm-2_avg','DNI_ThPyrh1_Wm-2_avg','GHI_ThPyra1_Wm-2_avg_ThPyra1_Wm-2_avg','GTI_RefCell1_Wm-2_avg_RefCell1_Wm-2_avg'])
            elif instruccion == 3:
                var1 = input("Select a variable: ")
                inf = int(input("Select inf value: "))
                sup = int(input("Select sup value: "))
                print("Realizar estudio estadistico")
                df = demostrador.depurarVariable(var1, inf, sup, df, "JulianTime")
            sigue = True
        else:
            sigue = False


def data_Train():
    train_df, val_df, test_df, path = demostrador.pedirDataSets("Banglades")
    demostrador.data_train(train_df, val_df, test_df, path, 'GTI_RefCell1_Wm-2_avg')


def add_Data_to_DataSet():
    climate_zone = input("path to csv to load: ")
    csv_path = input("path to csv to load: ")
    new_data = input("path to csv to load: ")
    demostrador.add_Data_to_DataSet(climate_zone, csv_path, new_data)


def predict():
    model, test_df, path = demostrador.pedirModeloyDatos("Banglades")
    test_df = eliminarVariablesDeUnaZona(test_df)
    basic_training = False
    demostrador.predict(test_df, model, path, 'JulianTime', 'GTI_RefCell1_Wm-2_avg', basic_training, 59, 60, 500, 0.9, 0.3, -0.3, 0.3)

def listar_Zonas_Prediccion_P():
    demostrador.listar_Zonas_Prediccion_P()


def listar_Zonas_Prediccion_S():
    demostrador.listar_Zonas_Prediccion_S()


def listar_Zonas_Modelos():
    demostrador.listar_Zonas_Modelos()


def detect_real_Time_errors():
    model, test_df, path = demostrador.pedirModeloyDatos("Banglades")
    test_df = eliminarVariablesDeUnaZona(test_df)
    demostrador.detect_real_Time_errors(path, test_df.iloc[59:60], model, 'Time', 'GTI_RefCell1_Wm-2_avg', 500, 0.9, 0.3)


def load_Data_set():
     name_data_set_raw, csv_path, climate_zone = pedirDirectorios()
     if(demostrador.cargarDataSet(climate_zone, name_data_set_raw, csv_path)):
         print("Data set: " + name_data_set_raw + " loaded")
     else:
         print("La zona climatologica ya existe. Si prosigue se perderá toda información")
         print("Aun así desea continuar. ")
         resp = input("Escriba si o no: ")
         if resp.lower() == "si":
             demostrador.forzarCargarDataSet(climate_zone, name_data_set_raw, csv_path)
             print("Data set: " + name_data_set_raw + " loaded")

def pre_process_data():
    path, df = demostrador.pedirDirectorioYData("Banglades", "solarData.csv")
    df = eliminarVariables(df)
    demostrador.preprocessData(path, df, 'JulianTime', 'GTI_RefCell1_Wm-2_avg', 59, 60,
                               ['DHI_ThPyra2_Wm-2_avg','DNI_ThPyrh1_Wm-2_avg','GHI_ThPyra1_Wm-2_avg','GTI_RefCell1_Wm-2_avg'], -0.3, 0.3)


def printOpciones2():
    print("Acciones disponibles: ")
    print("1. Listar zonas climatologicas y sus datos")
    print("2. Listar predicciones para una zona climatologica")
    print("3. Revisar resultados para una zona climatologica")
    print("4. Estudiar un data set")
    print("5. Realizar un entrenamiento")
    print("6. Añadir datos a un data set")
    print("7. Realizar una predicción")
    print("8. Listar zonas con una predicción productiva")
    print("9. Listar zonas con una predicción solar")
    print("10. Listar zonas con un modelo predictivo")
    print("11. Realizar pre procesamiento del data set")
    print("12. Chequear valores reales para detectar fallas en el panel")
    print("13. Cargar un data set nuevo")


def main():
    # df = pd.read_csv("solarBanglades.csv")
    # years = df.pop('#year')
    # months = df.pop('month')
    # days = df.pop('day')
    # hours = df.pop('hour')
    # minutes = df.pop('minute')
    # df
    # dateTimes = []
    # for i in df.index:
    #     b = dateTime.dateTime(years[i], months[i], days[i], hours[i], minutes[i], 0, 0)
    #     dateTimes.append(b)
    # df["Time"] = dateTimes
    # df.to_csv("solarBanglades.csv", index=False)
    # exit(1)
    printOpciones2()
    option = int(input("Introduce the option yo wanna take: "))
    if option == 1:
        listar_Zonas_All()
    elif option == 2:
        listar_Predicciones_Zona()
    elif option == 3:
        listar_Resultados_Zona()
    elif option == 4:
        estudiar_DataSet()
    elif option == 5:
        data_Train()
    elif option == 6:
        add_Data_to_DataSet()
    elif option == 7:
        predict()
    elif option == 8:
        listar_Zonas_Prediccion_P()
    elif option == 9:
        listar_Zonas_Prediccion_S()
    elif option == 10:
        listar_Zonas_Modelos()
    elif option == 11:
        pre_process_data()
    elif option == 12:
        detect_real_Time_errors()
    elif option == 13:
        load_Data_set()

main()