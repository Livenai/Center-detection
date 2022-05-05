from pandas import array
import enlighten
import numpy as np
from colored import fg
from math import isnan, isinf
from copy import deepcopy
from data_loader import width, height

import torch
device = ("cuda" if torch.cuda.is_available() else "cpu")


C = fg(10)
B = fg(15)

loss_hist = np.array([]) # Uso temporal


# Aux para el early stopping
best_test_loss = 999999999
current_best_model = None

bad_epoch_threshold = 0.6
max_bad_epochs = 3





def __acumAndGetMeanLoss(new_loss_number):
    """
    Funcion que acumula el error y devuelve la media y desviacion
    tipica de todos los errores acumulados
    """
    global loss_hist
    loss_hist = np.append(loss_hist, new_loss_number)

    return float(np.mean(loss_hist)), float(np.std(loss_hist))




def checkMeanDist(loader, model):
    """ Funcion para evaluar el modelo con los datos que ofrezca el loader """

    model.eval()
    dist_list = []

    with torch.no_grad():
        for imgs, labels in loader:

            imgs = imgs.to(device=device).float()
            labels = labels.to(device=device).float()

            preds = model(imgs)
            
            for p, y in list(zip(preds, labels)):
                den_p = ((p.cpu().numpy() +1) /2 ) * np.array([width, height])
                den_y = ((y.cpu().numpy() +1) /2 ) * np.array([width, height])

                dist = np.linalg.norm(den_y - den_p)

                dist_list.append(dist)
            

    model.train()

    # Calculamos la media y devolvemos
    return np.mean(dist_list)


    

def iniciarEntrenamiento(model, loss_fn, optimizer, train_loader, validation_loader, epochs=2):
    """
    Funcion privada que inicia el entrenamiento.

    La funcion es bloqueante y realiza el entrenamiento completo del
    modelo a traves de todas las epocas.
    Una vez el entrenamiento termina, se almacenan las metricas finales en
    la variable de clase history.
    """
    global best_test_loss, current_best_model, bad_epoch_threshold, max_bad_epochs

    # Todo lo que envueleve a donex es para poder ver los datos del dataset.
    # Poniendolo a True se quita dicha funcionalidad
    donex = True

    # Historial de metricas
    history = {
                "loss": [],
                "val_loss": [],
                "accuracy": [],
                "val_accuracy": []
                }

    # Ponemos el modelo en modo entrenamiento
    model.train()

    bar_manager = enlighten.get_manager()
    epochs_bar = bar_manager.counter(total=epochs, desc="Epochs:  ", unit='Epochs', position=2, leave=True, color=(150,255,0))


    # Entrenamos!!            ======================  loop  =======================
    for epoch in range(epochs):

        ent_loss_list = []
        num_correct = 1
        num_samples = 1

        train_bar = bar_manager.counter(total=len(train_loader), desc="Training:  ", unit='img', position=1, leave=False, color=(50,150,0))
        for imgs, labels in train_loader:
            # Preparamso las imagenes
            imgs = imgs.to(device)
            labels = labels.to(device)
            if not donex:
                print(C + "--------------------------- Datos de los Tensores del Dataset ---------------------------\n\n")
                #print(torch.unique(imgs))
                #print(imgs)
                print("dimensiones:  ", imgs.size())
                print("dtype:  ", imgs.dtype , "\n\n")

                print("Label:    ", labels, "     dimensiones:  ", labels.size(), "    dtype:  ", labels.dtype , "\n\n" + B)
                donex = True

            # Sacamos las predicciones
            outputs = model(imgs)



            # Obtenemos el error
            loss = loss_fn(outputs, labels)
            ent_loss_list.append(loss.item())

            # Back-propagation y entrenamiento
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Tick de la barra de entrenamiento
            loss_mean, loss_std = __acumAndGetMeanLoss(loss.item())
            train_bar.desc = "Trainig:  loss= " + str(round(loss_mean,4)) + "  std_dev= " + str(round(loss_std,2)) + " "
            train_bar.update()


        # Guardamos las metricas de la epoca
        train_loss = np.mean(ent_loss_list)


        # Comprobamos si hay NaN o Inf en las metricas
        if isnan(train_loss) or isinf(train_loss):
            # Guardamos las metricas y paramos de entrenar, pues seria inutil continuar
            bar_manager.remove(train_bar)
            print("La red contiene NaN")
            break


        # Borramos la barra de entrenamiento
        bar_manager.remove(train_bar)


        # Tick de la barra de epocas
        prefix_epochs_bar = "Epochs:  dist= "+str()
        epochs_bar.desc = prefix_epochs_bar
        epochs_bar.update()


        # Early stop. Guardamos la mejor epoca hasta ahora
        # Si esta epoca es la mejor:
        if train_loss < best_test_loss:
            # Guardamos su modelo temporalmente
            current_best_model = deepcopy(model)

            # Actualizamos el mejor loss
            best_test_loss = train_loss


        # Sacamos la media de distancias de los puntos
        dist_mean = checkMeanDist(validation_loader, model)

        # Mostramos las metricas
        print("e " + str(epoch) + ":\t ", end="")
        print("  loss: " + C + str(train_loss) + B, end="")
        print("  dist_mean: " + C + str(dist_mean) + B)
        





    # Destruimos las barras
    bar_manager.remove(epochs_bar)

    # Reestablecemos el modelo al modelo de la mejor epoca
    model = current_best_model


    return history