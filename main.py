from custom_model import CL_CustomModel
import torch
from train import iniciarEntrenamiento
from data_loader import baseDataset




LEARNING_RATE = 0.1
EPOCHS = 30
batch_size = 1





def obtenerFuncionDeCosteYOptimizador(model):
    """
    Funcion que define y devuelve la funcion de coste y el optimizador
    """
    # Definimos la funcion de coste (la que calcula el error)
    loss_fn = torch.nn.MSELoss(reduction='sum')

    # Cambiamos el loss a BCEWithLogitsLoss para ponderar los casos
    # La ponderacion va dentro de un tensor con con un numero por cada clase en la red, indicando su ponderacion
    #loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1, 1]))

    # Definimos el optimizador que se encargara de hacer el descenso del gradiente
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Probamos el BCEWithLogitsLoss para reducir los falsos positivos
    #pesos = torch.Tensor([1,1]) # pesos de cada clase. la idea es penalizar mas la clase que queremos que tenga menos FP
    #optimizer = torch.optim.BCEWithLogitsLoss(pos_weight=pesos)

    return loss_fn, optimizer










print("-----  inicio del test -----")

device = ("cuda" if torch.cuda.is_available() else "cpu")


net_layers = [
    {"layer_type": "Conv2d", "in_channels": 3, "out_channels": 20, "kernel_size": 3},
    {"layer_type": "MaxPool2d", "kernel_size": (2,2)},
    {"layer_type": "Conv2d", "in_channels": 20, "out_channels": 40, "kernel_size": 3},
    {"layer_type": "MaxPool2d", "kernel_size": (2,2)},
    {"layer_type": "Conv2d", "in_channels": 40, "out_channels": 100, "kernel_size": 3},
    {"layer_type": "MaxPool2d", "kernel_size": (2,2)},
    {"layer_type": "Conv2d", "in_channels": 100, "out_channels": 200, "kernel_size": 3},
    {"layer_type": "MaxPool2d", "kernel_size": (2,2)},

    {"layer_type": "Flatten"},

    {"layer_type": "Linear", "in_features": 200*16*23, "out_features": 50},
    {"layer_type": "Sigmoid"},
    {"layer_type": "Linear", "in_features": 50, "out_features": 10},
    {"layer_type": "Sigmoid"},
    {"layer_type": "Linear", "in_features": 10, "out_features": 2}
]
print("creacion de net_layer_list")

CM = CL_CustomModel(net_layers, device)
print("creacion de la instancia de la clase customModel")

print("\n")
print(CM)

num_params = sum(p.numel() for p in CM.parameters())
print("Parametros: ", num_params)


loss_fn, optimizer = obtenerFuncionDeCosteYOptimizador(CM)




# Dataset loader de Pilar

dataset = baseDataset('dataset/SinRuido.json')
trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)




iniciarEntrenamiento(CM, loss_fn, optimizer, trainloader, trainloader , epochs=EPOCHS)


print("-----  fin del test -----")