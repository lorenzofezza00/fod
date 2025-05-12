import sys
import os
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

from utils.utils import load_Cityscapes_dataset, FaultyNetwork, load_faults, filter_faults

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './network')))
import torch
from network.models.fast_scnn import get_fast_scnn
from network.utils.visualize import get_color_pallete
from network.data_loader import datasets

use_gpu = True
model = get_fast_scnn('citys', aux=False, pretrained=True, root='./network/weights')
device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
faulty_model = FaultyNetwork(model)

faults = load_faults("./fault_injection/output/Fast-SCNN_38_fault_list_30_3.csv")
filtered_faults = filter_faults(faults, bit=30)

tr_loader, va_loader, te_loader = load_Cityscapes_dataset(root_dir='./cityscapes')

dataset = va_loader.dataset

df = pd.read_csv('./fod/faulty_output_dataset/faulty_output_dataset.csv')

def update_mask(indice, flag=True):
    frame_id = df.iloc[indice]['Frame']
    fault_id = df.iloc[indice]['Injection']
    lab = int(df.iloc[indice]['Label'])

    img, _ = dataset[frame_id]
    mask, _ = faulty_model.get_mask(img.unsqueeze(0).to(device), filtered_faults[fault_id])
    mask = get_color_pallete(mask)
    if not flag:
        plt.imshow(mask)
        plt.axis('off')
        plt.show()
    else:
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        axs[0].imshow(mask)
        axs[0].axis('off')
        colore = 'red' if lab == 1 else 'green'
        testo = 'C' if lab == 1 else 'NC'
        axs[1].imshow(np.ones((10, 10, 3), dtype=np.uint8) * 255)  # sfondo bianco
        axs[1].add_patch(plt.Rectangle((0, 0), 10, 10, color=colore))
        axs[1].text(5, 5, testo, color='white', fontsize=15, ha='center', va='center', weight='bold')
        axs[1].axis('off')
        plt.tight_layout()
        plt.show()

# slider = widgets.IntSlider(value=0, min=0, max=len(df)-1, step=1, description='Index:')
# widgets.interactive(update_mask, indice=slider)

def save_image(indice, name=""):
    frame_id = df.iloc[indice]['Frame']
    fault_id = df.iloc[indice]['Injection']
    img, _ = dataset[frame_id]
    mask = faulty_model.get_mask(img.unsqueeze(0).to(device), filtered_faults[fault_id])
    mask = get_color_pallete(mask)
    mask.save(f"./imgs_vids/{name}{indice}_fm_{frame_id}_{fault_id}.png")
    print(f"Immagine {indice} salvata")

def update_from_textbox(change):
    try:
        indice = int(indice_textbox.value)
        slider.value = indice
        update_mask(indice)
    except ValueError:
        print("Per favore, inserisci un indice valido.")

slider = widgets.IntSlider(value=0, min=0, max=len(df)-1, step=1, description='Index:')
indice_textbox = widgets.IntText(value=0, description="Indice:", style={'description_width': 'initial'})
save_button = widgets.Button(description="Salva Immagine")

indice_textbox.observe(update_from_textbox, names='value')
save_button.on_click(lambda x: save_image(slider.value))

# display(slider, indice_textbox, save_button)

# update_mask(slider.value)