import sys
import os
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.utils import load_Cityscapes_dataset, FaultyNetwork, CleanNetwork, load_faults, filter_faults
import torch
from network.models.fast_scnn import get_fast_scnn
from network.utils.visualize import get_color_pallete
from network.data_loader import datasets

co = True

if co:
    use_gpu = True
    model = get_fast_scnn('citys', aux=False, pretrained=True, root='../network/weights')
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    clean_model = CleanNetwork(model)
    tr_loader, va_loader, te_loader = load_Cityscapes_dataset(root_dir='../cityscapes')
    dataset = va_loader.dataset
    def save_image(indice, name=""):
        img, _ = dataset[indice]
        mask, _ = clean_model.get_mask(img.unsqueeze(0).to(device))
        mask = get_color_pallete(mask)    
        mask.save(f"./faulty_output_dataset/co/cm_{indice}.png")
        print(f"Immagine {indice} salvata")

    os.makedirs('./faulty_output_dataset', exist_ok=True)
    os.makedirs('./faulty_output_dataset/co', exist_ok=True)
    for i in tqdm(range(len(dataset))):
        save_image(i)
else:
    use_gpu = True
    model = get_fast_scnn('citys', aux=False, pretrained=True, root='../network/weights')
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    faulty_model = FaultyNetwork(model)

    faults = load_faults("../fault_injection/output/Fast-SCNN_38_fault_list_30_3.csv")
    filtered_faults = filter_faults(faults, bit=30)

    tr_loader, va_loader, te_loader = load_Cityscapes_dataset(root_dir='../cityscapes')

    dataset = va_loader.dataset

    df = pd.read_csv('./faulty_output_dataset.csv')

    def save_image(indice, name=""):
        frame_id = df.iloc[indice]['Frame']
        fault_id = df.iloc[indice]['Injection']
        label = df.iloc[indice]['Label']
        img, _ = dataset[frame_id]
        mask, _ = faulty_model.get_mask(img.unsqueeze(0).to(device), filtered_faults[fault_id])
        mask = get_color_pallete(mask)    
        mask.save(f"./faulty_output_dataset/{'c' if label == 1 else 'nc'}/fm{indice}_{frame_id}_{fault_id}.png")
        print(f"Immagine {indice} salvata")

    os.makedirs('./faulty_output_dataset', exist_ok=True)
    os.makedirs('./faulty_output_dataset/c', exist_ok=True)
    os.makedirs('./faulty_output_dataset/nc', exist_ok=True)

    for i in tqdm(range(len(df))):
    # for i in tqdm(range(1053, len(df))):
        save_image(i)