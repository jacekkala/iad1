# madaline_ocr.py
import sys
import os
import numpy as np
from PIL import Image

def load_data(directory):
    description_path = os.path.join(directory, 'description.txt')
    if not os.path.exists(description_path):
        print(f"Błąd: Brak pliku {description_path}")
        sys.exit(1)

    data = []
    with open(description_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            filename, label = line.split(':', 1)
            filepath = os.path.join(directory, filename)
            
            img = Image.open(filepath).convert('L')
            img_array = np.array(img)
            
            # konwersja do postaci binarnej i wektoryzacja
            unipolar_vector = np.where(img_array < 128, 1.0, 0.0).flatten()
            
            # normalizacja
            norm = np.linalg.norm(unipolar_vector)
            if norm > 0:
                normalized_vector = unipolar_vector / norm
            else:
                normalized_vector = unipolar_vector
                
            data.append((normalized_vector, label))
    return data

def main():
    if len(sys.argv) != 3:
        print("Użycie: python madaline_ocr.py train_directory test_directory")
        sys.exit(1)

    train_directory = sys.argv[1]
    test_directory = sys.argv[2]

    training_data = load_data(train_directory)
    if not training_data:
        print("Brak danych treningowych.")
        sys.exit(1)
        
    # wagi i etykiety z danych treningowych
    weights = [item[0] for item in training_data]
    train_labels = [item[1] for item in training_data]

    test_data = load_data(test_directory)

    for test_vector, test_label in test_data:
        max_cnf = -2.0
        winner_idx = -1
        
        for i, weight_vector in enumerate(weights):
            # cos(w, x)
            cnf = np.dot(weight_vector, test_vector)
            
            if cnf > max_cnf:
                max_cnf = cnf
                winner_idx = i
                
        winner_label = train_labels[winner_idx]
        print(f"{test_label} -> {winner_label}, confidence: {max_cnf:.3f}")

if __name__ == "__main__":
    main()