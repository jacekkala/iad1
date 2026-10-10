# font_generator.py
import sys
import os
import random
from PIL import Image, ImageDraw, ImageFont

def main():
    if len(sys.argv) != 9:
        print("Użycie: python font_generator.py w h x y font_file letter noise_level output_directory")
        sys.exit(1)

    # parametry
    w = int(sys.argv[1])
    h = int(sys.argv[2])
    x = int(sys.argv[3])
    y = int(sys.argv[4])
    font_file = sys.argv[5]
    letter = sys.argv[6]
    noise_level = int(sys.argv[7])
    output_directory = sys.argv[8]

    os.makedirs(output_directory, exist_ok=True)

    # białe tło
    image = Image.new('L', (w, h), color=255)
    draw = ImageDraw.Draw(image)

    try:
        # czcionka
        font = ImageFont.truetype(font_file, int(h * 0.7)) 
    except IOError:
        print(f"Błąd: Nie można załadować pliku czcionki {font_file}.")
        sys.exit(1)

    # rysowanie litery
    draw.text((x, y), letter, fill=0, font=font)

    # dodawanie szumu
    pixels = image.load()
    for i in range(w):
        for j in range(h):
            if random.random() < (noise_level / 100.0):
                # odwrócenie koloru piksela (0 -> 255, 255 -> 0)
                pixels[i, j] = 255 - pixels[i, j]

    output_image_path = os.path.join(output_directory, f"{letter}.png")
    image.save(output_image_path)

    description_path = os.path.join(output_directory, "description.txt")
    description_line = f"{letter}.png:letter {letter}, noise level {noise_level}%\n"
    
    with open(description_path, 'a', encoding='utf-8') as f:
        f.write(description_line)

if __name__ == "__main__":
    main()