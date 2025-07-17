from image_tools import (
    load_image_as_rgb_matrix,
    rgb_to_grayscale,
    grayscale_to_binary,
    show_image
)

if __name__ == "__main__":
    caminho = "imagem.png"

    # Carrega imagem colorida
    rgb = load_image_as_rgb_matrix(caminho)

    # Converte para tons de cinza
    grayscale = rgb_to_grayscale(rgb)
    show_image(grayscale, mode="L")  # Exibe a imagem em tons de cinza

    # Converte para imagem binária (preto/branco)
    binaria = grayscale_to_binary(grayscale, threshold=128)
    show_image(binaria, mode="L")  # Também usamos "L", pois são tons (0 ou 255)
