from PIL import Image


def load_image_as_rgb_matrix(path):
    with Image.open(path) as img:
        img = img.convert("RGB")  # Converte para modo RGB se não estiver
        width, height = img.size
        pixels = img.load()

        rgb_matrix = []
        for y in range(height):
            row = []
            for x in range(width):
                row.append(pixels[x, y])  # Adiciona o pixel (R, G, B)
            rgb_matrix.append(row)

        return rgb_matrix


def rgb_to_grayscale(image_pixels):
    height = len(image_pixels)
    width = len(image_pixels[0]) if height > 0 else 0

    grayscale_image = []

    for y in range(height):
        row = []
        for x in range(width):
            r, g, b = image_pixels[y][x]

            # Converte usando ponderações perceptuais
            gray = int(0.299 * r + 0.587 * g + 0.114 * b)
            row.append(gray)
        grayscale_image.append(row)

    return grayscale_image


def grayscale_to_binary(grayscale_image, threshold=128):
    """
    Converte uma imagem em tons de cinza para imagem binária (preto e branco).

    Parâmetros:
    - grayscale_image: matriz 2D de tons de cinza (0–255)
    - threshold: valor de corte. Pixels >= threshold viram branco (255); caso contrário, preto (0)

    Retorno:
    - matriz 2D com valores 0 ou 255
    """
    height = len(grayscale_image)
    width = len(grayscale_image[0]) if height > 0 else 0

    binary_image = []

    for y in range(height):
        row = []
        for x in range(width):
            pixel = grayscale_image[y][x]
            binary_value = 255 if pixel >= threshold else 0
            row.append(binary_value)
        binary_image.append(row)

    return binary_image


def show_image(image_matrix, mode="RGB"):
    """
    Exibe uma matriz de imagem (RGB, L ou binária) em uma janela do sistema.

    Parâmetros:
    - image_matrix: matriz 2D de pixels
    - mode: modo de cor da imagem ("RGB", "L" ou "1")
    """
    height = len(image_matrix)
    width = len(image_matrix[0]) if height > 0 else 0

    img = Image.new(mode, (width, height))

    # Achata a matriz para uma lista 1D
    flat_pixels = [pixel for row in image_matrix for pixel in row]

    img.putdata(flat_pixels)

    img.show()
