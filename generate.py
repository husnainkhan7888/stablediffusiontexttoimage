from model import load_model

def generate_image(prompt):
    pipeline = load_model()
    image = pipeline(prompt).images[0]
    return image

if __name__ == "__main__":
    prompt = "A futuristic cityscape at sunset"
    image = generate_image(prompt)
    image.save("output.png")  # Save the generated image
    print("Image saved as output.png")
