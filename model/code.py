import bnn
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from IPython.display import display
from scipy.ndimage import sobel

# Initialize classifier once
hw_classifier = bnn.CnvClassifier(bnn.NETWORK_CNVW1A1, "cifar10", bnn.RUNTIME_HW)

def detect_edges(image):
    gray = image.convert("L")  # Convert to grayscale
    img_array = np.array(gray)

    # Apply Sobel edge detection
    dx = sobel(img_array, axis=0)
    dy = sobel(img_array, axis=1)
    mag = np.hypot(dx, dy)  # Magnitude of gradient
    mag = (mag / mag.max()) * 255  # Normalize to 0-255
    return Image.fromarray(mag.astype(np.uint8))

def classify_image(image_path):
    img_original = Image.open(image_path).convert("RGB")
    img_original.thumbnail((512, 512), Image.ANTIALIAS)
    img_for_display = img_original.copy()

    # Resize for classification
    img_resized = img_original.resize((32, 32))
    class_scores = hw_classifier.classify_image_details(img_resized)

    # Top-3 class scores
    top_indices = np.argsort(class_scores)[-3:][::-1]
    label_text = "\n".join([
        f"{hw_classifier.class_name(i)}: {class_scores[i]:.4f}" for i in top_indices
    ])

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(img_for_display)
    text_w, text_h = draw.multiline_textsize(label_text, font=font, spacing=4)
    padding = 10
    x = img_for_display.width - text_w - padding
    y = img_for_display.height - text_h - padding

    overlay = Image.new('RGBA', img_for_display.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([x - padding, y - padding, x + text_w + padding, y + text_h + padding], fill=(0, 0, 0, 160))

    img_rgba = img_for_display.convert("RGBA")
    img_with_overlay = Image.alpha_composite(img_rgba, overlay)
    final_draw = ImageDraw.Draw(img_with_overlay)
    final_draw.multiline_text((x, y), label_text, font=font, fill=(255, 255, 255, 255), spacing=4)

    img_final = img_with_overlay.convert("RGB")

    # Show edge detection result
    edge_img = detect_edges(img_original).convert("RGB")
    display(edge_img)
    print("🟡 Edge detection output (grayscale shown above)")

    # Show classified image
    display(img_final)
    print("🟢 Classified image shown above")

    # Optionally save
    img_final.save("classified_output.jpg")
    edge_img.save("edges_output.jpg")

# Example usage
classify_image("dog.jpeg")