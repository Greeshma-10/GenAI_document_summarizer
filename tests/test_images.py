import fitz

file_path = "image_test.pdf"

doc = fitz.open(file_path)

total_images = 0

for page_index in range(len(doc)):
    page = doc.load_page(page_index)

    images = page.get_images(full=True)

    print(f"\nPage {page_index+1} → {len(images)} images found")

    for img_index, img in enumerate(images):

        xref = img[0]
        base_image = doc.extract_image(xref)

        image_bytes = base_image["image"]
        image_ext = base_image["ext"]

        filename = f"extracted_page{page_index+1}_{img_index}.{image_ext}"

        with open(filename, "wb") as f:
            f.write(image_bytes)

        print("Saved:", filename)

    total_images += len(images)

print("\nTotal images extracted:", total_images)