import os
import fitz
from PIL import Image
import io
import base64
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

def image_extract(filepath, output_dir):
    """Extract images from PDF and save them to output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the PDF
    pdf_document = fitz.open(filepath)
    
    # Get the filename without extension
    filename = os.path.splitext(os.path.basename(filepath))[0]
    
    image_paths = []
    
    # Iterate through each page
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        
        # Get all images on the page
        image_list = page.get_images(full=True)
        
        # Iterate through images
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Get image extension
            ext = base_image["ext"]
            
            # Create image path
            image_path = os.path.join(output_dir, f"{filename}_page{page_num+1}_img{img_index+1}.{ext}")
            
            # Save image
            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)
            
            image_paths.append(image_path)
    
    pdf_document.close()
    
    return image_paths

def encode_image(image_path):
    """Encode image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def image_summary_doc(directory_path, vision_llm):
    """Create documents with image summaries."""
    image_documents = []
    
    # Get all image files in the directory
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]
    
    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        
        # Encode the image
        encoded_image = encode_image(image_path)
        
        # Create a message with the image
        # message = [
        #     {
        #         "type": "text",
        #         "text": "Describe this image in detail and extract any text visible in it."
        #     },
        #     {
        #         "type": "image_url",
        #         "image_url": {
        #             "url": f"data:image/jpeg;base64,{encoded_image}"
        #         }
        #     }
        # ]
        
        message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "Describe this image in detail and extract any text visible in it.",
        },
        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{encoded_image}"},
    ]
)

        
        try:
            # Get image description from vision model
            response = vision_llm.invoke([message])
            image_description = response.content
            
            # Create a document with the image description
            doc = Document(
                page_content=image_description,
                metadata={
                    "source": image_path,
                    "type": "image",
                    "filename": image_file
                }
            )
            
            image_documents.append(doc)
        except Exception as e:
            print(f"Error processing image {image_file}: {str(e)}")
    
    return image_documents
