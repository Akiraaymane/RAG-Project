import os
from pypdf import PdfReader

class DocumentLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_documents(self):
        loaded_docs = []

        for filename in os.listdir(self.data_path):
            if filename.lower().endswith(".pdf"):
                full_path = os.path.join(self.data_path, filename)

                reader = PdfReader(full_path)

                full_text = ""
                for page in reader.pages:
                    text = page.extract_text() or ""
                    full_text += text + "\n"

                doc = {
                    "text": full_text,
                    "metadata": {
                        "filename": filename,
                        "path": full_path,
                        "filetype": "pdf"
                    }
                }

                loaded_docs.append(doc)

        return loaded_docs





# Pseudocode Representation:

# Class DocumentLoader:
#     Function __init__(data_path):
#         Store the folder path where PDFs are located

#     Function load_documents():
#         Create an empty list called loaded_docs

#         For each file in the folder:
#             If the file ends with ".pdf":
#                 Get the full path of the file

#                 Open the PDF file using a PDF reader

#                 Create an empty string called full_text

#                 For each page in the PDF:
#                     Extract the text from the page
#                     Add the text to full_text with a newline

#                 Create a dictionary called doc:
#                     text -> full_text
#                     metadata -> dictionary with filename, path, and filetype

#                 Add doc to loaded_docs

#         Return the list loaded_docs
